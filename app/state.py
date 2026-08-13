import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from app.fall_detector import FallDetector, FallResult


@dataclass
class FrameSnapshot:
    timestamp: float
    xyxy: np.ndarray
    center: tuple          # (cx, cy)
    width: int
    height: int
    aspect_ratio: float
    confidence: float
    keypoints: Optional[np.ndarray]   # (17, 2) or None
    kp_conf: Optional[np.ndarray]     # (17,)  or None
    pose_confidence: float            # mean of visible kp confidences
    pose_quality: str                 # "OK" | "LOW QUALITY"
    shoulder_center: Optional[tuple]
    hip_center: Optional[tuple]
    torso_angle: Optional[float]      # degrees from vertical
    position_delta: Optional[float]   # pixel distance from previous frame
    observability: str = "LOW"        # HIGH | MEDIUM | LOW
    visible_keypoints: int = 0        # count of keypoints above threshold


@dataclass
class PersonState:
    person_id: int
    track_age: int = 0
    last_seen: float = field(default_factory=time.time)
    history: deque = field(default_factory=deque)
    fall_detector: Optional[object] = field(default=None, repr=False)
    fall_result: Optional[object] = field(default=None, repr=False)
    risk_engine: Optional[object] = field(default=None, repr=False)
    risk_result: Optional[object] = field(default=None, repr=False)

    def update(self, snapshot: FrameSnapshot, max_frames: int):
        self.track_age += 1
        self.last_seen = snapshot.timestamp
        self.history.append(snapshot)
        if len(self.history) > max_frames:
            self.history.popleft()

    @property
    def latest(self) -> Optional[FrameSnapshot]:
        return self.history[-1] if self.history else None


class StateManager:
    def __init__(self, max_frames: int, expiry_seconds: float, pose_conf_threshold: float,
                 fall_cfg: dict = None, risk_cfg: dict = None,
                 camera_id: str = "CAMERA_01", logger=None, incident_store=None):
        self.max_frames = max_frames
        self.expiry_seconds = expiry_seconds
        self.pose_conf_threshold = pose_conf_threshold
        self.fall_cfg = fall_cfg
        self.risk_cfg = risk_cfg
        self.camera_id = camera_id
        self.logger = logger
        self.incident_store = incident_store
        self._states: dict[int, PersonState] = {}
        self._current_frame = None

    def update(self, persons: list, timestamp: float, frame=None):
        self._current_frame = frame
        seen_ids = set()

        for p in persons:
            pid = p["track_id"]
            seen_ids.add(pid)

            if pid not in self._states:
                from app.fall_detector import FallDetector, FallResult
                from app.risk_engine import RiskEngine, RiskResult
                ps = PersonState(person_id=pid)
                if self.fall_cfg:
                    ps.fall_detector = FallDetector({"fall_detection": self.fall_cfg})
                    ps.fall_result = FallResult()
                if self.risk_cfg:
                    ps.risk_engine = RiskEngine(
                        {"risk_engine": self.risk_cfg}, camera_id=self.camera_id
                    )
                    ps.risk_result = RiskResult()
                self._states[pid] = ps

            state = self._states[pid]
            prev = state.latest

            x1, y1, x2, y2 = p["xyxy"]
            w = int(x2 - x1)
            h = int(y2 - y1)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            aspect_ratio = round(w / h, 3) if h > 0 else 0.0

            kps = p["keypoints"]
            kp_conf = p["kp_conf"]

            pose_conf, pose_quality, shoulder_center, hip_center, torso_angle, \
                observability, visible_kps = \
                _extract_pose_features(kps, kp_conf, self.pose_conf_threshold, self.fall_cfg)

            position_delta = None
            if prev is not None:
                dx = cx - prev.center[0]
                dy = cy - prev.center[1]
                position_delta = round(math.sqrt(dx * dx + dy * dy), 2)

            snap = FrameSnapshot(
                timestamp=timestamp,
                xyxy=p["xyxy"],
                center=(cx, cy),
                width=w,
                height=h,
                aspect_ratio=aspect_ratio,
                confidence=p["confidence"],
                keypoints=kps,
                kp_conf=kp_conf,
                pose_confidence=pose_conf,
                pose_quality=pose_quality,
                shoulder_center=shoulder_center,
                hip_center=hip_center,
                torso_angle=torso_angle,
                position_delta=position_delta,
                observability=observability,
                visible_keypoints=visible_kps,
            )
            state.update(snap, self.max_frames)

            # Run fall detection after history is updated
            if state.fall_detector is not None:
                state.fall_result = state.fall_detector.update(state, self.logger)

            # Run risk engine after fall detection
            if state.risk_engine is not None and state.fall_result is not None:
                prev_risk_state = state.risk_result.state.value if state.risk_result else "NORMAL"
                state.risk_result = state.risk_engine.update(
                    state, state.fall_result, self.logger
                )
                # Incident management
                if self.incident_store is not None:
                    rs = state.risk_result.state.value
                    if rs in ("POTENTIAL_HEALTH_EMERGENCY", "POSSIBLE_EMERGENCY",
                              "MONITORING", "POSSIBLE_FALL"):
                        state.risk_result.incident_id = self.incident_store.ingest(
                            self.camera_id, pid,
                            state.fall_result, state.risk_result,
                            self._current_frame,
                        )
                    elif rs in ("NORMAL", "RECOVERED") and prev_risk_state not in ("NORMAL", "RECOVERED"):
                        self.incident_store.notify_risk_cleared(self.camera_id, pid)

        self._expire(timestamp)
        return self._states

    def _expire(self, now: float):
        expired = [
            pid for pid, s in self._states.items()
            if (now - s.last_seen) > self.expiry_seconds
        ]
        for pid in expired:
            del self._states[pid]

    def get_all(self) -> dict:
        return self._states


def _extract_pose_features(kps, kp_conf, threshold, fall_cfg=None):
    """Compute pose features. Returns (pose_conf, quality, shoulder_center, hip_center, torso_angle, observability, visible_kps)."""
    if kps is None or kp_conf is None:
        return 0.0, "LOW QUALITY", None, None, None, "LOW", 0

    visible_mask = kp_conf > threshold
    visible_kps = int(visible_mask.sum())
    pose_conf = round(float(kp_conf[visible_mask].mean()) if visible_mask.any() else 0.0, 3)
    quality = "OK" if pose_conf >= threshold else "LOW QUALITY"

    # Observability: combines pose confidence + visible keypoint count
    obs_high = fall_cfg.get("observability_high_threshold", 0.55) if fall_cfg else 0.55
    obs_low  = fall_cfg.get("observability_low_threshold", 0.30) if fall_cfg else 0.30
    min_kps  = fall_cfg.get("min_visible_keypoints", 5) if fall_cfg else 5
    if pose_conf >= obs_high and visible_kps >= min_kps:
        observability = "HIGH"
    elif pose_conf >= obs_low and visible_kps >= 3:
        observability = "MEDIUM"
    else:
        observability = "LOW"

    def _midpoint(i, j):
        if kp_conf[i] > threshold and kp_conf[j] > threshold:
            return (
                round((kps[i][0] + kps[j][0]) / 2, 1),
                round((kps[i][1] + kps[j][1]) / 2, 1),
            )
        return None

    shoulder_center = _midpoint(5, 6)
    hip_center = _midpoint(11, 12)

    torso_angle = None
    if shoulder_center and hip_center:
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        torso_angle = round(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)), 1)

    return pose_conf, quality, shoulder_center, hip_center, torso_angle, observability, visible_kps
