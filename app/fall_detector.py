"""
fall_detector.py — Temporal fall / collapse detection state machine.

SCORING FORMULA
---------------
fall_score = (
    0.35 * posture_score       +   # current body orientation
    0.30 * transition_score    +   # upright→horizontal transition evidence
    0.20 * persistence_score   +   # how long low posture has been held
    0.15 * motion_score            # post-fall stillness
)

transition_score is penalised when the transition is slow/controlled
(sitting down, lying down intentionally) vs rapid/uncontrolled (fall).

STATES
------
NORMAL
  → POSSIBLE_FALL          (fall_score >= threshold for N frames)
  → UNKNOWN_LOW_POSTURE    (person is horizontal but no fall transition observed)

POSSIBLE_FALL
  → FALL_CONFIRMED         (persists for confirm_fall_seconds)
  → NORMAL                 (score drops — stumble/recovery)

FALL_CONFIRMED
  → POST_FALL_MONITORING

POST_FALL_MONITORING
  → RECOVERED              (torso returns upright for recovery_upright_seconds)

RECOVERED
  → NORMAL                 (after stabilisation)

UNKNOWN_LOW_POSTURE
  → NORMAL                 (person stands up for unknown_low_posture_upright_seconds)
  → POSSIBLE_FALL          (a fall transition becomes observable while in this state)

OBSERVABILITY
-------------
HIGH   — pose_conf >= obs_high AND visible_kps >= min_visible_keypoints
MEDIUM — pose_conf >= obs_low  AND visible_kps >= 3
LOW    — below MEDIUM thresholds

In LOW observability: state machine does not advance toward FALL_CONFIRMED.
In MEDIUM: advances but with reduced transition score weight.

IMPORTANT: This is a prototype computer-vision system.
It detects POSSIBLE FALL / POSSIBLE COLLAPSE based on visual signals only.
It is NOT a medical diagnosis tool.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FallState(Enum):
    NORMAL               = "NORMAL"
    POSSIBLE_FALL        = "POSSIBLE_FALL"
    FALL_CONFIRMED       = "FALL_CONFIRMED"
    POST_FALL_MONITORING = "POST_FALL_MONITORING"
    RECOVERED            = "RECOVERED"
    UNKNOWN_LOW_POSTURE  = "UNKNOWN_LOW_POSTURE"   # horizontal, no observed fall


@dataclass
class FallResult:
    state: FallState = FallState.NORMAL
    fall_score: float = 0.0
    posture_score: float = 0.0
    transition_score: float = 0.0
    persistence_score: float = 0.0
    motion_score: float = 0.0
    pose_quality: float = 0.0
    observability: str = "LOW"
    transition_reason: str = ""        # machine-readable reason for last transition
    state_entered_at: float = field(default_factory=time.time)
    possible_fall_frames: int = 0
    recovery_start_time: Optional[float] = None

    @property
    def state_duration(self) -> float:
        return time.time() - self.state_entered_at


class FallDetector:
    """
    Per-person fall detection state machine.
    One instance per tracked person — completely independent.
    """

    def __init__(self, cfg: dict):
        fd = cfg["fall_detection"]
        self.min_pose_conf          = fd["min_pose_confidence"]
        self.min_track_age          = fd["min_track_age"]
        self.horiz_angle            = fd["horizontal_torso_angle"]
        self.upright_angle          = fd["upright_torso_angle"]
        self.horiz_ar               = fd["horizontal_aspect_ratio"]
        self.fall_vel_thresh        = fd["fall_velocity_threshold"]
        self.transition_window      = fd["transition_window_frames"]
        self.possible_frames_req    = fd["possible_fall_frames"]
        self.confirm_seconds        = fd["confirm_fall_seconds"]
        self.post_fall_seconds      = fd["post_fall_duration_seconds"]
        self.recovery_seconds       = fd["recovery_upright_seconds"]
        self.stillness_thresh       = fd["stillness_threshold"]
        # Phase 4A additions
        self.controlled_trans_s     = fd.get("controlled_transition_seconds", 1.2)
        self.min_upright_before     = fd.get("min_upright_frames_before_fall", 6)
        self.obs_high               = fd.get("observability_high_threshold", 0.55)
        self.obs_low                = fd.get("observability_low_threshold", 0.30)
        self.unknown_upright_s      = fd.get("unknown_low_posture_upright_seconds", 1.5)

        self.result = FallResult()

    def update(self, state, logger=None) -> FallResult:
        snap = state.latest
        if snap is None:
            return self.result

        # Gate: not enough track history
        if state.track_age < self.min_track_age:
            return self.result

        # Gate: pose quality too low
        if snap.pose_confidence < self.min_pose_conf:
            self.result.pose_quality = snap.pose_confidence
            self.result.observability = snap.observability
            return self.result

        history = list(state.history)

        posture_score     = self._posture_score(snap)
        transition_score  = self._transition_score(history)
        persistence_score = self._persistence_score(history)
        motion_score      = self._motion_score(history)

        # Observability modulates transition score weight
        obs = snap.observability
        obs_weight = 1.0 if obs == "HIGH" else (0.6 if obs == "MEDIUM" else 0.2)
        effective_transition = transition_score * obs_weight

        fall_score = (
            0.35 * posture_score      +
            0.30 * effective_transition +
            0.20 * persistence_score  +
            0.15 * motion_score
        )

        self.result.posture_score     = round(posture_score, 3)
        self.result.transition_score  = round(transition_score, 3)
        self.result.persistence_score = round(persistence_score, 3)
        self.result.motion_score      = round(motion_score, 3)
        self.result.fall_score        = round(fall_score, 3)
        self.result.pose_quality      = round(snap.pose_confidence, 3)
        self.result.observability     = obs

        self._advance_state(snap, fall_score, history, logger, state.person_id)
        return self.result

    # ------------------------------------------------------------------
    # Sub-score calculations
    # ------------------------------------------------------------------

    def _posture_score(self, snap) -> float:
        """
        Current body orientation score.
        0.0 = clearly upright, 1.0 = clearly horizontal.
        Torso angle weighted more than aspect ratio.
        """
        angle_score = 0.0
        if snap.torso_angle is not None:
            lo, hi = self.upright_angle, self.horiz_angle
            angle_score = max(0.0, min(1.0, (snap.torso_angle - lo) / (hi - lo + 1e-6)))

        ar_score = 0.0
        if snap.aspect_ratio > 0:
            ar_score = max(0.0, min(1.0, (snap.aspect_ratio - 0.8) / (self.horiz_ar - 0.8 + 1e-6)))

        return 0.65 * angle_score + 0.35 * ar_score

    def _transition_score(self, history: list) -> float:
        """
        Upright→horizontal transition score.

        Phase 4A improvement: penalise SLOW transitions.
        A genuine fall completes in < 0.5s.
        Sitting down or lying intentionally takes > 1s.
        If the transition duration exceeds controlled_transition_seconds,
        the score is scaled down proportionally.

        Also gates on min_upright_frames_before_fall: if the person was
        NOT upright before the window, no transition score is awarded
        (handles person entering frame already lying down).
        """
        window = history[-self.transition_window:]
        if len(window) < 4:
            return 0.0

        mid = len(window) // 2
        early, late = window[:mid], window[mid:]

        early_angles = [s.torso_angle for s in early if s.torso_angle is not None]
        late_angles  = [s.torso_angle for s in late  if s.torso_angle is not None]

        if not early_angles or not late_angles:
            return 0.0

        early_min = min(early_angles)
        late_max  = max(late_angles)
        delta = late_max - early_min

        # Gate: person must have been upright before the transition
        upright_count = sum(
            1 for s in early
            if s.torso_angle is not None and s.torso_angle < self.upright_angle + 10
        )
        if upright_count < self.min_upright_before // 2:
            # Not enough upright history — likely entered already low
            return 0.0

        angle_score = max(0.0, min(1.0, delta / 40.0))
        vel_score   = self._velocity_score(window)
        raw_score   = max(angle_score, vel_score)

        if raw_score < 0.1:
            return 0.0

        # Phase 4A: penalise slow/controlled transitions
        # Find the first frame where angle crossed horiz_angle
        transition_start = None
        transition_end   = None
        for s in window:
            if s.torso_angle is not None:
                if s.torso_angle < self.horiz_angle and transition_start is None:
                    transition_start = s.timestamp
                if s.torso_angle >= self.horiz_angle:
                    transition_end = s.timestamp

        if transition_start is not None and transition_end is not None:
            duration = max(transition_end - transition_start, 1e-3)
            # Fast fall: duration << controlled_trans_s → penalty near 0
            # Slow sit: duration >> controlled_trans_s → heavy penalty
            speed_factor = max(0.1, min(1.0, self.controlled_trans_s / duration))
            raw_score *= speed_factor

        return round(raw_score, 3)

    def _velocity_score(self, window: list) -> float:
        """
        Rapid downward movement score.
        Velocity normalised by box height (resolution-independent).
        Only peak velocity matters — a single fast frame is enough.
        """
        if len(window) < 3:
            return 0.0

        velocities = []
        for i in range(1, len(window)):
            prev, curr = window[i - 1], window[i]
            if curr.height == 0:
                continue
            dy = curr.center[1] - prev.center[1]
            dt = max(curr.timestamp - prev.timestamp, 1e-3)
            vel = (dy / curr.height) / dt
            velocities.append(vel)

        if not velocities:
            return 0.0

        return max(0.0, min(1.0, max(velocities) / self.fall_vel_thresh))

    def _persistence_score(self, history: list) -> float:
        """
        Fraction of recent frames showing low/horizontal posture.
        Builds slowly — a brief bend scores low, prolonged low posture scores high.
        """
        window = history[-self.transition_window:]
        if not window:
            return 0.0
        low_count = sum(
            1 for s in window
            if s.torso_angle is not None and s.torso_angle > self.horiz_angle
        )
        return low_count / len(window)

    def _motion_score(self, history: list) -> float:
        """
        Post-event stillness score.
        High = not moving (consistent with collapse).
        Low = moving (consistent with intentional activity / recovery).
        """
        recent = history[-8:]
        if len(recent) < 3:
            return 0.0
        deltas = [s.position_delta for s in recent if s.position_delta is not None]
        if not deltas:
            return 0.0
        avg_delta = sum(deltas) / len(deltas)
        return max(0.0, min(1.0, 1.0 - avg_delta / (self.stillness_thresh * 2)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_currently_low(self, snap) -> bool:
        return (snap.torso_angle is not None and snap.torso_angle > self.horiz_angle) \
            or snap.aspect_ratio > self.horiz_ar

    def _is_currently_upright(self, snap) -> bool:
        return snap.torso_angle is not None and snap.torso_angle < self.upright_angle

    def _had_upright_history(self, history: list) -> bool:
        """True if person was upright for at least min_upright_before frames recently."""
        window = history[-self.transition_window:]
        upright = sum(
            1 for s in window
            if s.torso_angle is not None and s.torso_angle < self.upright_angle + 10
        )
        return upright >= self.min_upright_before

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _advance_state(self, snap, fall_score: float, history: list, logger, pid: int):
        now = time.time()

        if self.result.state == FallState.NORMAL:
            if self._is_currently_low(snap):
                if self._had_upright_history(history) and fall_score >= 0.55:
                    # Upright history + high score → possible fall
                    self.result.possible_fall_frames += 1
                    if self.result.possible_fall_frames >= self.possible_frames_req:
                        self._transition(FallState.POSSIBLE_FALL, logger, pid,
                                         "rapid_downward_motion+posture_transition")
                elif not self._had_upright_history(history):
                    # No upright history → unknown, don't call it a fall
                    self._transition(FallState.UNKNOWN_LOW_POSTURE, logger, pid,
                                     "low_posture_without_observed_fall")
                else:
                    self.result.possible_fall_frames = 0
            else:
                self.result.possible_fall_frames = 0

        elif self.result.state == FallState.POSSIBLE_FALL:
            if fall_score < 0.45:
                self._transition(FallState.NORMAL, logger, pid, "score_dropped_stumble_recovery")
                self.result.possible_fall_frames = 0
            elif snap.observability == "LOW":
                # Can't confirm with low observability — hold in POSSIBLE_FALL
                pass
            elif self.result.state_duration >= self.confirm_seconds:
                self._transition(FallState.FALL_CONFIRMED, logger, pid,
                                 "fall_persisted_confirm_duration")

        elif self.result.state == FallState.FALL_CONFIRMED:
            self._transition(FallState.POST_FALL_MONITORING, logger, pid,
                             "entering_post_fall_monitoring")

        elif self.result.state == FallState.POST_FALL_MONITORING:
            if self._is_currently_upright(snap):
                if self.result.recovery_start_time is None:
                    self.result.recovery_start_time = now
                elif (now - self.result.recovery_start_time) >= self.recovery_seconds:
                    self._transition(FallState.RECOVERED, logger, pid,
                                     "upright_posture_restored")
            else:
                self.result.recovery_start_time = None

        elif self.result.state == FallState.RECOVERED:
            if self.result.state_duration >= self.recovery_seconds:
                self._transition(FallState.NORMAL, logger, pid, "recovery_stabilised")
                self.result.possible_fall_frames = 0
                self.result.recovery_start_time = None

        elif self.result.state == FallState.UNKNOWN_LOW_POSTURE:
            if self._is_currently_upright(snap):
                if self.result.recovery_start_time is None:
                    self.result.recovery_start_time = now
                elif (now - self.result.recovery_start_time) >= self.unknown_upright_s:
                    self._transition(FallState.NORMAL, logger, pid,
                                     "stood_up_from_unknown_low_posture")
                    self.result.recovery_start_time = None
            else:
                self.result.recovery_start_time = None
                # If a fall transition becomes observable while in this state
                if self._had_upright_history(history) and fall_score >= 0.55:
                    self.result.possible_fall_frames += 1
                    if self.result.possible_fall_frames >= self.possible_frames_req:
                        self._transition(FallState.POSSIBLE_FALL, logger, pid,
                                         "fall_transition_observed_from_unknown_state")

    def _transition(self, new_state: FallState, logger, pid: int, reason: str = ""):
        from app.event_log import log_transition
        prev = self.result.state
        self.result.state = new_state
        self.result.state_entered_at = time.time()
        self.result.transition_reason = reason

        if logger is not None:
            log_transition(logger, pid, prev.value, new_state.value, {
                "fall":       self.result.fall_score,
                "posture":    self.result.posture_score,
                "transition": self.result.transition_score,
                "persist":    self.result.persistence_score,
                "motion":     self.result.motion_score,
                "pose_q":     self.result.pose_quality,
                "obs":        self.result.observability,
                "reason":     reason,
            })
