import cv2
import numpy as np
from app.detector import SKELETON

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX
HUD_COLOR = (0, 255, 255)
KP_COLOR  = (0, 200, 255)
SKEL_COLOR = (255, 128, 0)

# Box color priority: risk state overrides fall state
RISK_COLORS = {
    "POTENTIAL_HEALTH_EMERGENCY": (0, 0, 255),      # bright red
    "POSSIBLE_EMERGENCY":         (0, 80, 255),      # red-orange
    "MONITORING":                 (0, 165, 255),     # orange
    "RECOVERED":                  (255, 200, 0),     # cyan-yellow
    "NORMAL":                     (0, 255, 0),       # green
}
FALL_COLORS = {
    "POSSIBLE_FALL":        (0, 165, 255),
    "FALL_CONFIRMED":       (0, 60, 255),
    "POST_FALL_MONITORING": (0, 80, 200),
    "RECOVERED":            (255, 200, 0),
    "NORMAL":               (0, 255, 0),
    "UNKNOWN_LOW_POSTURE":  (180, 180, 0),    # dark yellow — uncertain
}
LOW_QUALITY_COLOR = (80, 80, 80)


def _box_color(fall_result, risk_result, pose_quality):
    if pose_quality != "OK":
        return LOW_QUALITY_COLOR
    if risk_result is not None:
        return RISK_COLORS.get(risk_result.state.value, (0, 255, 0))
    if fall_result is not None:
        return FALL_COLORS.get(fall_result.state.value, (0, 255, 0))
    return (0, 255, 0)


def draw_persons(frame, states: dict, debug: bool = False):
    for pid, state in states.items():
        snap = state.latest
        if snap is None:
            continue

        fr = state.fall_result
        rr = state.risk_result
        color = _box_color(fr, rr, snap.pose_quality)

        x1, y1, x2, y2 = snap.xyxy
        thickness = 3 if (rr and rr.state.value == "POTENTIAL_HEALTH_EMERGENCY") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label: Person ID + highest active state
        state_label = _primary_label(fr, rr)
        label = f"Person #{pid}  {state_label}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, 0.5, (0, 0, 0), 1)

        # Score line below box (only when not NORMAL)
        if rr and rr.state.value != "NORMAL":
            score_line = f"Risk:{rr.health_emergency_score:.2f}  Fall:{fr.fall_score:.2f}  Down:{rr.time_since_fall:.1f}s"
            cv2.putText(frame, score_line, (x1 + 2, y2 + 14), FONT, 0.42, color, 1)
        elif fr and fr.state.value not in ("NORMAL",):
            cv2.putText(frame, f"Fall:{fr.fall_score:.2f}  Pose:{snap.pose_confidence:.2f}",
                        (x1 + 2, y2 + 14), FONT, 0.42, color, 1)

        # Skeleton
        if snap.keypoints is not None and snap.kp_conf is not None:
            _draw_skeleton(frame, snap.keypoints, snap.kp_conf)

        if debug:
            _draw_debug(frame, pid, state, snap, fr, rr)

    return frame


def _primary_label(fr, rr) -> str:
    if rr and rr.state.value != "NORMAL":
        return rr.state.value
    if fr and fr.state.value != "NORMAL":
        return fr.state.value
    return "NORMAL"


def draw_emergency_banner(frame, states: dict):
    """Full-width red banner at top when any person is in emergency state."""
    h, w = frame.shape[:2]
    emergency_persons = [
        pid for pid, s in states.items()
        if s.risk_result and s.risk_result.state.value == "POTENTIAL_HEALTH_EMERGENCY"
    ]
    if not emergency_persons:
        return frame

    ids = ", ".join(f"#{p}" for p in emergency_persons)
    banner = f"  !! POTENTIAL HEALTH EMERGENCY — Person {ids}  !!"
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 200), -1)
    cv2.putText(frame, banner, (8, 25), FONTB, 0.65, (255, 255, 255), 2)
    return frame


def draw_incident_panel(frame, states: dict):
    """Right-side panel listing active incidents with timeline summary."""
    active = [
        (pid, s) for pid, s in states.items()
        if s.risk_result and s.risk_result.state.value not in ("NORMAL",)
    ]
    if not active:
        return frame

    h, w = frame.shape[:2]
    panel_x = w - 230
    y = 50

    cv2.rectangle(frame, (panel_x - 4, 40), (w - 2, min(h - 2, 40 + len(active) * 80 + 10)),
                  (20, 20, 20), -1)
    cv2.putText(frame, "ACTIVE INCIDENTS", (panel_x, y), FONT, 0.45, HUD_COLOR, 1)
    y += 18

    for pid, s in active:
        rr = s.risk_result
        fr = s.fall_result
        color = RISK_COLORS.get(rr.state.value, (200, 200, 200))
        lines = [
            f"Person #{pid}",
            f"  {rr.state.value}",
            f"  Risk:{rr.health_emergency_score:.2f}  Down:{rr.time_since_fall:.1f}s",
            f"  ID:{rr.incident_id or 'N/A'}",
        ]
        for line in lines:
            cv2.putText(frame, line, (panel_x, y), FONT, 0.38, color, 1)
            y += 14
        y += 4

    return frame


def _draw_skeleton(frame, kps: np.ndarray, kp_conf: np.ndarray, conf_thresh: float = 0.3):
    for i, (x, y) in enumerate(kps):
        if kp_conf[i] > conf_thresh and x > 0 and y > 0:
            cv2.circle(frame, (int(x), int(y)), 3, KP_COLOR, -1)
    for i, j in SKELETON:
        if (kp_conf[i] > conf_thresh and kp_conf[j] > conf_thresh
                and kps[i][0] > 0 and kps[j][0] > 0):
            cv2.line(frame,
                     (int(kps[i][0]), int(kps[i][1])),
                     (int(kps[j][0]), int(kps[j][1])),
                     SKEL_COLOR, 2)


def _draw_debug(frame, pid: int, state, snap, fr, rr):
    x1, y1, x2, y2 = snap.xyxy
    lines = [
        f"ID:{pid} age:{state.track_age} hist:{len(state.history)}",
        f"center:{snap.center}",
        f"box:{snap.width}x{snap.height} ar:{snap.aspect_ratio}",
        f"torso:{snap.torso_angle}deg" if snap.torso_angle is not None else "torso:N/A",
        f"pose:{snap.pose_confidence:.2f} kps:{snap.visible_keypoints} [{snap.observability}]",
        f"delta:{snap.position_delta}px" if snap.position_delta is not None else "delta:N/A",
    ]
    if fr:
        lines += [
            "-- fall --",
            f"state:{fr.state.value}",
            f"obs:{fr.observability}",
            f"score:{fr.fall_score:.3f}",
            f"posture:{fr.posture_score:.3f}",
            f"trans:{fr.transition_score:.3f}",
            f"persist:{fr.persistence_score:.3f}",
            f"motion:{fr.motion_score:.3f}",
            f"dur:{fr.state_duration:.1f}s",
            f"reason:{fr.transition_reason}" if fr.transition_reason else "",
        ]
        lines = [l for l in lines if l]  # remove empty
    if rr:
        lines += [
            "-- risk --",
            f"state:{rr.state.value}",
            f"risk:{rr.health_emergency_score:.3f}",
            f"fall_ev:{rr.fall_evidence_score:.3f}",
            f"time_dn:{rr.time_down_score:.3f}",
            f"inact:{rr.inactivity_score:.3f}",
            f"no_rec:{rr.no_recovery_score:.3f}",
            f"down:{rr.time_since_fall:.1f}s",
            f"dur:{rr.state_duration:.1f}s",
        ]

    tx, ty = x2 + 6, y1 + 12
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, FONT, 0.38, 1)
        cv2.rectangle(frame, (tx - 1, ty - th - 2), (tx + tw + 2, ty + 2), (20, 20, 20), -1)
        cv2.putText(frame, line, (tx, ty), FONT, 0.38, (200, 255, 200), 1)
        ty += th + 5


def draw_hud(frame, fps: float, device: str, person_count: int, pose_ms: float):
    lines = [
        f"DEVICE: {device.upper()}",
        f"FPS: {fps:.1f}",
        f"TRACKED: {person_count}",
        f"POSE: {pose_ms:.0f}ms",
    ]
    y = 50  # offset to clear emergency banner
    for line in lines:
        cv2.putText(frame, line, (10, y), FONT, 0.6, HUD_COLOR, 2)
        y += 22
    return frame
