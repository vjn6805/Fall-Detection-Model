import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import sys
import cv2

sys.path.insert(0, ".")

from app.config import load_config
from app.detector import PersonDetector
from app.video import open_source, read_frame
from app.state import StateManager
from app.event_log import setup_event_logger
from app.camera_registry import CameraRegistry
from app.incident_manager import IncidentManager
from app.visualization import draw_persons, draw_hud, draw_emergency_banner, draw_incident_panel


def run():
    try:
        cfg = load_config("config.yaml")
    except FileNotFoundError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    try:
        detector = PersonDetector(
            model_name=cfg["model"]["name"],
            confidence=cfg["model"]["confidence"],
            input_size=cfg["model"]["input_size"],
            tracker=cfg["tracking"]["tracker"],
        )
    except RuntimeError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    log_cfg = cfg.get("logging", {})
    camera_id = cfg.get("camera", {}).get("id", "CAM-001")
    registry = CameraRegistry(cfg.get("camera", {}).get("registry", "cameras.yaml"))
    incident_mgr = IncidentManager(
        db_path=cfg.get("incident", {}).get("db_path", "outputs/incidents.db"),
        camera_registry=registry,
        snapshot_dir=cfg.get("evidence", {}).get("snapshot_dir", "outputs/snapshots"),
    )

    state_mgr = StateManager(
        max_frames=cfg["history"]["max_frames"],
        expiry_seconds=cfg["history"]["expiry_seconds"],
        pose_conf_threshold=cfg["tracking"]["pose_conf_threshold"],
        fall_cfg=cfg.get("fall_detection"),
        risk_cfg=cfg.get("risk_engine"),
        camera_id=camera_id,
        logger=setup_event_logger(log_cfg.get("event_log", "outputs/events.log")),
        incident_store=incident_mgr,
    )

    try:
        cap, source_label = open_source(cfg)
    except RuntimeError as e:
        print(f"[Error] {e}")
        sys.exit(1)

    debug = cfg["display"].get("debug", False)
    print(f"[Source] {source_label}")
    print(f"[Info] Debug mode: {'ON' if debug else 'OFF'}")
    print("[Info] Press 'q' to quit.")

    fps = 0.0
    pose_ms = 0.0
    frame_times = []

    while True:
        t_start = time.perf_counter()

        frame, ok = read_frame(cap)
        if not ok:
            print("[Info] Stream ended or frame unreadable.")
            break

        try:
            t_pose = time.perf_counter()
            persons = detector.track(frame)
            pose_ms = (time.perf_counter() - t_pose) * 1000
        except Exception as e:
            print(f"[Warning] Inference error: {e}")
            persons = []

        now = time.perf_counter()
        states = state_mgr.update(persons, now, frame=frame)

        draw_emergency_banner(frame, states)
        draw_persons(frame, states, debug=debug)
        draw_incident_panel(frame, states)
        draw_hud(frame, fps, detector.device, len(states), pose_ms)

        cv2.imshow("CCTV Detection — Phase 5", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[Info] Quit requested.")
            break

        t_end = time.perf_counter()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

    cap.release()
    cv2.destroyAllWindows()

    if frame_times:
        avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
        avg_ms = (sum(frame_times) / len(frame_times)) * 1000
        print(f"[Perf] Avg FPS: {avg_fps:.1f} | Avg latency: {avg_ms:.1f}ms | Device: {detector.device.upper()}")


if __name__ == "__main__":
    run()
