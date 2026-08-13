"""
evaluate.py — Offline validation tool for fall and emergency detection.

Usage:
    python tools/evaluate.py --video videos/fall_no_recovery/test1.mp4
    python tools/evaluate.py --video videos/fall_no_recovery/test1.mp4 --config config.yaml
    python tools/evaluate.py --dir videos/fall_no_recovery/
    python tools/evaluate.py --dir videos/ --recursive

Requires a matching .csv label file alongside each video.
See tools/label_format.md for the label format.

Output:
    Per-video metrics + aggregate summary printed to console.
    Results saved to outputs/eval_results.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import csv
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from app.config import load_config
from app.detector import PersonDetector
from app.state import StateManager
from app.fall_detector import FallState
from app.risk_engine import RiskState


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

FALL_EVENTS     = {"fall_start", "fall_confirmed", "emergency"}
EMERGENCY_EVENTS = {"emergency"}
NEGATIVE_EVENTS  = {"normal", "sitting", "lying", "bending", "exercise"}


def load_labels(csv_path: Path) -> list:
    """Load label CSV. Returns list of (frame, event) sorted by frame."""
    labels = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append((int(row["frame"]), row["event"].strip()))
    return sorted(labels, key=lambda x: x[0])


def get_event_at_frame(labels: list, frame_idx: int) -> str:
    """Return the ground truth event label at a given frame (inherits previous)."""
    current = "normal"
    for frame, event in labels:
        if frame <= frame_idx:
            current = event
        else:
            break
    return current


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------

def evaluate_video(video_path: Path, cfg: dict) -> dict:
    label_path = video_path.with_suffix(".csv")
    if not label_path.exists():
        print(f"  [Skip] No label file: {label_path}")
        return None

    labels = load_labels(label_path)
    end_frame = max(f for f, _ in labels) if labels else 9999

    try:
        detector = PersonDetector(
            model_name=cfg["model"]["name"],
            confidence=cfg["model"]["confidence"],
            input_size=cfg["model"]["input_size"],
            tracker=cfg["tracking"]["tracker"],
        )
    except RuntimeError as e:
        print(f"  [Error] {e}")
        return None

    state_mgr = StateManager(
        max_frames=cfg["history"]["max_frames"],
        expiry_seconds=cfg["history"]["expiry_seconds"],
        pose_conf_threshold=cfg["tracking"]["pose_conf_threshold"],
        fall_cfg=cfg.get("fall_detection"),
        risk_cfg=cfg.get("risk_engine"),
        camera_id=cfg.get("camera", {}).get("id", "CAMERA_01"),
        logger=None,
        incident_store=None,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [Error] Cannot open: {video_path}")
        return None

    # Metrics counters
    fall_tp = fall_fp = fall_tn = fall_fn = 0
    emrg_tp = emrg_fp = emrg_tn = emrg_fn = 0
    frame_idx = 0
    inference_times = []

    # Detection latency: frame where fall was first detected vs ground truth
    gt_fall_frame = next((f for f, e in labels if e == "fall_start"), None)
    detected_fall_frame = None

    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        try:
            persons = detector.track(frame)
        except Exception:
            persons = []
        inference_times.append(time.perf_counter() - t0)

        states = state_mgr.update(persons, time.perf_counter())

        gt_event = get_event_at_frame(labels, frame_idx)
        gt_is_fall      = gt_event in FALL_EVENTS
        gt_is_emergency = gt_event in EMERGENCY_EVENTS
        gt_is_negative  = gt_event in NEGATIVE_EVENTS

        # Determine system output: any person in a fall/emergency state?
        sys_fall = any(
            s.fall_result and s.fall_result.state not in (FallState.NORMAL,)
            for s in states.values()
        )
        sys_emergency = any(
            s.risk_result and s.risk_result.state == RiskState.POTENTIAL_HEALTH_EMERGENCY
            for s in states.values()
        )

        # Track detection latency
        if sys_fall and detected_fall_frame is None:
            detected_fall_frame = frame_idx

        # Fall metrics (only count frames with clear ground truth)
        if gt_is_fall:
            if sys_fall:
                fall_tp += 1
            else:
                fall_fn += 1
        elif gt_is_negative:
            if sys_fall:
                fall_fp += 1
            else:
                fall_tn += 1

        # Emergency metrics
        if gt_is_emergency:
            if sys_emergency:
                emrg_tp += 1
            else:
                emrg_fn += 1
        elif gt_is_negative:
            if sys_emergency:
                emrg_fp += 1
            else:
                emrg_tn += 1

        frame_idx += 1

    cap.release()

    # Compute metrics
    def metrics(tp, fp, tn, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return round(precision, 3), round(recall, 3), round(f1, 3)

    fp, fr, ff = metrics(fall_tp, fall_fp, fall_tn, fall_fn)
    ep, er, ef = metrics(emrg_tp, emrg_fp, emrg_tn, emrg_fn)

    avg_inf_ms = round(sum(inference_times) / len(inference_times) * 1000, 1) if inference_times else 0
    fps = round(1000 / avg_inf_ms, 1) if avg_inf_ms > 0 else 0

    detection_latency = None
    if gt_fall_frame is not None and detected_fall_frame is not None:
        detection_latency = detected_fall_frame - gt_fall_frame  # frames

    result = {
        "video": str(video_path),
        "frames_evaluated": frame_idx,
        "fall": {
            "tp": fall_tp, "fp": fall_fp, "tn": fall_tn, "fn": fall_fn,
            "precision": fp, "recall": fr, "f1": ff,
        },
        "emergency": {
            "tp": emrg_tp, "fp": emrg_fp, "tn": emrg_tn, "fn": emrg_fn,
            "precision": ep, "recall": er, "f1": ef,
        },
        "detection_latency_frames": detection_latency,
        "avg_inference_ms": avg_inf_ms,
        "fps": fps,
    }

    _print_result(result)
    return result


def _print_result(r: dict):
    print(f"\n  Video: {Path(r['video']).name}  ({r['frames_evaluated']} frames)")
    print(f"  Inference: {r['avg_inference_ms']}ms avg  |  {r['fps']} FPS")
    if r["detection_latency_frames"] is not None:
        print(f"  Detection latency: {r['detection_latency_frames']} frames after fall_start")
    f = r["fall"]
    print(f"  Fall     — P:{f['precision']:.3f}  R:{f['recall']:.3f}  F1:{f['f1']:.3f}"
          f"  TP:{f['tp']} FP:{f['fp']} TN:{f['tn']} FN:{f['fn']}")
    e = r["emergency"]
    print(f"  Emergency— P:{e['precision']:.3f}  R:{e['recall']:.3f}  F1:{e['f1']:.3f}"
          f"  TP:{e['tp']} FP:{e['fp']} TN:{e['tn']} FN:{e['fn']}")


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def aggregate(results: list) -> dict:
    valid = [r for r in results if r is not None]
    if not valid:
        return {}

    def avg_metric(key, sub):
        vals = [r[key][sub] for r in valid if r[key][sub] is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "videos_evaluated": len(valid),
        "fall": {
            "precision": avg_metric("fall", "precision"),
            "recall":    avg_metric("fall", "recall"),
            "f1":        avg_metric("fall", "f1"),
            "total_fp":  sum(r["fall"]["fp"] for r in valid),
            "total_fn":  sum(r["fall"]["fn"] for r in valid),
        },
        "emergency": {
            "precision": avg_metric("emergency", "precision"),
            "recall":    avg_metric("emergency", "recall"),
            "f1":        avg_metric("emergency", "f1"),
            "total_fp":  sum(r["emergency"]["fp"] for r in valid),
            "total_fn":  sum(r["emergency"]["fn"] for r in valid),
        },
        "avg_fps": round(sum(r["fps"] for r in valid) / len(valid), 1),
        "avg_inference_ms": round(sum(r["avg_inference_ms"] for r in valid) / len(valid), 1),
    }


def print_summary(agg: dict):
    print("\n" + "=" * 60)
    print("AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"Videos evaluated : {agg['videos_evaluated']}")
    print(f"Avg FPS          : {agg['avg_fps']}")
    print(f"Avg inference    : {agg['avg_inference_ms']}ms")
    f = agg["fall"]
    print(f"\nFall Detection")
    print(f"  Precision : {f['precision']:.3f}")
    print(f"  Recall    : {f['recall']:.3f}")
    print(f"  F1        : {f['f1']:.3f}")
    print(f"  Total FP  : {f['total_fp']}")
    print(f"  Total FN  : {f['total_fn']}")
    e = agg["emergency"]
    print(f"\nEmergency Detection")
    print(f"  Precision : {e['precision']:.3f}")
    print(f"  Recall    : {e['recall']:.3f}")
    print(f"  F1        : {e['f1']:.3f}")
    print(f"  Total FP  : {e['total_fp']}")
    print(f"  Total FN  : {e['total_fn']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def collect_videos(args) -> list:
    videos = []
    if args.video:
        videos.append(Path(args.video))
    elif args.dir:
        base = Path(args.dir)
        pattern = "**/*.mp4" if args.recursive else "*.mp4"
        videos = sorted(base.glob(pattern))
    return videos


def main():
    parser = argparse.ArgumentParser(description="Evaluate fall/emergency detection on labeled videos.")
    parser.add_argument("--video",     help="Single video file path")
    parser.add_argument("--dir",       help="Directory of videos")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--config",    default="config.yaml", help="Config file path")
    parser.add_argument("--out",       default="outputs/eval_results.json", help="Output JSON path")
    args = parser.parse_args()

    if not args.video and not args.dir:
        parser.print_help()
        sys.exit(1)

    cfg = load_config(args.config)
    videos = collect_videos(args)

    if not videos:
        print("[Error] No .mp4 files found.")
        sys.exit(1)

    print(f"[Evaluate] {len(videos)} video(s) | config: {args.config}")
    results = []
    for vp in videos:
        print(f"\n[Video] {vp}")
        results.append(evaluate_video(vp, cfg))

    agg = aggregate(results)
    print_summary(agg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": agg, "per_video": results}, f, indent=2, default=str)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
