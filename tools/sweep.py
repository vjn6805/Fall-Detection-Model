"""
sweep.py — Threshold calibration sweep tool.

Runs evaluate.py across a grid of threshold configurations and prints
a comparison table. Identifies the best configuration by F1 score.

Usage:
    python tools/sweep.py --dir videos/fall_no_recovery/
    python tools/sweep.py --dir videos/ --recursive
    python tools/sweep.py --video videos/fall_no_recovery/test1.mp4

The sweep grid is defined in SWEEP_GRID below.
Modify it to test the parameters you care about.
Results saved to outputs/sweep_results.json.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import copy
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_config
from tools.evaluate import evaluate_video, aggregate

# ---------------------------------------------------------------------------
# Sweep grid — edit these to test different threshold combinations
# Each entry is a dict of config overrides (dot-path → value)
# ---------------------------------------------------------------------------

SWEEP_GRID = [
    {
        "name": "Config-A  (default)",
        "fall_detection.horizontal_torso_angle": 50.0,
        "fall_detection.confirm_fall_seconds":   1.5,
        "fall_detection.possible_fall_frames":   4,
        "risk_engine.emergency_score_threshold": 0.72,
        "risk_engine.emergency_persist_seconds": 3.0,
    },
    {
        "name": "Config-B  (more sensitive)",
        "fall_detection.horizontal_torso_angle": 40.0,
        "fall_detection.confirm_fall_seconds":   1.0,
        "fall_detection.possible_fall_frames":   3,
        "risk_engine.emergency_score_threshold": 0.65,
        "risk_engine.emergency_persist_seconds": 2.0,
    },
    {
        "name": "Config-C  (more conservative)",
        "fall_detection.horizontal_torso_angle": 60.0,
        "fall_detection.confirm_fall_seconds":   2.5,
        "fall_detection.possible_fall_frames":   6,
        "risk_engine.emergency_score_threshold": 0.80,
        "risk_engine.emergency_persist_seconds": 4.0,
    },
    {
        "name": "Config-D  (fast confirm, high emergency bar)",
        "fall_detection.horizontal_torso_angle": 45.0,
        "fall_detection.confirm_fall_seconds":   1.0,
        "fall_detection.possible_fall_frames":   3,
        "risk_engine.emergency_score_threshold": 0.80,
        "risk_engine.emergency_persist_seconds": 5.0,
    },
]


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dot-path overrides to a config dict. Returns modified copy."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if key == "name":
            continue
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def run_sweep(videos: list, base_cfg: dict) -> list:
    sweep_results = []

    for grid_entry in SWEEP_GRID:
        name = grid_entry.get("name", "unnamed")
        cfg = apply_overrides(base_cfg, grid_entry)

        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        per_video = []
        for vp in videos:
            print(f"\n  [Video] {vp.name}")
            r = evaluate_video(vp, cfg)
            per_video.append(r)

        agg = aggregate(per_video)
        sweep_results.append({
            "config_name":  name,
            "overrides":    {k: v for k, v in grid_entry.items() if k != "name"},
            "summary":      agg,
            "per_video":    per_video,
        })

    return sweep_results


def print_comparison(sweep_results: list):
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP COMPARISON")
    print("=" * 80)
    header = f"{'Config':<35} {'Fall-P':>7} {'Fall-R':>7} {'Fall-F1':>8} {'Emrg-P':>7} {'Emrg-R':>7} {'Emrg-F1':>8} {'FPS':>6}"
    print(header)
    print("-" * 80)

    best_f1 = -1
    best_name = ""

    for r in sweep_results:
        name = r["config_name"]
        s = r["summary"]
        if not s:
            continue
        f = s["fall"]
        e = s["emergency"]
        combined_f1 = (f["f1"] + e["f1"]) / 2
        if combined_f1 > best_f1:
            best_f1 = combined_f1
            best_name = name

        print(f"{name:<35} {f['precision']:>7.3f} {f['recall']:>7.3f} {f['f1']:>8.3f}"
              f" {e['precision']:>7.3f} {e['recall']:>7.3f} {e['f1']:>8.3f} {s['avg_fps']:>6.1f}")

    print("-" * 80)
    print(f"\nBest combined F1: {best_name}  (F1={best_f1:.3f})")
    print("\nTradeoff notes:")
    print("  Config-B (sensitive): higher recall, more false positives")
    print("  Config-C (conservative): fewer false positives, may miss slow falls")
    print("  Config-D: fast fall confirm but strict emergency — good for active environments")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep for fall/emergency detection.")
    parser.add_argument("--video",     help="Single video file")
    parser.add_argument("--dir",       help="Directory of videos")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--out",       default="outputs/sweep_results.json")
    args = parser.parse_args()

    if not args.video and not args.dir:
        parser.print_help()
        sys.exit(1)

    base_cfg = load_config(args.config)

    videos = []
    if args.video:
        videos = [Path(args.video)]
    elif args.dir:
        base = Path(args.dir)
        pattern = "**/*.mp4" if args.recursive else "*.mp4"
        videos = sorted(base.glob(pattern))

    if not videos:
        print("[Error] No .mp4 files found.")
        sys.exit(1)

    # Filter to only videos that have label files
    labeled = [v for v in videos if v.with_suffix(".csv").exists()]
    if not labeled:
        print("[Error] No labeled videos found (no matching .csv files).")
        print("  See tools/label_format.md for the label format.")
        sys.exit(1)

    print(f"[Sweep] {len(labeled)} labeled video(s) | {len(SWEEP_GRID)} configurations")
    sweep_results = run_sweep(labeled, base_cfg)
    print_comparison(sweep_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
