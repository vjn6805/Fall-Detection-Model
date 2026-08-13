# CCTV Health Emergency Detection System
### Computer Vision Prototype — NOT a medical device

---

## Disclaimer

This is a **computer-vision research prototype**.
It detects **POTENTIAL HEALTH EMERGENCY** based on visual behavior only.
It is **NOT** a medical diagnosis tool.
It is **NOT** production-ready.
It is **NOT** a substitute for professional emergency response systems.
All detections should be treated as unverified alerts requiring human review.

---

## Architecture

```
Video Source (file / webcam / RTSP)
    ↓
OpenCV frame reader
    ↓
YOLOv8n-pose  (detection + 17-keypoint pose in one pass)
    ↓
ByteTrack     (multi-object tracking, stable IDs)
    ↓
StateManager  (per-person history buffer)
    ↓
FallDetector  (temporal fall state machine)
    ↓
RiskEngine    (health emergency risk scoring)
    ↓
IncidentStore (local JSON event log)
    ↓
Visualization (OpenCV overlay)
```

---

## Installation

```bash
pip install -r requirements.txt
```

`yolov8n-pose.pt` (~7MB) downloads automatically on first run.

---

## Running

```bash
python app/main.py
```

Press `q` to quit.

---

## Configuration

All parameters live in `config.yaml`. No thresholds are hard-coded.

### Source

```yaml
source:
  type: video       # video | webcam | rtsp
  path: videos/test.mp4
  webcam_id: 0
  rtsp_url: rtsp://...
```

### Model

```yaml
model:
  name: yolov8n-pose.pt
  confidence: 0.4
  input_size: 640    # reduce to 320 for faster CPU
```

### Tracking

```yaml
tracking:
  tracker: bytetrack
  max_missing_frames: 10
  pose_conf_threshold: 0.3
```

### Fall Detection Thresholds

```yaml
fall_detection:
  min_pose_confidence: 0.25
  min_track_age: 8
  horizontal_torso_angle: 50.0
  upright_torso_angle: 30.0
  horizontal_aspect_ratio: 1.3
  fall_velocity_threshold: 0.08
  transition_window_frames: 12
  possible_fall_frames: 4
  confirm_fall_seconds: 1.5
  post_fall_duration_seconds: 10.0
  recovery_upright_seconds: 2.0
  stillness_threshold: 6.0
```

### Risk Engine Thresholds

```yaml
risk_engine:
  emergency_score_threshold: 0.72
  emergency_persist_seconds: 3.0
  emergency_clear_seconds: 4.0
  time_down_reference_seconds: 15.0
  inactivity_window_frames: 20
  inactivity_stillness_px: 8.0
  recovery_motion_threshold: 15.0
```

---

## Model Selection

**YOLOv8n-pose** was selected because:
- Single forward pass produces both bounding boxes and 17 COCO keypoints
- Smallest YOLOv8 variant (~7MB) — suitable for CPU inference
- No training required — pretrained on COCO
- Integrated with ByteTrack via Ultralytics `.track()` API

Alternative considered: separate detection + pose models (e.g. YOLOv8n + ViTPose).
Rejected because two model calls per frame would halve CPU FPS.

---

## Tracking

**ByteTrack** (bundled in Ultralytics):
- Assigns stable integer IDs across frames
- Tolerates brief occlusions via `max_missing_frames`
- IDs increment and never reuse within a session
- Each person's state machine is completely independent

---

## Pose Estimation

17 COCO keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.

Features extracted per frame:
- Shoulder center, hip center
- Torso vector and angle from vertical (0° = upright, 90° = horizontal)
- Bounding-box aspect ratio (w/h)
- Position delta (pixel movement between frames)
- Mean keypoint confidence

---

## Fall Detection

### Principle
A fall is a **temporal event**, not a static posture.
The system looks for: UPRIGHT → RAPID CHANGE → HORIZONTAL → REMAINS DOWN.

### Scoring Formula
```
fall_score = 0.35 × posture_score
           + 0.30 × transition_score
           + 0.20 × persistence_score
           + 0.15 × motion_score
```

### State Machine
```
NORMAL → POSSIBLE_FALL → FALL_CONFIRMED → POST_FALL_MONITORING → RECOVERED → NORMAL
```

Transitions require temporal evidence — never a single frame.

### What it does NOT do
- A person lying down does NOT automatically trigger a fall
- A person sitting does NOT trigger a fall
- A person bending does NOT trigger a fall
- A stumble followed by immediate recovery returns to NORMAL

---

## Health Emergency Risk Scoring

### Principle
A confirmed fall is NOT automatically an emergency.
The risk engine evaluates post-fall behavior.

### Scoring Formula
```
health_emergency_score = 0.30 × fall_evidence
                       + 0.25 × time_down
                       + 0.25 × inactivity
                       + 0.20 × no_recovery
```

### State Machine
```
NORMAL → MONITORING → POSSIBLE_EMERGENCY → POTENTIAL_HEALTH_EMERGENCY
                                         ↓
                                      RECOVERED → NORMAL
```

---

## Threshold Justification

| Threshold | Value | Rationale |
|---|---|---|
| `horizontal_torso_angle` | 50° | Midpoint between forward bend (~35°) and lying (~80°). Catches falls without triggering on bends. |
| `min_track_age` | 8 frames | Prevents person entering frame already horizontal from triggering fall. |
| `confirm_fall_seconds` | 1.5s | Filters stumble+recovery. Person must stay down for 1.5s before FALL_CONFIRMED. |
| `possible_fall_frames` | 4 | ~0.5s at 8 FPS. Prevents single-frame noise. |
| `emergency_score_threshold` | 0.72 | Requires strong multi-signal evidence. Reduces false positives from sitting/lying. |
| `emergency_persist_seconds` | 3.0s | Score must stay above threshold for 3s before emergency declared. |
| `time_down_reference_seconds` | 15s | Person down for 15s with no recovery = full time_down score. |

All thresholds are prototype computer-vision values. They are NOT medically validated.

---

## Validation

### Dataset Structure

```
videos/
  normal/           walking, standing
  sitting/          intentional sitting
  lying/            intentional lying down
  bending/          bending, picking up objects
  exercise/         yoga, floor exercises
  fall_recovery/    fall + immediate recovery
  fall_no_recovery/ fall + remains down
  occlusion/        partial occlusion
  multi_person/     2+ people
  poor_lighting/    low light
```

### Running Evaluation

```bash
# Single video (requires matching .csv label file)
python tools/evaluate.py --video videos/fall_no_recovery/test1.mp4

# All videos in a directory
python tools/evaluate.py --dir videos/fall_no_recovery/

# All videos recursively
python tools/evaluate.py --dir videos/ --recursive
```

### Threshold Sweep

```bash
python tools/sweep.py --dir videos/ --recursive
```

Compares 4 configurations (default, sensitive, conservative, fast-confirm).
Results saved to `outputs/sweep_results.json`.

### Label Format

See `tools/label_format.md`. Each video needs a matching `.csv` file.

---

## Performance

| Metric | Value |
|---|---|
| Device | CPU (no CUDA available) |
| Model | YOLOv8n-pose |
| Input size | 640px |
| Avg inference | ~120–160ms |
| End-to-end FPS | ~6–8 FPS |
| Model size | ~7MB |

To improve CPU performance: set `input_size: 320` in `config.yaml` (~12–15 FPS).

---

## Known Limitations

1. **CPU-only**: ~6–8 FPS at 640px. Insufficient for fast falls at low FPS.
2. **Camera angle dependency**: Torso angle calculation assumes a roughly frontal or side-on camera. Overhead cameras will produce incorrect torso angles.
3. **Occlusion**: ByteTrack IDs reset after `max_missing_frames`. History is lost on ID reset.
4. **Pose failure**: If YOLOv8n-pose cannot detect keypoints (small person, poor lighting, unusual clothing), fall scoring is gated out.
5. **Aspect ratio ambiguity**: A person sitting cross-legged may have a wide bounding box. Mitigated by requiring torso angle confirmation.
6. **Entry-frame false positives**: Person entering frame already horizontal is gated by `min_track_age`. May still trigger if track age threshold is too low.
7. **Multi-person occlusion**: When two people overlap, bounding boxes may merge or swap IDs.
8. **No temporal ground truth for thresholds**: Thresholds were set by reasoning, not measured calibration. Validation dataset required for proper calibration.
9. **Not real-time at high resolution**: 640px input is borderline for real-time on CPU.

---

## Known False Positive Cases

| Scenario | Cause | Mitigation |
|---|---|---|
| Person sits on floor | Wide bounding box + low hip center | `min_track_age` + transition score requirement |
| Person does yoga | Horizontal posture without rapid transition | Transition score requires upright→horizontal change |
| Person enters frame lying down | No transition observed | `min_track_age` gate |
| Camera perspective distortion | Torso angle miscalculated | Pose confidence gate |
| Tracker ID switch | New track starts with no history | `min_track_age` gate |

---

## Known False Negative Cases

| Scenario | Cause | Mitigation |
|---|---|---|
| Very fast fall | Insufficient frames to build transition score | Lower `possible_fall_frames` |
| Person too small in frame | Low pose confidence | Reduce `min_pose_confidence` (increases noise) |
| Partial occlusion during fall | Keypoints unavailable | Pose confidence gate prevents false positive but also misses fall |
| Low FPS source | Transition window spans too few frames | Reduce `transition_window_frames` |
| Unusual fall direction (sideways) | Torso angle may not change significantly | Aspect ratio signal provides partial coverage |

---

## Privacy

- No face recognition
- No identity recognition
- No biometric data stored
- Only bounding box coordinates, pose keypoints, and derived scores are processed
- All data is local — no cloud transmission
- Incident logs contain only: timestamp, camera ID, person track ID (temporary), scores

---

## Event Logs

- `outputs/events.log` — human-readable state transition log
- `outputs/incidents.json` — structured incident records with full timeline

---

## Debug Mode

Set `display.debug: true` in `config.yaml` for per-person overlay showing all scores.

---

## Recommended Next Improvements

1. Collect and label a proper validation dataset
2. Run threshold sweep on labeled data and select thresholds by measured F1
3. Add GPU support (install CUDA PyTorch build)
4. Test with overhead camera angle and adjust torso angle calculation
5. Add pose-unavailable fallback using bounding-box-only signals
6. Implement notification/reporting layer (Phase 5+)

---

## Final Technical Report

### System

| Component | Selection |
|---|---|
| Detection + Pose | YOLOv8n-pose (Ultralytics, COCO pretrained) |
| Tracker | ByteTrack (bundled in Ultralytics) |
| Framework | Python 3.12, PyTorch 2.13 CPU, OpenCV 4.11 |

### Performance (CPU, 640px input)

| Metric | Value |
|---|---|
| Avg inference latency | ~130ms |
| End-to-end FPS | ~6–8 |
| Model size | 7MB |
| Peak persons supported | 5+ (FPS degrades with count) |

### Fall Detection (prototype thresholds, no labeled dataset yet)

| Metric | Status |
|---|---|
| Precision | Requires labeled dataset |
| Recall | Requires labeled dataset |
| F1 | Requires labeled dataset |
| Detection latency | ~0.5–2s after fall onset |

### Emergency Detection (prototype thresholds, no labeled dataset yet)

| Metric | Status |
|---|---|
| Precision | Requires labeled dataset |
| Recall | Requires labeled dataset |
| F1 | Requires labeled dataset |

### Main False Positives (expected, by design analysis)
- Person sitting on floor with wide bounding box
- Person entering frame already horizontal (mitigated by `min_track_age`)
- Yoga / floor exercise (mitigated by transition score requirement)

### Main False Negatives (expected, by design analysis)
- Very fast falls at low FPS
- Person too small / far from camera
- Sideways falls (torso angle change minimal)
- Occlusion during fall

### Known Limitations
See Known Limitations section above.

### Recommended Next Improvements
See Recommended Next Improvements section above.

---

*This system is a computer-vision prototype. It is not a medical device,
not a certified emergency response system, and not a substitute for
professional monitoring. All detections require human verification.*
