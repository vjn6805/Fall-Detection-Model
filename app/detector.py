import torch
from ultralytics import YOLO

PERSON_CLASS_ID = 0

# COCO 17-keypoint indices
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton connections (pairs of KP_NAMES indices)
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms
    (5, 11), (6, 12), (11, 12),                  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),      # legs
    (0, 5), (0, 6),                              # head-shoulders
]


class PersonDetector:
    def __init__(self, model_name: str, confidence: float, input_size: int, tracker: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence = confidence
        self.input_size = input_size
        self.tracker_cfg = f"{tracker}.yaml"
        try:
            self.model = YOLO(model_name)
            print(f"[Detector] Model '{model_name}' loaded on {self.device.upper()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    def track(self, frame):
        """
        Run pose model with ByteTrack on a single frame.
        Returns list of dicts per tracked person:
          {track_id, xyxy, confidence, keypoints, kp_conf}
        """
        results = self.model.track(
            frame,
            device=self.device,
            conf=self.confidence,
            imgsz=self.input_size,
            classes=[PERSON_CLASS_ID],
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )

        persons = []
        r = results[0]

        if r.boxes is None or r.boxes.id is None:
            return persons

        boxes = r.boxes
        kps = r.keypoints  # may be None if model has no pose head

        for i, box in enumerate(boxes):
            track_id = int(box.id[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])

            keypoints = None
            kp_conf = None
            if kps is not None:
                xy = kps.xy[i].cpu().numpy()       # (17, 2)
                conf_kp = kps.conf[i].cpu().numpy() # (17,)
                keypoints = xy
                kp_conf = conf_kp

            persons.append({
                "track_id": track_id,
                "xyxy": xyxy,
                "confidence": conf,
                "keypoints": keypoints,
                "kp_conf": kp_conf,
            })

        return persons
