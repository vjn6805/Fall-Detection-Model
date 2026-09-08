import os
import cv2
import time
import numpy as np
import logging
import threading
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("evidence_manager")

class RollingBuffer:
    def __init__(self, pre_event_seconds: float, fallback_fps: float = 10.0):
        self.pre_event_seconds = pre_event_seconds
        self.fallback_fps = fallback_fps
        self.frames = deque()  # stores (timestamp, jpeg_bytes)
        self.timestamps = deque(maxlen=100)  # used for rolling FPS estimation

    def append(self, timestamp: float, frame: np.ndarray):
        # Compress frame to JPEG to conserve memory
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        
        self.frames.append((timestamp, encoded.tobytes()))
        self.timestamps.append(timestamp)

        # Estimate FPS
        fps = self.estimate_fps()
        max_len = int(self.pre_event_seconds * fps)
        max_len = max(max_len, 5)  # Ensure at least a minimal buffer length

        while len(self.frames) > max_len:
            self.frames.popleft()

    def estimate_fps(self) -> float:
        if len(self.timestamps) < 2:
            return self.fallback_fps
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return self.fallback_fps
        fps = len(self.timestamps) / duration
        if 1.0 <= fps <= 120.0:
            return fps
        return self.fallback_fps

    def get_frames(self) -> list:
        return list(self.frames)


class ActiveRecording:
    def __init__(self, incident_id: str, camera_id: str, pre_event_frames: list, post_event_seconds: float, fps: float):
        self.incident_id = incident_id
        self.camera_id = camera_id
        self.pre_event_frames = pre_event_frames  # List of (timestamp, jpeg_bytes)
        self.post_event_seconds = post_event_seconds
        self.fps = fps
        self.post_event_frames = []
        self.target_post_frames = int(post_event_seconds * fps)
        self.started_at = time.time()
        self.status = "CAPTURING"

    def append(self, timestamp: float, frame: np.ndarray):
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.post_event_frames.append((timestamp, encoded.tobytes()))

    def is_finished(self) -> bool:
        # Stop if we hit the target frames OR elapsed time exceeds post-event seconds + 5s buffer (failsafe)
        elapsed = time.time() - self.started_at
        if len(self.post_event_frames) >= self.target_post_frames:
            return True
        if elapsed > (self.post_event_seconds + 5.0):
            logger.warning(f"Failsafe triggered for {self.incident_id}: stopped post-event capture due to timeout.")
            return True
        return False


class EvidenceManager:
    def __init__(self, config: dict, incident_manager):
        self.config = config
        self.incident_mgr = incident_manager
        
        evidence_cfg = config.get("evidence", {})
        self.enabled = evidence_cfg.get("enabled", True)
        self.pre_event_seconds = float(evidence_cfg.get("pre_event_seconds", 10.0))
        self.post_event_seconds = float(evidence_cfg.get("post_event_seconds", 10.0))
        self.video_dir = Path(evidence_cfg.get("video_dir", "outputs/evidence"))
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        # camera_id -> RollingBuffer
        self._buffers = {}
        # incident_id -> ActiveRecording
        self._active_recordings = {}
        self._lock = threading.Lock()

    def update(self, camera_id: str, frame: np.ndarray, timestamp: float):
        if not self.enabled:
            return

        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = RollingBuffer(self.pre_event_seconds)
            self._buffers[camera_id].append(timestamp, frame)

            # Append to any active recordings for this camera
            finished_ids = []
            for inc_id, rec in self._active_recordings.items():
                if rec.camera_id == camera_id and rec.status == "CAPTURING":
                    rec.append(timestamp, frame)
                    if rec.is_finished():
                        finished_ids.append(inc_id)

            for inc_id in finished_ids:
                rec = self._active_recordings[inc_id]
                rec.status = "WRITING"
                # Start background thread to write video
                threading.Thread(target=self._write_evidence_clip, args=(rec,), daemon=True).start()

    def trigger_capture(self, incident_id: str, camera_id: str, person_id: int, risk_result, frame_at_event: np.ndarray):
        if not self.enabled:
            return

        with self._lock:
            if incident_id in self._active_recordings:
                return  # Prevent duplicates

            # Get pre-event frames
            pre_frames = []
            fps = 10.0
            if camera_id in self._buffers:
                pre_frames = self._buffers[camera_id].get_frames()
                fps = self._buffers[camera_id].estimate_fps()

            logger.info(f"Triggering evidence capture for {incident_id}. Pre-event frames: {len(pre_frames)}, estimated FPS: {fps:.1f}")
            self._active_recordings[incident_id] = ActiveRecording(
                incident_id, camera_id, pre_frames, self.post_event_seconds, fps
            )
            
            # Immediately update status in db to CAPTURING
            self.incident_mgr.update_evidence(
                incident_id=incident_id,
                evidence_path="",
                start_time="",
                end_time="",
                duration=0.0,
                status="CAPTURING"
            )

    def _write_evidence_clip(self, recording: ActiveRecording):
        inc_id = recording.incident_id
        try:
            all_frames = recording.pre_event_frames + recording.post_event_frames
            if not all_frames:
                raise ValueError("No frames available for video clip.")

            # Decompress first frame to get size
            first_frame_bytes = all_frames[0][1]
            first_frame = cv2.imdecode(np.frombuffer(first_frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if first_frame is None:
                raise ValueError("Could not decode first frame.")
            
            height, width, _ = first_frame.shape
            filepath = self.video_dir / f"{inc_id}.mp4"

            # Determine timestamps
            start_ts = all_frames[0][0]
            end_ts = all_frames[-1][0]
            duration = end_ts - start_ts

            # Codecs to try: avc1 (H264), mp4v (MPEG-4), MJPG
            codecs = ['avc1', 'mp4v', 'MJPG']
            writer = None
            used_codec = None

            for codec in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(str(filepath), fourcc, recording.fps, (width, height))
                    if writer.isOpened():
                        used_codec = codec
                        break
                except Exception as e:
                    if writer:
                        writer.release()
                    writer = None

            if writer is None or not writer.isOpened():
                raise RuntimeError("Could not initialize VideoWriter with any of H264, MP4V, or MJPG.")

            logger.info(f"Writing evidence video to {filepath} using codec {used_codec}...")
            
            for timestamp, frame_bytes in all_frames:
                img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    writer.write(img)

            writer.release()
            logger.info(f"Successfully wrote evidence video {filepath}.")

            # Format timestamps
            start_str = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Determine if it's PARTIAL or READY
            target_pre = recording.post_event_seconds  # using pre_event_seconds config
            # Wait, calculate pre-event duration actually captured
            pre_duration = 0.0
            if recording.pre_event_frames:
                pre_duration = recording.pre_event_frames[-1][0] - recording.pre_event_frames[0][0]

            if pre_duration < (self.pre_event_seconds * 0.9):
                status = "PARTIAL"
                logger.warning(f"Evidence clip for {inc_id} is PARTIAL. Pre-event duration: {pre_duration:.1f}s (target: {self.pre_event_seconds}s)")
            else:
                status = "READY"

            # Update DB with final information
            self.incident_mgr.update_evidence(
                incident_id=inc_id,
                evidence_path=str(filepath),
                start_time=start_str,
                end_time=end_str,
                duration=round(duration, 1),
                status=status
            )

        except Exception as e:
            logger.error(f"Failed to generate evidence clip for {inc_id}: {e}")
            self.incident_mgr.update_evidence(
                incident_id=inc_id,
                evidence_path="",
                start_time="",
                end_time="",
                duration=0.0,
                status="FAILED"
            )
        finally:
            with self._lock:
                if inc_id in self._active_recordings:
                    del self._active_recordings[inc_id]
