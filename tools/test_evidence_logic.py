import os
import sys
import time
import numpy as np
import cv2
import sqlite3
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.incident_manager import IncidentManager, Incident
from app.evidence_manager import EvidenceManager
from app.camera_registry import CameraRegistry

class DummyState:
    def __init__(self, value):
        self.value = value

class DummyRiskResult:
    def __init__(self, score, state_val, time_since_fall):
        self.health_emergency_score = score
        self.state = DummyState(state_val)
        self.time_since_fall = time_since_fall

def run_test():
    print("=== STARTING EVIDENCE SYSTEM UNIT TEST ===")
    
    # Create test directories
    db_path = "outputs/test_incidents.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    registry = CameraRegistry("cameras.yaml")
    
    # Custom config for testing
    cfg = {
        "evidence": {
            "enabled": True,
            "pre_event_seconds": 3,
            "post_event_seconds": 3,
            "video_dir": "outputs/test_evidence",
            "snapshot_dir": "outputs/test_snapshots"
        },
        "incident": {
            "db_path": db_path
        }
    }
    
    # Clean previous test outputs
    test_video_dir = Path(cfg["evidence"]["video_dir"])
    if test_video_dir.exists():
        for f in test_video_dir.glob("*"):
            try:
                os.remove(f)
            except Exception:
                pass
    
    incident_mgr = IncidentManager(
        db_path=db_path,
        camera_registry=registry,
        snapshot_dir=cfg["evidence"]["snapshot_dir"],
        evidence_dir=cfg["evidence"]["video_dir"]
    )
    
    evidence_mgr = EvidenceManager(cfg, incident_mgr)
    incident_mgr.set_evidence_manager(evidence_mgr)
    
    camera_id = "CAM-001"
    person_id = 99
    
    print("\n1. Simulating 5 seconds of pre-event video stream (10 FPS)...")
    # Feed 50 frames (5 seconds)
    w, h = 640, 480
    start_time = time.time()
    for i in range(50):
        # Create a frame with frame index text to visually trace
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"CAM: {camera_id} | FRAME: {i}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        timestamp = start_time + (i * 0.1)
        evidence_mgr.update(camera_id, frame, timestamp)
        time.sleep(0.01) # fast simulation
        
    print(f"Rolling buffer frame count: {len(evidence_mgr._buffers[camera_id].frames)} (expected: ~30)")
    
    print("\n2. Triggering fall event...")
    dummy_risk = DummyRiskResult(0.95, "POTENTIAL_HEALTH_EMERGENCY", 15.0)
    
    # Create incident (this will trigger evidence_mgr.trigger_capture internally)
    event_frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(event_frame, "FALL TRIGGER FRAME", (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Mock fall result for ingest
    class DummyFallResult:
        def __init__(self):
            self.fall_score = 0.85
            self.observability = "HIGH"
            
    inc_id = incident_mgr._create_incident(
        camera_id=camera_id,
        person_id=person_id,
        event_type="POTENTIAL_HEALTH_EMERGENCY",
        fall_result=DummyFallResult(),
        risk_result=dummy_risk,
        frame=event_frame
    )
    
    print(f"Created incident ID: {inc_id}")
    
    # Ensure capture is active
    rec = evidence_mgr._active_recordings.get(inc_id)
    if rec:
        print(f"Active recording status: {rec.status} | Target post frames: {rec.target_post_frames}")
    else:
        print("ERROR: Active recording not found!")
        sys.exit(1)
        
    print("\n3. Simulating 5 seconds of post-event video stream...")
    for i in range(50, 100):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(frame, f"CAM: {camera_id} | POST-FRAME: {i}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        timestamp = start_time + (i * 0.1)
        evidence_mgr.update(camera_id, frame, timestamp)
        time.sleep(0.01)
        
    print("Waiting for video writer thread to finish...")
    # Wait up to 5s for the background thread to finish writing
    for _ in range(50):
        time.sleep(0.1)
        inc = incident_mgr.get_incident(inc_id)
        if inc and inc.get("evidence_status") in ("READY", "PARTIAL", "FAILED"):
            break
            
    # Load and check DB entry
    inc = incident_mgr.get_incident(inc_id)
    print("\n4. Verifying SQLite Database Entries:")
    print(f"  Incident ID:      {inc['incident_id']}")
    print(f"  Evidence Status:  {inc['evidence_status']}")
    print(f"  Evidence Path:    {inc['evidence_path']}")
    print(f"  Snapshot Path:    {inc['snapshot_path']}")
    print(f"  Start Time:       {inc['evidence_start_time']}")
    print(f"  End Time:         {inc['evidence_end_time']}")
    print(f"  Duration:         {inc['evidence_duration']}s")
    
    # Verify local file storage
    mp4_path = Path(inc['evidence_path']) if inc['evidence_path'] else None
    jpg_path = Path(inc['snapshot_path']) if inc['snapshot_path'] else None
    
    print("\n5. Verifying file system storage:")
    if mp4_path and mp4_path.exists():
        print(f"  [PASS] MP4 video file exists: {mp4_path} ({mp4_path.stat().st_size} bytes)")
    else:
        print(f"  [FAIL] MP4 video file does not exist or empty!")
        
    if jpg_path and jpg_path.exists():
        print(f"  [PASS] JPG snapshot file exists: {jpg_path} ({jpg_path.stat().st_size} bytes)")
    else:
        print(f"  [FAIL] JPG snapshot file does not exist or empty!")
        
    # Clean up test DB
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print("\n=== EVIDENCE SYSTEM UNIT TEST COMPLETED ===")

if __name__ == "__main__":
    run_test()
