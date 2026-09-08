"""
incident_manager.py — Structured incident management layer.

ARCHITECTURE
------------
Detection Engine  →  RiskEngine (POTENTIAL_HEALTH_EMERGENCY)
                          ↓
                   IncidentManager.ingest()
                          ↓
                   Deduplication check
                          ↓
                   Incident created / updated in SQLite
                          ↓
                   Dashboard reads SQLite
                          ↓
                   Human: CONFIRM or DISMISS
                          ↓
                   ResponseDispatcher.dispatch()  ← stub, Phase 5

INCIDENT LIFECYCLE
------------------
DETECTED → UNDER_REVIEW → CONFIRMED → RESOLVED
                        → DISMISSED

SEVERITY
--------
LOW      — fall + immediate recovery (POSSIBLE_FALL only, risk < 0.5)
MEDIUM   — fall confirmed, risk 0.5–0.72
HIGH     — POTENTIAL_HEALTH_EMERGENCY, risk 0.72–0.88
CRITICAL — POTENTIAL_HEALTH_EMERGENCY, risk > 0.88 OR time_down > 30s

All severity levels are operational priorities for this prototype.
They are NOT medically validated.

DEDUPLICATION
-------------
Key: (camera_id, person_id, event_type)
An active incident (status in DETECTED/UNDER_REVIEW) for the same key
is updated rather than duplicated.

IMPORTANT: This is a prototype. All incidents require human review.
"""

import sqlite3
import time
import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("incident_manager")


# ---------------------------------------------------------------------------
# Incident dataclass
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    incident_id: str
    camera_id: str
    camera_name: str
    camera_address: str
    camera_zone: str
    camera_lat: float
    camera_lon: float
    event_type: str                  # POTENTIAL_HEALTH_EMERGENCY | POSSIBLE_FALL
    severity: str                    # LOW | MEDIUM | HIGH | CRITICAL
    person_id: int
    fall_score: float
    health_emergency_score: float
    observation_quality: str
    duration_down: float
    reasons: list                    # human-readable list of detection signals
    status: str = "DETECTED"         # DETECTED | UNDER_REVIEW | CONFIRMED | DISMISSED | RESOLVED
    ai_state: str = ""               # raw AI state string
    created_at: str = ""
    updated_at: str = ""
    confirmed_at: Optional[str] = None
    confirmed_by: Optional[str] = None
    dismissed_at: Optional[str] = None
    dismissed_by: Optional[str] = None
    dismissal_reason: Optional[str] = None
    evidence_path: Optional[str] = None   # path to snapshot image
    resolved_at: Optional[str] = None

    def to_response_payload(self) -> dict:
        """Structured payload ready for future Phase 5 response dispatch."""
        return {
            "incident_id":           self.incident_id,
            "event_type":            self.event_type,
            "camera_id":             self.camera_id,
            "location": {
                "latitude":  self.camera_lat,
                "longitude": self.camera_lon,
            },
            "address":               self.camera_address,
            "zone":                  self.camera_zone,
            "timestamp":             self.created_at,
            "confidence":            self.health_emergency_score,
            "fall_score":            self.fall_score,
            "duration_down_seconds": self.duration_down,
            "severity":              self.severity,
            "status":                self.status,
            "reasons":               self.reasons,
        }


# ---------------------------------------------------------------------------
# Severity classifier
# ---------------------------------------------------------------------------

def _classify_severity(event_type: str, risk_score: float, duration_down: float) -> str:
    """
    Operational severity for prototype use only.
    NOT medically validated.

    CRITICAL  — PHE + (risk > 0.88 OR down > 30s)
    HIGH      — PHE + risk 0.72–0.88
    MEDIUM    — POSSIBLE_FALL or risk 0.50–0.72
    LOW       — below MEDIUM thresholds
    """
    if event_type == "POTENTIAL_HEALTH_EMERGENCY":
        if risk_score > 0.88 or duration_down > 30.0:
            return "CRITICAL"
        return "HIGH"
    if risk_score >= 0.50:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Reason builder
# ---------------------------------------------------------------------------

def _build_reasons(fall_result, risk_result) -> list:
    """
    Converts numeric sub-scores into human-readable detection signals.
    Uses measurable thresholds — never vague language.
    """
    reasons = []
    if fall_result:
        if fall_result.transition_score > 0.3:
            reasons.append("Rapid downward movement detected")
        if fall_result.posture_score > 0.5:
            reasons.append("Abnormal posture transition detected")
        if fall_result.persistence_score > 0.4:
            reasons.append("Person remained in low posture")
        if fall_result.motion_score > 0.5:
            reasons.append("Limited movement after posture change")
        if fall_result.transition_reason:
            reasons.append(f"Transition signal: {fall_result.transition_reason.replace('_', ' ')}")
    if risk_result:
        if risk_result.no_recovery_score > 0.6:
            reasons.append("No recovery movement detected")
        if risk_result.inactivity_score > 0.6:
            reasons.append("Prolonged inactivity observed")
        if risk_result.time_down_score > 0.5:
            reasons.append(f"Person down for {risk_result.time_since_fall:.1f}s")
        obs = getattr(fall_result, "observability", "UNKNOWN") if fall_result else "UNKNOWN"
        reasons.append(f"Pose observability: {obs}")
    return reasons if reasons else ["Detection signal threshold exceeded"]


# ---------------------------------------------------------------------------
# Incident ID generator
# ---------------------------------------------------------------------------

def _make_incident_id() -> str:
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    short = uuid.uuid4().hex[:4].upper()
    return f"INC-{date_str}-{short}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id         TEXT PRIMARY KEY,
    camera_id           TEXT NOT NULL,
    camera_name         TEXT,
    camera_address      TEXT,
    camera_zone         TEXT,
    camera_lat          REAL,
    camera_lon          REAL,
    event_type          TEXT,
    severity            TEXT,
    person_id           INTEGER,
    fall_score          REAL,
    health_emergency_score REAL,
    observation_quality TEXT,
    duration_down       REAL,
    reasons             TEXT,   -- JSON array
    status              TEXT DEFAULT 'DETECTED',
    ai_state            TEXT,
    created_at          TEXT,
    updated_at          TEXT,
    confirmed_at        TEXT,
    confirmed_by        TEXT,
    dismissed_at        TEXT,
    dismissed_by        TEXT,
    dismissal_reason    TEXT,
    evidence_path       TEXT,   -- video clip file path
    evidence_start_time TEXT,
    evidence_end_time   TEXT,
    evidence_duration   REAL,
    evidence_status     TEXT DEFAULT 'PENDING',
    snapshot_path       TEXT,
    resolved_at         TEXT
);

CREATE TABLE IF NOT EXISTS incident_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id     TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    event           TEXT NOT NULL,   -- state transition label
    detail          TEXT,            -- JSON with scores
    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT,
    address     TEXT,
    zone        TEXT,
    latitude    REAL,
    longitude   REAL,
    source      TEXT,
    status      TEXT DEFAULT 'NORMAL',   -- NORMAL | INCIDENT
    last_updated TEXT
);
"""


# ---------------------------------------------------------------------------
# IncidentManager
# ---------------------------------------------------------------------------

class IncidentManager:
    """
    Replaces IncidentStore. Consumes validated AI events from RiskEngine
    and manages the full incident lifecycle in SQLite.

    Thread-safety: SQLite WAL mode + check_same_thread=False is sufficient
    for the single-writer (detection loop) + single-reader (dashboard) pattern.
    """

    def __init__(self, db_path: str, camera_registry, snapshot_dir: str = "outputs/snapshots", evidence_dir: str = "outputs/evidence"):
        self._db_path = db_path
        self._registry = camera_registry
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._evidence_dir = Path(evidence_dir)
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        self._evidence_mgr = None

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # Schema migrations for existing databases
        for col_name, col_type in [
            ("evidence_start_time", "TEXT"),
            ("evidence_end_time", "TEXT"),
            ("evidence_duration", "REAL"),
            ("evidence_status", "TEXT DEFAULT 'PENDING'"),
            ("snapshot_path", "TEXT")
        ]:
            try:
                self._conn.execute(f"ALTER TABLE incidents ADD COLUMN {col_name} {col_type}")
                self._conn.commit()
            except sqlite3.OperationalError:
                pass

        self._sync_cameras()

        # In-memory dedup index: (camera_id, person_id, event_type) → incident_id
        self._active_key: dict[tuple, str] = {}
        self._load_active_keys()

    def set_evidence_manager(self, evidence_mgr):
        self._evidence_mgr = evidence_mgr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, camera_id: str, person_id: int, fall_result, risk_result,
               frame: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Called every frame from StateManager when risk state is active.
        Returns incident_id if an incident was created or updated, else None.
        Only creates incidents for POTENTIAL_HEALTH_EMERGENCY or POSSIBLE_FALL.
        """
        if risk_result is None:
            return None

        rs = risk_result.state.value
        if rs not in ("POTENTIAL_HEALTH_EMERGENCY", "POSSIBLE_EMERGENCY",
                      "MONITORING", "POSSIBLE_FALL"):
            return None

        # Map internal states to incident event types
        if rs == "POTENTIAL_HEALTH_EMERGENCY":
            event_type = "POTENTIAL_HEALTH_EMERGENCY"
        else:
            event_type = "POSSIBLE_FALL"

        dedup_key = (camera_id, person_id, event_type)
        existing_id = self._active_key.get(dedup_key)

        if existing_id:
            self._update_incident(existing_id, fall_result, risk_result)
            return existing_id
        else:
            inc_id = self._create_incident(
                camera_id, person_id, event_type, fall_result, risk_result, frame
            )
            self._active_key[dedup_key] = inc_id
            return inc_id

    def confirm(self, incident_id: str, confirmed_by: str = "Human Operator") -> bool:
        """Operator confirms the incident. Returns True if successful."""
        try:
            now = _now_iso()
            self._conn.execute(
                "UPDATE incidents SET status='CONFIRMED', confirmed_at=?, confirmed_by=?, updated_at=? WHERE incident_id=?",
                (now, confirmed_by, now, incident_id)
            )
            self._log_event(incident_id, "CONFIRMED", {"confirmed_by": confirmed_by})
            self._conn.commit()
            self._update_camera_status(incident_id)
            ResponseDispatcher.dispatch(self.get_incident(incident_id))
            return True
        except Exception as e:
            logger.error(f"confirm failed for {incident_id}: {e}")
            return False

    def dismiss(self, incident_id: str, dismissed_by: str = "Human Operator",
                reason: str = "") -> bool:
        """Operator dismisses the incident. Returns True if successful."""
        try:
            now = _now_iso()
            self._conn.execute(
                "UPDATE incidents SET status='DISMISSED', dismissed_at=?, dismissed_by=?, dismissal_reason=?, updated_at=? WHERE incident_id=?",
                (now, dismissed_by, reason or None, now, incident_id)
            )
            self._log_event(incident_id, "DISMISSED", {"dismissed_by": dismissed_by, "reason": reason})
            self._conn.commit()
            self._remove_active_key(incident_id)
            self._update_camera_status_by_camera(
                self._conn.execute("SELECT camera_id FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()["camera_id"]
            )
            return True
        except Exception as e:
            logger.error(f"dismiss failed for {incident_id}: {e}")
            return False

    def resolve(self, incident_id: str) -> bool:
        """Mark a confirmed incident as resolved."""
        try:
            now = _now_iso()
            self._conn.execute(
                "UPDATE incidents SET status='RESOLVED', resolved_at=?, updated_at=? WHERE incident_id=?",
                (now, now, incident_id)
            )
            self._log_event(incident_id, "RESOLVED", {})
            self._conn.commit()
            self._remove_active_key(incident_id)
            return True
        except Exception as e:
            logger.error(f"resolve failed for {incident_id}: {e}")
            return False

    def get_incident(self, incident_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM incidents WHERE incident_id=?", (incident_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["reasons"] = json.loads(d["reasons"] or "[]")
        d["timeline"] = self.get_timeline(incident_id)
        return d

    def get_timeline(self, incident_id: str) -> list:
        rows = self._conn.execute(
            "SELECT timestamp, event, detail FROM incident_events WHERE incident_id=? ORDER BY id",
            (incident_id,)
        ).fetchall()
        result = []
        for r in rows:
            entry = {"timestamp": r["timestamp"], "event": r["event"]}
            if r["detail"]:
                try:
                    entry["detail"] = json.loads(r["detail"])
                except Exception:
                    entry["detail"] = r["detail"]
            result.append(entry)
        return result

    def list_incidents(self, status: str = None, camera_id: str = None,
                       event_type: str = None, limit: int = 100) -> list:
        q = "SELECT * FROM incidents WHERE 1=1"
        params = []
        if status:
            q += " AND status=?"
            params.append(status)
        if camera_id:
            q += " AND camera_id=?"
            params.append(camera_id)
        if event_type:
            q += " AND event_type=?"
            params.append(event_type)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(q, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["reasons"] = json.loads(d["reasons"] or "[]")
            result.append(d)
        return result

    def list_cameras(self) -> list:
        return [dict(r) for r in self._conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()]

    def get_camera(self, camera_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
        return dict(row) if row else None

    def notify_risk_cleared(self, camera_id: str, person_id: int):
        """Called when risk engine returns to NORMAL — close active dedup keys."""
        for event_type in ("POTENTIAL_HEALTH_EMERGENCY", "POSSIBLE_FALL"):
            key = (camera_id, person_id, event_type)
            if key in self._active_key:
                del self._active_key[key]
        self._update_camera_status_by_camera(camera_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_incident(self, camera_id, person_id, event_type,
                         fall_result, risk_result, frame) -> str:
        cam = self._registry.get(camera_id)
        loc = cam.get("location", {})
        inc_id = _make_incident_id()
        now = _now_iso()

        reasons = _build_reasons(fall_result, risk_result)
        severity = _classify_severity(
            event_type,
            risk_result.health_emergency_score,
            risk_result.time_since_fall,
        )

        snap_path = None
        if frame is not None:
            snap_path = self._save_snapshot(frame, inc_id, person_id, risk_result)

        self._conn.execute("""
            INSERT INTO incidents
            (incident_id, camera_id, camera_name, camera_address, camera_zone,
             camera_lat, camera_lon, event_type, severity, person_id,
             fall_score, health_emergency_score, observation_quality,
             duration_down, reasons, status, ai_state, created_at, updated_at,
             snapshot_path, evidence_status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            inc_id, camera_id,
            cam.get("name", camera_id),
            loc.get("address", ""),
            loc.get("zone", ""),
            loc.get("latitude", 0.0),
            loc.get("longitude", 0.0),
            event_type, severity, person_id,
            round(fall_result.fall_score if fall_result else 0.0, 3),
            round(risk_result.health_emergency_score, 3),
            getattr(fall_result, "observability", "UNKNOWN") if fall_result else "UNKNOWN",
            round(risk_result.time_since_fall, 1),
            json.dumps(reasons),
            "DETECTED",
            risk_result.state.value,
            now, now,
            str(snap_path) if snap_path else None,
            "PENDING"
        ))
        self._log_event(inc_id, "DETECTED", {
            "risk_score": risk_result.health_emergency_score,
            "fall_score": fall_result.fall_score if fall_result else 0.0,
            "severity": severity,
        })
        self._conn.commit()

        # Immediately move to UNDER_REVIEW (human attention required)
        self._conn.execute(
            "UPDATE incidents SET status='UNDER_REVIEW', updated_at=? WHERE incident_id=?",
            (now, inc_id)
        )
        self._log_event(inc_id, "UNDER_REVIEW", {"note": "Awaiting human review"})
        self._conn.commit()

        self._set_camera_incident(camera_id)
        logger.info(f"Incident {inc_id} created | {camera_id} | Person#{person_id} | {event_type} | {severity}")

        if self._evidence_mgr and self._evidence_mgr.enabled:
            self._evidence_mgr.trigger_capture(inc_id, camera_id, person_id, risk_result, frame)

        return inc_id

    def _update_incident(self, incident_id: str, fall_result, risk_result):
        """Update scores on existing active incident — no duplicate creation."""
        now = _now_iso()
        reasons = _build_reasons(fall_result, risk_result)
        severity = _classify_severity(
            self._conn.execute("SELECT event_type FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()["event_type"],
            risk_result.health_emergency_score,
            risk_result.time_since_fall,
        )
        self._conn.execute("""
            UPDATE incidents SET
                fall_score=?, health_emergency_score=?, duration_down=?,
                severity=?, reasons=?, ai_state=?, updated_at=?
            WHERE incident_id=?
        """, (
            round(fall_result.fall_score if fall_result else 0.0, 3),
            round(risk_result.health_emergency_score, 3),
            round(risk_result.time_since_fall, 1),
            severity,
            json.dumps(reasons),
            risk_result.state.value,
            now,
            incident_id,
        ))
        self._conn.commit()

    def update_evidence(self, incident_id: str, evidence_path: str, start_time: str, end_time: str, duration: float, status: str):
        try:
            now = _now_iso()
            self._conn.execute("""
                UPDATE incidents SET 
                    evidence_path=?, 
                    evidence_start_time=?, 
                    evidence_end_time=?, 
                    evidence_duration=?, 
                    evidence_status=?, 
                    updated_at=? 
                WHERE incident_id=?
            """, (evidence_path, start_time, end_time, duration, status, now, incident_id))
            self._conn.commit()
            logger.info(f"Updated evidence for {incident_id}: status={status}, path={evidence_path}")
            self._log_event(incident_id, "EVIDENCE_UPDATED", {"status": status, "path": evidence_path})
        except Exception as e:
            logger.error(f"Failed to update evidence for {incident_id}: {e}")

    def _save_snapshot(self, frame: np.ndarray, incident_id: str,
                       person_id: int, risk_result) -> Optional[Path]:
        try:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            fname = self._snapshot_dir / f"{incident_id}.jpg"
            # Overlay minimal info — no face recognition, no identity
            annotated = frame.copy()
            cv2.putText(annotated, f"INC:{incident_id}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated, f"Person#{person_id}  Risk:{risk_result.health_emergency_score:.2f}",
                        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated, ts, (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imwrite(str(fname), annotated)
            return fname
        except Exception as e:
            logger.warning(f"Snapshot save failed: {e}")
            return None

    def _log_event(self, incident_id: str, event: str, detail: dict):
        self._conn.execute(
            "INSERT INTO incident_events (incident_id, timestamp, event, detail) VALUES (?,?,?,?)",
            (incident_id, _now_iso(), event, json.dumps(detail) if detail else None)
        )

    def _sync_cameras(self):
        """Populate cameras table from registry."""
        for cam in self._registry.all():
            loc = cam.get("location", {})
            self._conn.execute("""
                INSERT OR REPLACE INTO cameras
                (camera_id, name, address, zone, latitude, longitude, source, status, last_updated)
                VALUES (?,?,?,?,?,?,?,
                    COALESCE((SELECT status FROM cameras WHERE camera_id=?), 'NORMAL'),
                    ?)
            """, (
                cam["camera_id"], cam.get("name", ""), loc.get("address", ""),
                loc.get("zone", ""), loc.get("latitude", 0.0), loc.get("longitude", 0.0),
                cam.get("source", ""),
                cam["camera_id"],
                _now_iso(),
            ))
        self._conn.commit()

    def _load_active_keys(self):
        rows = self._conn.execute(
            "SELECT incident_id, camera_id, person_id, event_type FROM incidents WHERE status IN ('DETECTED','UNDER_REVIEW')"
        ).fetchall()
        for r in rows:
            key = (r["camera_id"], r["person_id"], r["event_type"])
            self._active_key[key] = r["incident_id"]

    def _set_camera_incident(self, camera_id: str):
        self._conn.execute(
            "UPDATE cameras SET status='INCIDENT', last_updated=? WHERE camera_id=?",
            (_now_iso(), camera_id)
        )
        self._conn.commit()

    def _update_camera_status(self, incident_id: str):
        row = self._conn.execute("SELECT camera_id FROM incidents WHERE incident_id=?", (incident_id,)).fetchone()
        if row:
            self._update_camera_status_by_camera(row["camera_id"])

    def _update_camera_status_by_camera(self, camera_id: str):
        active = self._conn.execute(
            "SELECT COUNT(*) as c FROM incidents WHERE camera_id=? AND status IN ('DETECTED','UNDER_REVIEW')",
            (camera_id,)
        ).fetchone()["c"]
        status = "INCIDENT" if active > 0 else "NORMAL"
        self._conn.execute(
            "UPDATE cameras SET status=?, last_updated=? WHERE camera_id=?",
            (status, _now_iso(), camera_id)
        )
        self._conn.commit()

    def _remove_active_key(self, incident_id: str):
        self._active_key = {k: v for k, v in self._active_key.items() if v != incident_id}


# ---------------------------------------------------------------------------
# ResponseDispatcher — Phase 5 stub
# ---------------------------------------------------------------------------

class ResponseDispatcher:
    """
    Stub for future Phase 5 external response integration.

    When Phase 5 is implemented, replace the body of dispatch() with
    actual integrations (SMS, ambulance API, etc.).

    DO NOT implement external dispatch in this phase.
    """

    @staticmethod
    def dispatch(incident: Optional[dict]):
        if incident is None:
            return
        payload = {
            "incident_id":           incident.get("incident_id"),
            "event_type":            incident.get("event_type"),
            "camera_id":             incident.get("camera_id"),
            "location": {
                "latitude":  incident.get("camera_lat"),
                "longitude": incident.get("camera_lon"),
            },
            "address":               incident.get("camera_address"),
            "timestamp":             incident.get("created_at"),
            "confidence":            incident.get("health_emergency_score"),
            "duration_down_seconds": incident.get("duration_down"),
            "severity":              incident.get("severity"),
            "status":                incident.get("status"),
        }
        # Phase 5: replace this log line with actual dispatch
        logger.info(f"[ResponseDispatcher] Incident ready for future response dispatch: {payload}")
