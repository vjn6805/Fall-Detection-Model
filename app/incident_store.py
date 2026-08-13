"""
incident_store.py — Local JSON incident storage.

Stores structured incident records for each potential health emergency.
Each incident tracks the full state transition timeline.
Designed to be consumed by Phase 5 reporting without schema changes.
"""

import json
import time
from pathlib import Path
from typing import Optional


class IncidentStore:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._incidents: dict[str, dict] = {}   # incident_id → record
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    records = json.load(f)
                    self._incidents = {r["event_id"]: r for r in records if r.get("event_id")}
            except (json.JSONDecodeError, KeyError):
                self._incidents = {}

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(list(self._incidents.values()), f, indent=2, default=str)

    def upsert(self, event: dict):
        """Create or update an incident record. Appends timeline entry."""
        eid = event.get("event_id")
        if not eid:
            return

        if eid not in self._incidents:
            self._incidents[eid] = {
                "event_id":   eid,
                "camera_id":  event.get("camera_id"),
                "person_id":  event.get("person_id"),
                "start_time": event.get("start_time"),
                "timeline":   [],
                "final_state": None,
                "peak_risk_score": 0.0,
            }

        record = self._incidents[eid]

        # Append timeline entry
        record["timeline"].append({
            "timestamp":     time.time(),
            "state":         event.get("current_state"),
            "risk_score":    event.get("risk_score"),
            "fall_score":    event.get("fall_score"),
            "duration_down": event.get("duration_down"),
            "reason":        event.get("reason"),
        })

        # Track peak risk
        risk = event.get("risk_score", 0.0)
        if risk > record["peak_risk_score"]:
            record["peak_risk_score"] = risk

        record["final_state"] = event.get("current_state")
        self._save()

    def close_incident(self, event_id: str, final_state: str):
        if event_id in self._incidents:
            self._incidents[event_id]["final_state"] = final_state
            self._incidents[event_id]["closed_at"] = time.time()
            self._save()

    def get_active(self) -> list:
        """Return incidents not yet resolved."""
        terminal = {"RECOVERED", "NORMAL"}
        return [r for r in self._incidents.values() if r.get("final_state") not in terminal]
