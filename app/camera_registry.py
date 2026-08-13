"""
camera_registry.py — Loads and provides camera metadata from cameras.yaml.

Each camera has: camera_id, name, source, location (lat/lon/address/zone).
The detection pipeline uses camera_id; the incident manager enriches incidents
with full camera metadata from this registry.
"""

import yaml
from pathlib import Path


class CameraRegistry:
    def __init__(self, path: str = "cameras.yaml"):
        self._cameras: dict[str, dict] = {}
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for cam in data.get("cameras", []):
                cid = cam.get("camera_id")
                if cid:
                    self._cameras[cid] = cam
        # Always provide a fallback so missing registry never crashes detection
        self._fallback = {
            "camera_id": "UNKNOWN",
            "name": "Unknown Camera",
            "source": "",
            "location": {"latitude": 0.0, "longitude": 0.0, "address": "Unknown", "zone": "Unknown"},
        }

    def get(self, camera_id: str) -> dict:
        return self._cameras.get(camera_id, {**self._fallback, "camera_id": camera_id})

    def all(self) -> list[dict]:
        return list(self._cameras.values())

    def ids(self) -> list[str]:
        return list(self._cameras.keys())
