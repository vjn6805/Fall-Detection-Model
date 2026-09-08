"""
dashboard/app.py — Command Center Dashboard

Runs as a separate process from the detection loop.
Detection loop writes to SQLite; dashboard reads from it.
If the dashboard crashes, detection continues unaffected.

Endpoints:
  GET  /                              — main dashboard UI
  GET  /api/cameras                   — all cameras + status
  GET  /api/cameras/<camera_id>       — single camera detail
  GET  /api/incidents                 — incident list (filterable)
  GET  /api/incidents/<id>            — single incident + timeline
  POST /api/incidents/<id>/confirm    — operator confirms
  POST /api/incidents/<id>/dismiss    — operator dismisses
  POST /api/incidents/<id>/resolve    — operator resolves confirmed incident
  GET  /api/incidents/<id>/snapshot   — serve snapshot image
  GET  /stream                        — SSE stream for real-time updates
"""

import sys
import os
import json
import time
import sqlite3
import queue
import threading
from pathlib import Path
from datetime import datetime, timezone

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Remove dashboard/ from path so 'app' resolves to app/ package, not dashboard/app.py
_dashboard_dir = str(Path(__file__).parent)
if _dashboard_dir in sys.path:
    sys.path.remove(_dashboard_dir)

from flask import Flask, jsonify, request, render_template, Response, send_file, abort
from app.camera_registry import CameraRegistry
from app.incident_manager import IncidentManager
from app.config import load_config

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_root = Path(__file__).parent.parent
cfg = load_config(str(_root / "config.yaml"))
camera_id_cfg = cfg.get("camera", {}).get("id", "CAM-001")
registry = CameraRegistry(str(_root / cfg.get("camera", {}).get("registry", "cameras.yaml")))
incident_mgr = IncidentManager(
    db_path=str(_root / cfg.get("incident", {}).get("db_path", "outputs/incidents.db")),
    camera_registry=registry,
    snapshot_dir=str(_root / cfg.get("evidence", {}).get("snapshot_dir", "outputs/snapshots")),
    evidence_dir=str(_root / cfg.get("evidence", {}).get("video_dir", "outputs/evidence")),
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# SSE subscriber queues
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()


def _broadcast(event_type: str, data: dict):
    msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


# ---------------------------------------------------------------------------
# SSE — real-time push to dashboard
# ---------------------------------------------------------------------------

@app.route("/stream")
def stream():
    def generate():
        q = queue.Queue(maxsize=50)
        with _sse_lock:
            _sse_clients.append(q)
        try:
            # Send initial state immediately
            yield f"event: init\ndata: {json.dumps({'status': 'connected'})}\n\n"
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with _sse_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Background thread: poll DB every 2s and broadcast changes
_last_poll_state: dict = {}

def _poll_loop():
    while True:
        try:
            cameras = incident_mgr.list_cameras()
            incidents = incident_mgr.list_incidents(limit=50)
            active = [i for i in incidents if i["status"] in ("DETECTED", "UNDER_REVIEW")]

            state_sig = json.dumps({
                "cameras": [(c["camera_id"], c["status"]) for c in cameras],
                "active_count": len(active),
                "incidents": [(i["incident_id"], i["status"], i["health_emergency_score"]) for i in active],
            })
            if state_sig != _last_poll_state.get("sig"):
                _last_poll_state["sig"] = state_sig
                _broadcast("update", {
                    "cameras": cameras,
                    "active_incidents": active,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        except Exception:
            pass
        time.sleep(2)

threading.Thread(target=_poll_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# API — Cameras
# ---------------------------------------------------------------------------

@app.route("/api/cameras")
def api_cameras():
    return jsonify(incident_mgr.list_cameras())


@app.route("/api/cameras/<camera_id>")
def api_camera(camera_id):
    cam = incident_mgr.get_camera(camera_id)
    if cam is None:
        abort(404)
    cam["recent_incidents"] = incident_mgr.list_incidents(camera_id=camera_id, limit=10)
    return jsonify(cam)


# ---------------------------------------------------------------------------
# API — Incidents
# ---------------------------------------------------------------------------

@app.route("/api/incidents")
def api_incidents():
    status = request.args.get("status")
    camera_id = request.args.get("camera_id")
    event_type = request.args.get("event_type")
    limit = int(request.args.get("limit", 100))
    return jsonify(incident_mgr.list_incidents(
        status=status, camera_id=camera_id, event_type=event_type, limit=limit
    ))


@app.route("/api/incidents/<incident_id>")
def api_incident(incident_id):
    inc = incident_mgr.get_incident(incident_id)
    if inc is None:
        abort(404)
    return jsonify(inc)


@app.route("/api/incidents/<incident_id>/confirm", methods=["POST"])
def api_confirm(incident_id):
    body = request.get_json(silent=True) or {}
    confirmed_by = body.get("confirmed_by", "Human Operator")
    ok = incident_mgr.confirm(incident_id, confirmed_by=confirmed_by)
    if not ok:
        return jsonify({"error": "confirm failed"}), 400
    inc = incident_mgr.get_incident(incident_id)
    _broadcast("incident_update", {"incident_id": incident_id, "status": "CONFIRMED"})
    return jsonify({"status": "CONFIRMED", "incident": inc})


@app.route("/api/incidents/<incident_id>/dismiss", methods=["POST"])
def api_dismiss(incident_id):
    body = request.get_json(silent=True) or {}
    dismissed_by = body.get("dismissed_by", "Human Operator")
    reason = body.get("reason", "")
    ok = incident_mgr.dismiss(incident_id, dismissed_by=dismissed_by, reason=reason)
    if not ok:
        return jsonify({"error": "dismiss failed"}), 400
    _broadcast("incident_update", {"incident_id": incident_id, "status": "DISMISSED"})
    return jsonify({"status": "DISMISSED"})


@app.route("/api/incidents/<incident_id>/resolve", methods=["POST"])
def api_resolve(incident_id):
    ok = incident_mgr.resolve(incident_id)
    if not ok:
        return jsonify({"error": "resolve failed"}), 400
    _broadcast("incident_update", {"incident_id": incident_id, "status": "RESOLVED"})
    return jsonify({"status": "RESOLVED"})


@app.route("/api/incidents/<incident_id>/snapshot")
def api_snapshot(incident_id):
    inc = incident_mgr.get_incident(incident_id)
    if inc is None:
        abort(404)
    path = inc.get("snapshot_path")
    if not path or not Path(path).exists():
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/incidents/<incident_id>/video")
def api_video(incident_id):
    inc = incident_mgr.get_incident(incident_id)
    if inc is None:
        abort(404)
    path = inc.get("evidence_path")
    if not path or not Path(path).exists():
        abort(404)
    return send_file(path, mimetype="video/mp4")


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    cameras = incident_mgr.list_cameras()
    active = incident_mgr.list_incidents(status="UNDER_REVIEW", limit=20)
    active += incident_mgr.list_incidents(status="DETECTED", limit=20)
    history = incident_mgr.list_incidents(limit=50)
    return render_template("index.html",
                           cameras=cameras,
                           active_incidents=active,
                           history=history)


if __name__ == "__main__":
    host = cfg.get("dashboard", {}).get("host", "127.0.0.1")
    port = cfg.get("dashboard", {}).get("port", 5000)
    print(f"[Dashboard] Starting at http://{host}:{port}")
    print("[Dashboard] Detection loop runs independently in app/main.py")
    app.run(host=host, port=port, debug=False, threaded=True)
