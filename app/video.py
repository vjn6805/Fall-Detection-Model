import cv2


def open_source(cfg: dict):
    """
    Open a cv2.VideoCapture from config.
    Returns (cap, source_label) or raises RuntimeError.
    """
    src_type = cfg["source"]["type"]

    if src_type == "webcam":
        device_id = cfg["source"].get("webcam_id", 0)
        cap = cv2.VideoCapture(device_id)
        label = f"Webcam {device_id}"
    elif src_type == "rtsp":
        url = cfg["source"].get("rtsp_url", "")
        if not url:
            raise RuntimeError("RTSP URL is empty in config.yaml")
        cap = cv2.VideoCapture(url)
        label = f"RTSP: {url}"
    elif src_type == "video":
        path = cfg["source"].get("path", "")
        if not path:
            raise RuntimeError("Video path is empty in config.yaml")
        cap = cv2.VideoCapture(path)
        label = f"File: {path}"
    else:
        raise RuntimeError(f"Unknown source type: '{src_type}'")

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source — {label}")

    return cap, label


def read_frame(cap: cv2.VideoCapture):
    """
    Read one frame. Returns (frame, ok).
    ok=False means end-of-stream or decode error.
    """
    ok, frame = cap.read()
    return frame, ok
