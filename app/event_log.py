import logging
import time
from pathlib import Path


def setup_event_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fall_events")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("[EVENT] %(message)s"))
        logger.addHandler(sh)
    return logger


def log_transition(logger: logging.Logger, pid: int, from_state: str, to_state: str, scores: dict):
    score_str = " ".join(f"{k}={v:.2f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in scores.items())
    logger.info(f"Person#{pid} {from_state} -> {to_state} | {score_str}")
