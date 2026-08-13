"""
risk_engine.py — Health Emergency Risk Assessment Engine.

PURPOSE
-------
Evaluates post-fall behavior to assess whether a confirmed fall may
represent a potential health emergency. This is a SEPARATE layer from
fall detection — a fall alone does not trigger an emergency.

RISK SCORING FORMULA
--------------------
health_emergency_score = (
    0.30 * fall_evidence_score   +  # strength of fall signal from Phase 3
    0.25 * time_down_score       +  # how long person has been in low posture
    0.25 * inactivity_score      +  # post-fall stillness / lack of movement
    0.20 * no_recovery_score        # absence of recovery behavior
)

All sub-scores are in [0.0, 1.0].
Final score >= emergency_score_threshold AND persists for
emergency_persist_seconds → POTENTIAL_HEALTH_EMERGENCY.

IMPORTANT: This system detects POTENTIAL HEALTH EMERGENCY based on
visual behavior only. It is NOT a medical diagnosis tool.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskState(Enum):
    NORMAL              = "NORMAL"
    MONITORING          = "MONITORING"           # fall detected, watching
    POSSIBLE_EMERGENCY  = "POSSIBLE_EMERGENCY"   # score rising
    POTENTIAL_HEALTH_EMERGENCY = "POTENTIAL_HEALTH_EMERGENCY"
    RECOVERED           = "RECOVERED"


@dataclass
class RiskResult:
    state: RiskState = RiskState.NORMAL
    health_emergency_score: float = 0.0
    fall_evidence_score: float = 0.0
    time_down_score: float = 0.0
    inactivity_score: float = 0.0
    no_recovery_score: float = 0.0
    state_entered_at: float = field(default_factory=time.time)
    above_threshold_since: Optional[float] = None   # when score first crossed threshold
    below_threshold_since: Optional[float] = None   # when score first dropped below threshold
    incident_id: Optional[str] = None
    fall_confirmed_at: Optional[float] = None       # wall-clock time of FALL_CONFIRMED

    @property
    def state_duration(self) -> float:
        return time.time() - self.state_entered_at

    @property
    def time_since_fall(self) -> float:
        if self.fall_confirmed_at is None:
            return 0.0
        return time.time() - self.fall_confirmed_at


class RiskEngine:
    """
    Per-person health emergency risk engine.
    Consumes FallResult from Phase 3 and produces RiskResult.
    One instance per tracked person — completely independent.
    """

    def __init__(self, cfg: dict, camera_id: str = "CAMERA_01"):
        re = cfg["risk_engine"]
        self.emergency_threshold    = re["emergency_score_threshold"]
        self.emergency_persist_s    = re["emergency_persist_seconds"]
        self.emergency_clear_s      = re["emergency_clear_seconds"]
        self.time_down_ref_s        = re["time_down_reference_seconds"]
        self.inactivity_window      = re["inactivity_window_frames"]
        self.inactivity_still_px    = re["inactivity_stillness_px"]
        self.recovery_motion_thresh = re["recovery_motion_threshold"]
        self.camera_id              = camera_id
        self.result                 = RiskResult()

    def update(self, person_state, fall_result, logger=None) -> RiskResult:
        """
        Called every frame. Consumes PersonState + FallResult.
        Returns updated RiskResult.
        """
        from app.fall_detector import FallState

        snap = person_state.latest
        if snap is None:
            return self.result

        fall_state = fall_result.state if fall_result else FallState.NORMAL

        # Only activate risk engine once a fall has been confirmed
        active_fall_states = {
            FallState.FALL_CONFIRMED,
            FallState.POST_FALL_MONITORING,
            FallState.RECOVERED,
        }

        if fall_state not in active_fall_states:
            # If we were monitoring and fall evidence dropped, handle recovery
            if self.result.state in (RiskState.MONITORING, RiskState.POSSIBLE_EMERGENCY,
                                     RiskState.POTENTIAL_HEALTH_EMERGENCY):
                if fall_state == FallState.NORMAL:
                    self._transition(RiskState.RECOVERED, logger, person_state.person_id)
            elif self.result.state == RiskState.RECOVERED:
                if self.result.state_duration >= 2.0:
                    self._transition(RiskState.NORMAL, logger, person_state.person_id)
                    self.result.fall_confirmed_at = None
                    self.result.incident_id = None
            return self.result

        # Record when fall was first confirmed
        if self.result.fall_confirmed_at is None:
            self.result.fall_confirmed_at = time.time()
            self.result.incident_id = str(uuid.uuid4())[:8]
            if self.result.state == RiskState.NORMAL:
                self._transition(RiskState.MONITORING, logger, person_state.person_id)

        history = list(person_state.history)

        fall_evidence_score = self._fall_evidence_score(fall_result)
        time_down_score     = self._time_down_score()
        inactivity_score    = self._inactivity_score(history)
        no_recovery_score   = self._no_recovery_score(history)

        health_score = (
            0.30 * fall_evidence_score +
            0.25 * time_down_score     +
            0.25 * inactivity_score    +
            0.20 * no_recovery_score
        )

        self.result.fall_evidence_score     = round(fall_evidence_score, 3)
        self.result.time_down_score         = round(time_down_score, 3)
        self.result.inactivity_score        = round(inactivity_score, 3)
        self.result.no_recovery_score       = round(no_recovery_score, 3)
        self.result.health_emergency_score  = round(health_score, 3)

        self._advance_state(health_score, fall_result, logger, person_state.person_id)
        return self.result

    # ------------------------------------------------------------------
    # Sub-score calculations
    # ------------------------------------------------------------------

    def _fall_evidence_score(self, fall_result) -> float:
        """
        Directly uses the fall_score from Phase 3.
        Strong fall evidence → high score.
        A weak or uncertain fall → lower contribution.
        """
        if fall_result is None:
            return 0.0
        return fall_result.fall_score

    def _time_down_score(self) -> float:
        """
        How long has the person been in a confirmed-fall state?
        Normalized against time_down_reference_seconds.
        15s down = score 1.0. Longer = capped at 1.0.
        """
        elapsed = self.result.time_since_fall
        return min(1.0, elapsed / (self.time_down_ref_s + 1e-6))

    def _inactivity_score(self, history: list) -> float:
        """
        Measures post-fall stillness.
        High score = person is not moving (consistent with incapacitation).
        Low score = person is moving (consistent with conscious activity).
        """
        window = history[-self.inactivity_window:]
        if len(window) < 3:
            return 0.0
        deltas = [s.position_delta for s in window if s.position_delta is not None]
        if not deltas:
            return 0.5  # unknown — neutral
        avg_delta = sum(deltas) / len(deltas)
        # Invert: stillness → high score
        return max(0.0, min(1.0, 1.0 - avg_delta / (self.inactivity_still_px * 2)))

    def _no_recovery_score(self, history: list) -> float:
        """
        Absence of recovery behavior.
        If person is moving significantly → recovery likely → low score.
        If person is completely still → no recovery → high score.
        """
        window = history[-self.inactivity_window:]
        if len(window) < 3:
            return 0.5
        deltas = [s.position_delta for s in window if s.position_delta is not None]
        if not deltas:
            return 0.5
        avg_delta = sum(deltas) / len(deltas)
        # High movement = recovery attempt = low no_recovery score
        return max(0.0, min(1.0, 1.0 - avg_delta / self.recovery_motion_thresh))

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _advance_state(self, score: float, fall_result, logger, pid: int):
        now = time.time()

        if self.result.state == RiskState.MONITORING:
            if score >= self.emergency_threshold:
                if self.result.above_threshold_since is None:
                    self.result.above_threshold_since = now
                elif (now - self.result.above_threshold_since) >= self.emergency_persist_s:
                    self._transition(RiskState.POSSIBLE_EMERGENCY, logger, pid)
            else:
                self.result.above_threshold_since = None

        elif self.result.state == RiskState.POSSIBLE_EMERGENCY:
            if score >= self.emergency_threshold:
                self._transition(RiskState.POTENTIAL_HEALTH_EMERGENCY, logger, pid)
            elif score < self.emergency_threshold * 0.8:
                self._transition(RiskState.MONITORING, logger, pid)
                self.result.above_threshold_since = None

        elif self.result.state == RiskState.POTENTIAL_HEALTH_EMERGENCY:
            if score < self.emergency_threshold:
                if self.result.below_threshold_since is None:
                    self.result.below_threshold_since = now
                elif (now - self.result.below_threshold_since) >= self.emergency_clear_s:
                    self._transition(RiskState.RECOVERED, logger, pid)
                    self.result.below_threshold_since = None
            else:
                self.result.below_threshold_since = None

        elif self.result.state == RiskState.RECOVERED:
            if self.result.state_duration >= 2.0:
                self._transition(RiskState.NORMAL, logger, pid)
                self.result.fall_confirmed_at = None
                self.result.incident_id = None

    def _transition(self, new_state: RiskState, logger, pid: int):
        from app.event_log import log_transition
        prev = self.result.state
        self.result.state = new_state
        self.result.state_entered_at = time.time()

        if logger is not None:
            log_transition(logger, pid, f"RISK:{prev.value}", f"RISK:{new_state.value}", {
                "risk": self.result.health_emergency_score,
                "fall_ev": self.result.fall_evidence_score,
                "time_dn": self.result.time_down_score,
                "inact": self.result.inactivity_score,
                "no_rec": self.result.no_recovery_score,
            })

    def to_event_dict(self, pid: int, fall_result) -> dict:
        """Structured event object for Phase 5 reporting consumption."""
        return {
            "event_id":      self.result.incident_id,
            "camera_id":     self.camera_id,
            "person_id":     pid,
            "start_time":    self.result.fall_confirmed_at,
            "current_state": self.result.state.value,
            "risk_score":    self.result.health_emergency_score,
            "fall_score":    fall_result.fall_score if fall_result else 0.0,
            "duration_down": round(self.result.time_since_fall, 1),
            "reason": {
                "fall_evidence": self.result.fall_evidence_score,
                "time_down":     self.result.time_down_score,
                "inactivity":    self.result.inactivity_score,
                "no_recovery":   self.result.no_recovery_score,
            },
        }
