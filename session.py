"""Clinical Assessment Session: Stage 1 Tapping -> Stage 2 Tremor -> Stage 3 Scorecard.

Manages the 10-second trial countdowns, data collection, metric locking,
and MDS-UPDRS risk determination.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

import dsp

STAGE_DURATION = 10.0       # 10 seconds per stage
TRANSITION_DURATION = 3.0   # 3 seconds preparation between Stage 1 and Stage 2


class SessionState:
    IDLE = "IDLE"
    STAGE1_COUNTDOWN = "STAGE1_COUNTDOWN"      # Finger Tapping (10s)
    STAGE1_LOCKED = "STAGE1_LOCKED"            # Locked in, prep for Stage 2 (3s)
    STAGE2_COUNTDOWN = "STAGE2_COUNTDOWN"      # Postural Tremor (10s)
    STAGE3_SCORECARD = "STAGE3_SCORECARD"      # Final Clinical Scorecard


@dataclass
class Stage1Results:
    rate: float = 0.0          # final tap speed (taps/second)
    decrement: float = 0.0     # fractional amplitude loss over trial
    slope: float = 0.0         # fatigue slope
    n_taps: int = 0            # total tap count
    cv: float = 0.0            # hesitation count / rhythm variability
    score: int = 4             # MDS-UPDRS score (0-4)
    label: str = "Severe"


@dataclass
class Stage2Results:
    peak_hz: float = 0.0       # dominant tremor frequency in Hz
    band_ratio: float = 0.0    # fraction of power inside 4-6 Hz tremor band
    rms: float = 0.0           # hand-normalized tremor displacement RMS
    score: int = 0             # MDS-UPDRS score (0-3)
    label: str = "Normal"


class AssessmentSession:
    """State machine governing Stage 1, Stage 2, and the Final Clinical Scorecard."""

    def __init__(self):
        self.state = SessionState.IDLE
        self.state_start_time = 0.0
        self.last_update_t = 0.0

        # Trial data buffers
        self.stage1_t: List[float] = []
        self.stage1_ap: List[float] = []

        self.stage2_t: List[float] = []
        self.stage2_iy: List[float] = []

        # Locked-in results
        self.stage1_res: Optional[Stage1Results] = None
        self.stage2_res: Optional[Stage2Results] = None

        self.is_elevated_risk = False
        self.risk_badge = "NORMAL"

    def start(self, current_time: Optional[float] = None):
        """Start or restart the clinical assessment session."""
        now = current_time if current_time is not None else time.perf_counter()
        self.state = SessionState.STAGE1_COUNTDOWN
        self.state_start_time = now
        self.stage1_t.clear()
        self.stage1_ap.clear()
        self.stage2_t.clear()
        self.stage2_iy.clear()
        self.stage1_res = None
        self.stage2_res = None
        self.is_elevated_risk = False
        self.risk_badge = "NORMAL"

    def reset(self):
        """Return to IDLE state."""
        self.state = SessionState.IDLE
        self.stage1_t.clear()
        self.stage1_ap.clear()
        self.stage2_t.clear()
        self.stage2_iy.clear()
        self.stage1_res = None
        self.stage2_res = None

    def time_remaining(self, current_time: float) -> float:
        """Seconds remaining in the active countdown state."""
        elapsed = current_time - self.state_start_time
        if self.state in (SessionState.STAGE1_COUNTDOWN, SessionState.STAGE2_COUNTDOWN):
            return max(0.0, STAGE_DURATION - elapsed)
        elif self.state == SessionState.STAGE1_LOCKED:
            return max(0.0, TRANSITION_DURATION - elapsed)
        return 0.0

    def progress_fraction(self, current_time: float) -> float:
        """Fraction of countdown completed (0.0 to 1.0)."""
        elapsed = current_time - self.state_start_time
        if self.state in (SessionState.STAGE1_COUNTDOWN, SessionState.STAGE2_COUNTDOWN):
            return float(np.clip(elapsed / STAGE_DURATION, 0.0, 1.0))
        elif self.state == SessionState.STAGE1_LOCKED:
            return float(np.clip(elapsed / TRANSITION_DURATION, 0.0, 1.0))
        return 0.0

    def update(self, t: float, aperture: float, index_y: float):
        """Step the assessment session state machine with the latest frame metrics."""
        self.last_update_t = t

        # --- STAGE 1: FINGER TAPPING ---
        if self.state == SessionState.STAGE1_COUNTDOWN:
            if np.isfinite(aperture):
                self.stage1_t.append(t)
                self.stage1_ap.append(aperture)

            elapsed = t - self.state_start_time
            if elapsed >= STAGE_DURATION:
                # Timer hit zero! Lock in Stage 1 results
                raw_s1 = dsp.analyze_finger_tapping(self.stage1_t, self.stage1_ap)
                self.stage1_res = Stage1Results(
                    rate=float(raw_s1.get("rate", 0.0)),
                    decrement=float(raw_s1.get("decrement", 0.0)),
                    slope=float(raw_s1.get("slope", 0.0)),
                    n_taps=int(raw_s1.get("n_taps", 0)),
                    cv=float(raw_s1.get("cv", 0.0)),
                    score=int(raw_s1.get("score", 4)),
                    label=dsp.SCORE_LABELS.get(raw_s1.get("score", 4), "Severe")
                )
                # Transition to locked banner & brief preparation for Stage 2
                self.state = SessionState.STAGE1_LOCKED
                self.state_start_time = t

        # --- TRANSITION TO STAGE 2 ---
        elif self.state == SessionState.STAGE1_LOCKED:
            elapsed = t - self.state_start_time
            if elapsed >= TRANSITION_DURATION:
                # Transition directly into Stage 2
                self.state = SessionState.STAGE2_COUNTDOWN
                self.state_start_time = t
                self.stage2_t.clear()
                self.stage2_iy.clear()

        # --- STAGE 2: POSTURAL TREMOR ---
        elif self.state == SessionState.STAGE2_COUNTDOWN:
            if np.isfinite(index_y):
                self.stage2_t.append(t)
                self.stage2_iy.append(index_y)

            elapsed = t - self.state_start_time
            if elapsed >= STAGE_DURATION:
                # Timer hit zero! Lock in Stage 2 tremor spectrum results
                raw_s2 = dsp.analyze_tremor_spectrum(self.stage2_t, self.stage2_iy)
                self.stage2_res = Stage2Results(
                    peak_hz=float(raw_s2.get("peak_hz", 0.0)),
                    band_ratio=float(raw_s2.get("band_ratio", 0.0)),
                    rms=float(raw_s2.get("rms", 0.0)),
                    score=int(raw_s2.get("score", 0)),
                    label=dsp.SCORE_LABELS.get(raw_s2.get("score", 0), "Normal")
                )

                # Determine overall clinical risk
                # Elevated risk if tapping score >= 2 (mild/moderate/severe bradykinesia)
                # OR tremor score >= 1 with significant 4-6 Hz band concentration
                s1_elevated = (self.stage1_res is not None and self.stage1_res.score >= 2)
                s2_elevated = (self.stage2_res.score >= 1 or
                               (self.stage2_res.band_ratio >= dsp.TREMOR_BAND_RATIO_MIN and
                                self.stage2_res.rms >= dsp.TREMOR_RMS[0]))

                if s1_elevated or s2_elevated:
                    self.is_elevated_risk = True
                    self.risk_badge = "ELEVATED RISK / ASYMMETRY DETECTED"
                else:
                    self.is_elevated_risk = False
                    self.risk_badge = "NORMAL"

                self.state = SessionState.STAGE3_SCORECARD
                self.state_start_time = t
