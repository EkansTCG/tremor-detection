"""Clinical Assessment Session: Decoupled Stage 1 Tapping & Stage 2 Tremor Sessions.

Provides:
- TappingSession: 10-second finger tapping countdown, aperture tracking, dsp.analyze_finger_tapping
- TremorSession: 10-second postural tremor countdown, index height tracking, dsp.analyze_tremor_spectrum
- AssessmentSession: Unified session supporting both decoupled modes and sequential runs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

import dsp

STAGE_DURATION = 10.0       # 10 seconds per trial
TRANSITION_DURATION = 3.0   # 3 seconds between stages for legacy full sequence


class SessionState:
    IDLE = "IDLE"
    COUNTDOWN = "COUNTDOWN"                    # Generic active countdown
    STAGE1_COUNTDOWN = "STAGE1_COUNTDOWN"      # Finger Tapping (10s)
    STAGE1_LOCKED = "STAGE1_LOCKED"            # Locked in, prep for Stage 2 (3s)
    STAGE2_COUNTDOWN = "STAGE2_COUNTDOWN"      # Postural Tremor (10s)
    STAGE3_SCORECARD = "STAGE3_SCORECARD"      # Final Clinical Scorecard
    COMPLETED = "COMPLETED"                    # Decoupled test complete


@dataclass
class Stage1Results:
    rate: float = 0.0          # final tap speed (taps/second)
    decrement: float = 0.0     # fractional amplitude loss over trial
    slope: float = 0.0         # fatigue slope
    n_taps: int = 0            # total tap count
    cv: float = 0.0            # hesitation count / rhythm variability
    score: int = 4             # MDS-UPDRS score (0-4)
    label: str = "Severe"
    is_elevated_risk: bool = False
    risk_badge: str = "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        elevated = (self.score >= 2 or self.decrement >= 0.35)
        badge = "ELEVATED RISK / ASYMMETRY DETECTED" if elevated else "NORMAL"
        return {
            "test_type": "tapping",
            "rate": round(self.rate, 2),
            "decrement": round(self.decrement, 3),
            "decrement_pct": round(self.decrement * 100, 1),
            "slope": round(self.slope, 4),
            "n_taps": self.n_taps,
            "cv": round(self.cv, 3),
            "score": self.score,
            "label": self.label,
            "is_elevated_risk": 1 if elevated else 0,
            "risk_badge": badge
        }


@dataclass
class Stage2Results:
    peak_hz: float = 0.0       # dominant tremor frequency in Hz
    band_ratio: float = 0.0    # fraction of power inside 4-6 Hz tremor band
    rms: float = 0.0           # hand-normalized tremor displacement RMS
    excursion: float = 0.0     # peak-to-peak excursion
    velocity_rms: float = 0.0  # RMS of movement velocity
    score: int = 0             # MDS-UPDRS score (0-4)
    label: str = "Normal"
    is_elevated_risk: bool = False
    risk_badge: str = "NORMAL"
    tremor_type: str = "Physiological (Normal)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_type": "tremor",
            "peak_hz": round(self.peak_hz, 2),
            "band_ratio": round(self.band_ratio, 3),
            "band_ratio_pct": round(self.band_ratio * 100, 1),
            "rms": round(self.rms, 4),
            "excursion": round(self.excursion, 4),
            "velocity_rms": round(self.velocity_rms, 3),
            "score": self.score,
            "label": self.label,
            "is_elevated_risk": 1 if self.is_elevated_risk else 0,
            "risk_badge": self.risk_badge,
            "tremor_type": self.tremor_type
        }


class TappingSession:
    """Decoupled 10-second Stage 1 Finger Tapping Session (MDS-UPDRS 3.4)."""

    def __init__(self):
        self.state = SessionState.IDLE
        self.state_start_time: Optional[float] = None
        self.last_update_t: float = 0.0
        self.t_buf: List[float] = []
        self.ap_buf: List[float] = []
        self.results: Optional[Stage1Results] = None
        self.is_saved: bool = False
        self.saved_id: Optional[str] = None

    def start(self, current_time: Optional[float] = None):
        self.state = SessionState.COUNTDOWN
        self.state_start_time = current_time  # If None, set on first update(t) frame
        self.last_update_t = current_time or 0.0
        self.t_buf.clear()
        self.ap_buf.clear()
        self.results = None
        self.is_saved = False
        self.saved_id = None

    def reset(self):
        self.state = SessionState.IDLE
        self.state_start_time = None
        self.t_buf.clear()
        self.ap_buf.clear()
        self.results = None
        self.is_saved = False
        self.saved_id = None

    def time_remaining(self, current_time: Optional[float] = None) -> float:
        if self.state == SessionState.COUNTDOWN:
            if self.state_start_time is None:
                return STAGE_DURATION
            ref_t = current_time if current_time is not None else self.last_update_t
            elapsed = ref_t - self.state_start_time
            return max(0.0, STAGE_DURATION - elapsed)
        return 0.0

    def progress_fraction(self, current_time: Optional[float] = None) -> float:
        if self.state == SessionState.COUNTDOWN:
            if self.state_start_time is None:
                return 0.0
            ref_t = current_time if current_time is not None else self.last_update_t
            elapsed = ref_t - self.state_start_time
            return float(np.clip(elapsed / STAGE_DURATION, 0.0, 1.0))
        return 1.0 if self.state == SessionState.COMPLETED else 0.0

    def is_completed(self) -> bool:
        return self.state == SessionState.COMPLETED

    def update(self, t: float, aperture: float) -> Optional[Stage1Results]:
        self.last_update_t = t
        if self.state != SessionState.COUNTDOWN:
            return self.results

        # Anchor timer to tracker timestamp t
        if self.state_start_time is None:
            self.state_start_time = t

        if np.isfinite(aperture):
            self.t_buf.append(t)
            self.ap_buf.append(aperture)

        elapsed = t - self.state_start_time
        if elapsed >= STAGE_DURATION:
            # 10s trial complete: analyze tapping
            raw_s1 = dsp.analyze_finger_tapping(self.t_buf, self.ap_buf)
            score = int(raw_s1.get("score", 4))
            decr = float(raw_s1.get("decrement", 0.0))
            elevated = (score >= 2 or decr >= 0.35)
            badge = "ELEVATED RISK / ASYMMETRY DETECTED" if elevated else "NORMAL"

            self.results = Stage1Results(
                rate=float(raw_s1.get("rate", 0.0)),
                decrement=decr,
                slope=float(raw_s1.get("slope", 0.0)),
                n_taps=int(raw_s1.get("n_taps", 0)),
                cv=float(raw_s1.get("cv", 0.0)),
                score=score,
                label=dsp.SCORE_LABELS.get(score, "Severe"),
                is_elevated_risk=elevated,
                risk_badge=badge
            )
            self.state = SessionState.COMPLETED
            return self.results
        return None


class TremorSession:
    """Decoupled 10-second Stage 2 Postural Tremor Session (MDS-UPDRS 3.15)."""

    def __init__(self):
        self.state = SessionState.IDLE
        self.state_start_time: Optional[float] = None
        self.last_update_t: float = 0.0
        self.t_buf: List[float] = []
        self.iy_buf: List[float] = []
        self.ix_buf: List[float] = []
        self.results: Optional[Stage2Results] = None
        self.is_saved: bool = False
        self.saved_id: Optional[str] = None

    def start(self, current_time: Optional[float] = None):
        self.state = SessionState.COUNTDOWN
        self.state_start_time = current_time  # Set on first update(t) frame if None
        self.last_update_t = current_time or 0.0
        self.t_buf.clear()
        self.iy_buf.clear()
        self.ix_buf.clear()
        self.results = None
        self.is_saved = False
        self.saved_id = None

    def reset(self):
        self.state = SessionState.IDLE
        self.state_start_time = None
        self.t_buf.clear()
        self.iy_buf.clear()
        self.ix_buf.clear()
        self.results = None
        self.is_saved = False
        self.saved_id = None

    def time_remaining(self, current_time: Optional[float] = None) -> float:
        if self.state == SessionState.COUNTDOWN:
            if self.state_start_time is None:
                return STAGE_DURATION
            ref_t = current_time if current_time is not None else self.last_update_t
            elapsed = ref_t - self.state_start_time
            return max(0.0, STAGE_DURATION - elapsed)
        return 0.0

    def progress_fraction(self, current_time: Optional[float] = None) -> float:
        if self.state == SessionState.COUNTDOWN:
            if self.state_start_time is None:
                return 0.0
            ref_t = current_time if current_time is not None else self.last_update_t
            elapsed = ref_t - self.state_start_time
            return float(np.clip(elapsed / STAGE_DURATION, 0.0, 1.0))
        return 1.0 if self.state == SessionState.COMPLETED else 0.0

    def is_completed(self) -> bool:
        return self.state == SessionState.COMPLETED

    def update(self, t: float, index_y: float, index_x: Optional[float] = None) -> Optional[Stage2Results]:
        self.last_update_t = t
        if self.state != SessionState.COUNTDOWN:
            return self.results

        # Anchor timer to tracker timestamp t
        if self.state_start_time is None:
            self.state_start_time = t

        if np.isfinite(index_y):
            self.t_buf.append(t)
            self.iy_buf.append(index_y)
            if index_x is not None and np.isfinite(index_x):
                self.ix_buf.append(index_x)

        elapsed = t - self.state_start_time
        if elapsed >= STAGE_DURATION:
            # 10s trial complete: analyze multi-axis tremor spectrum
            x_coords = self.ix_buf if len(self.ix_buf) == len(self.t_buf) else None
            raw_s2 = dsp.analyze_tremor_spectrum(self.t_buf, self.iy_buf, x_coord=x_coords)

            self.results = Stage2Results(
                peak_hz=float(raw_s2.get("peak_hz", 0.0)),
                band_ratio=float(raw_s2.get("band_ratio", 0.0)),
                rms=float(raw_s2.get("rms", 0.0)),
                excursion=float(raw_s2.get("excursion", 0.0)),
                velocity_rms=float(raw_s2.get("velocity_rms", 0.0)),
                score=int(raw_s2.get("score", 0)),
                label=str(raw_s2.get("label", "Normal")),
                is_elevated_risk=bool(raw_s2.get("is_elevated_risk", False)),
                risk_badge=str(raw_s2.get("risk_badge", "NORMAL")),
                tremor_type=str(raw_s2.get("tremor_type", "Physiological (Normal)"))
            )
            self.state = SessionState.COMPLETED
            return self.results
        return None



class AssessmentSession:
    """State machine governing Stage 1, Stage 2, or both."""

    def __init__(self, mode: str = "full"):
        self.mode = mode  # "full", "tapping", or "tremor"
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
        if self.mode == "tremor":
            self.state = SessionState.STAGE2_COUNTDOWN
        else:
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

                if self.mode == "tapping":
                    self.state = SessionState.STAGE3_SCORECARD
                else:
                    self.state = SessionState.STAGE1_LOCKED
                self.state_start_time = t

        # --- TRANSITION TO STAGE 2 (Full mode only) ---
        elif self.state == SessionState.STAGE1_LOCKED:
            elapsed = t - self.state_start_time
            if elapsed >= TRANSITION_DURATION:
                self.state = SessionState.STAGE2_COUNTDOWN
                self.state_start_time = t
                self.stage2_t.clear()
                self.stage2_iy.clear()

        # --- STAGE 2: POSTURAL TREMOR ---
        # --- STAGE 2: POSTURAL TREMOR ---
        elif self.state == SessionState.STAGE2_COUNTDOWN:
            if np.isfinite(index_y):
                self.stage2_t.append(t)
                self.stage2_iy.append(index_y)

            elapsed = t - self.state_start_time
            if elapsed >= STAGE_DURATION:
                raw_s2 = dsp.analyze_tremor_spectrum(self.stage2_t, self.stage2_iy)
                s2_elevated = bool(raw_s2.get("is_elevated_risk", False))

                self.stage2_res = Stage2Results(
                    peak_hz=float(raw_s2.get("peak_hz", 0.0)),
                    band_ratio=float(raw_s2.get("band_ratio", 0.0)),
                    rms=float(raw_s2.get("rms", 0.0)),
                    excursion=float(raw_s2.get("excursion", 0.0)),
                    velocity_rms=float(raw_s2.get("velocity_rms", 0.0)),
                    score=int(raw_s2.get("score", 0)),
                    label=str(raw_s2.get("label", "Normal")),
                    is_elevated_risk=s2_elevated,
                    risk_badge=str(raw_s2.get("risk_badge", "NORMAL")),
                    tremor_type=str(raw_s2.get("tremor_type", "Physiological (Normal)"))
                )

                s1_elevated = (self.stage1_res is not None and self.stage1_res.score >= 2)

                if s1_elevated or s2_elevated:
                    self.is_elevated_risk = True
                    self.risk_badge = "ELEVATED RISK / ASYMMETRY DETECTED"
                else:
                    self.is_elevated_risk = False
                    self.risk_badge = "NORMAL"

                self.state = SessionState.STAGE3_SCORECARD
                self.state_start_time = t
