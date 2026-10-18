# pyrefly: ignore[missing-import]
"""HUD: Clinical Motor Assessment Overlays, Graphs, and Decoupled Test Views.

Supports:
- Stage 1 — Finger Tapping Test (Aperture tracking, scrolling waveform, 10s countdown, Stage 1 Scorecard)
- Stage 2 — Postural Tremor Spectral Test (Index tip height, live spectrum, 10s countdown, Stage 2 Scorecard)
- Unified Assessment HUD for backward compatibility
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional, Union

# pyrefly: ignore[missing-import]
import cv2
import numpy as np

from session import AssessmentSession, SessionState, TappingSession, TremorSession, Stage1Results, Stage2Results
from vision import (HandFrame, HandTracker, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP,
                    THUMB_TIP, WRIST)

WINDOW_FPS = 30.0
WINDOW_SIZE = 300                 # 300 samples / 30 fps = 10 s of history
RATIO_OPEN = 1.0                  # aperture drawn as fully "open" (green) at/above this


def dist2(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2D pixel points."""
    return float(np.linalg.norm(np.asarray(a[:2], float) - np.asarray(b[:2], float)))


def palm_ratio(hf: HandFrame) -> float:
    """Thumb-index pinch distance / palm size, measured in 2D image coordinates (pixels).

    Normalizes 2D thumb-index distance by 2D palm size (average of wrist->middle MCP
    and index MCP->pinky MCP in pixels), eliminating 3D depth-estimation jitter.
    """
    px = hf.pixels
    palm = 0.5 * (dist2(px[WRIST], px[MIDDLE_MCP]) + dist2(px[INDEX_MCP], px[PINKY_MCP]))
    if palm < 1e-6:
        return float("nan")
    return dist2(px[THUMB_TIP], px[INDEX_TIP]) / palm


def index_tip_height(hf: HandFrame) -> float:
    """Index fingertip vertical height (up = positive), normalized by hand size in pixels.

    Uses image space because MediaPipe world joints are hand-centered and would hide
    whole-hand tremor. Dividing by on-screen hand size makes it distance-invariant, matching
    the hand-size-normalized TREMOR_RMS cutoffs in dsp.py.
    """
    size_px = hf.extras.get("hand_size_px", 0.0)
    if size_px < 1e-6:
        return float("nan")
    return float(-hf.pixels[INDEX_TIP, 1] / size_px)   # image y grows downward -> negate


def index_tip_coords(hf: HandFrame) -> tuple[float, float]:
    """Index fingertip (x, y) coordinates normalized by hand size in pixels for omnidirectional tremor tracking."""
    size_px = hf.extras.get("hand_size_px", 0.0)
    if size_px < 1e-6:
        return float("nan"), float("nan")
    return float(hf.pixels[INDEX_TIP, 0] / size_px), float(-hf.pixels[INDEX_TIP, 1] / size_px)


class RollingWindow:
    """Three parallel sliding-window queues, max WINDOW_SIZE samples each, sampled at ~WINDOW_FPS.

    Faster frames are skipped so the window always spans about WINDOW_SIZE / WINDOW_FPS seconds.
    """

    def __init__(self, maxlen: int = WINDOW_SIZE, fps: float = WINDOW_FPS):
        self.maxlen = maxlen
        self.period = 1.0 / fps
        self.timestamps: deque[float] = deque(maxlen=maxlen)   # seconds
        self.aperture: deque[float] = deque(maxlen=maxlen)     # Stage 1: finger tapping
        self.index_y: deque[float] = deque(maxlen=maxlen)      # Stage 2: tremor
        self._next_t = None

    def push(self, t: float, aperture: float, index_y: float) -> bool:
        """Append one sample to all three queues. Returns False if skipped to hold 30 fps."""
        if self._next_t is not None and t < self._next_t - 0.25 * self.period:
            return False
        # schedule the next slot; resync if we fell behind (dropped frames)
        self._next_t = max((self._next_t or t) + self.period, t)
        self.timestamps.append(t)
        self.aperture.append(aperture)
        self.index_y.append(index_y)
        return True

    def clear(self):
        self.timestamps.clear(); self.aperture.clear(); self.index_y.clear()
        self._next_t = None

    def __len__(self):
        return len(self.timestamps)

    @property
    def full(self) -> bool:
        return len(self) == self.maxlen

    @property
    def span(self) -> float:
        return self.timestamps[-1] - self.timestamps[0] if len(self) > 1 else 0.0

    def arrays(self):
        """Snapshot as numpy arrays (timestamps, aperture, index_y)."""
        return (np.fromiter(self.timestamps, float), np.fromiter(self.aperture, float),
                np.fromiter(self.index_y, float))


# ------------------------------------------------------------------ Technica 2026: Dare to Dive Palette (BGR for OpenCV)
COLOR_OCEAN_NAVY = (45, 22, 8)       # Deep Ocean Navy BGR (#08162d)
COLOR_OCEAN_MID  = (90, 46, 15)      # Mid Ocean BGR (#0f2e5a)
COLOR_HOT_PINK   = (182, 114, 244)   # Coral / Sunset Pink BGR (#f472b6)
COLOR_AQUA_CYAN  = (255, 220, 56)    # Ocean Wave Aqua Cyan BGR (#38dcff)
COLOR_NEON_CYAN  = (255, 229, 0)     # Bioluminescent Cyan BGR (#00e5ff)

# Strict 4-Tier Clinical Severity Colors (BGR format for OpenCV)
TIER_BGR = {
    0: (166, 184, 20),   # Score 0: Seafoam Teal (20, 184, 166)
    1: (8, 179, 234),    # Score 1: Soft Coral / Sunlit Gold (234, 179, 8) - NOT RED!
    2: (22, 115, 249),   # Score 2: Warm Sunset Orange (249, 115, 22) - NOT RED!
    3: (94, 63, 244),    # Score 3: Deep Crimson Red (244, 63, 94)
    4: (94, 63, 244),    # Score 4: Deep Crimson Red (244, 63, 94)
}

def get_tier_color(score: int) -> tuple[int, int, int]:
    return TIER_BGR.get(min(max(int(score), 0), 4), (166, 184, 20))

def get_tier_label(score: int, is_tremor: bool = False) -> str:
    if score == 0:
        return "NORMAL / CALM WATERS"
    elif score == 1:
        return "SLIGHT / MILD EMERGENCE"
    elif score == 2:
        return "MILD MOTOR FLUCTUATION"
    else:
        return "ELEVATED RISK / ACTIVE TREMOR" if is_tremor else "ELEVATED RISK / SEVERE ASYMMETRY"


# ------------------------------------------------------------------ drawing helpers
def _ratio_color(r: float):
    """BGR color transitioning from Ocean Aqua Cyan to Ribbon Pink."""
    if not np.isfinite(r):
        return (128, 128, 128)
    k = float(np.clip(r / RATIO_OPEN, 0.0, 1.0))
    # Blend from Aqua Cyan (255, 220, 56) to Ribbon Pink (182, 114, 244)
    b = int(255 * (1 - k) + 182 * k)
    g = int(220 * (1 - k) + 114 * k)
    r_val = int(56 * (1 - k) + 244 * k)
    return (b, g, r_val)


def draw_pinch_cue(frame, hf: HandFrame, ratio: float):
    """Colored thumb-index line and ring with Ocean & Ribbon Pink styling."""
    p1, p2 = hf.pixels[THUMB_TIP], hf.pixels[INDEX_TIP]
    color = _ratio_color(ratio)
    cv2.line(frame, tuple(p1), tuple(p2), color, 3, cv2.LINE_AA)
    mid = tuple(((p1 + p2) // 2).astype(int))
    radius = int(8 + 26 * np.clip(ratio / RATIO_OPEN, 0, 1.5)) if np.isfinite(ratio) else 8
    cv2.circle(frame, mid, radius, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
    cv2.circle(frame, mid, 4, COLOR_HOT_PINK, -1, cv2.LINE_AA)
    cv2.putText(frame, f"Aperture: {ratio:.2f}", (mid[0] + radius + 8, mid[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)


def draw_tremor_cue(frame, hf: HandFrame, iy: float):
    """Visual crosshairs and stability beacon with Ocean Aqua Cyan and Hot Pink."""
    p = tuple(hf.pixels[INDEX_TIP])
    cv2.circle(frame, p, 14, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
    cv2.circle(frame, p, 4, COLOR_HOT_PINK, -1, cv2.LINE_AA)
    cv2.line(frame, (p[0] - 22, p[1]), (p[0] + 22, p[1]), COLOR_AQUA_CYAN, 1, cv2.LINE_AA)
    cv2.line(frame, (p[0], p[1] - 22), (p[0], p[1] + 22), COLOR_AQUA_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Y: {iy:+.3f}", (p[0] + 18, p[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_HOT_PINK, 2, cv2.LINE_AA)


def _draw_series_panel(frame, win: RollingWindow, values: np.ndarray, x0: int, y0: int,
                       w: int, h: int, label: str, color, lo: float, hi: float,
                       ref: float | None = None):
    """Translucent scrolling plot with Deep Ocean Navy background & Pink/Cyan accents."""
    H, W = frame.shape[:2]
    if y0 + h > H or x0 + w > W or y0 < 0 or x0 < 0:
        return

    # Deep Ocean Navy glassmorphic panel
    roi = frame[y0:y0 + h, x0:x0 + w]
    navy_box = np.full_like(roi, 0)
    navy_box[:] = COLOR_OCEAN_NAVY
    cv2.addWeighted(roi, 0.28, navy_box, 0.72, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), COLOR_HOT_PINK, 1, cv2.LINE_AA)

    gy0, gy1 = y0 + 26, y0 + h - 18
    span = max(hi - lo, 1e-6)
    to_y = lambda v: gy1 - (v - lo) / span * (gy1 - gy0)

    if ref is not None:
        ry = int(to_y(ref))
        if gy0 <= ry <= gy1:
            cv2.line(frame, (x0 + 4, ry), (x0 + w - 4, ry), (100, 70, 30), 1, cv2.LINE_AA)

    if len(values) > 1:
        xs = x0 + 4 + (np.arange(len(values)) + (win.maxlen - len(values))) * (w - 8) / (win.maxlen - 1)
        ok = np.isfinite(values)
        ys = to_y(np.where(ok, values, lo))
        pts = np.column_stack([xs, np.clip(ys, gy0, gy1)]).astype(np.int32)
        start = None
        for i in range(len(pts) + 1):
            if i < len(pts) and ok[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start > 1:
                    cv2.polylines(frame, [pts[start:i]], False, color, 2, cv2.LINE_AA)
                start = None
        if ok[-1]:
            cv2.circle(frame, tuple(pts[-1]), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(pts[-1]), 6, color, 1, cv2.LINE_AA)

    cur = f"{values[-1]:+.3f}" if len(values) and np.isfinite(values[-1]) else "--"
    cv2.putText(frame, f"{label}  {cur}", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (240, 240, 240), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ Decoupled Test Graphs
def draw_tapping_graph(frame, win: RollingWindow, w: int = 460, h: int = 130, margin: int = 20):
    """Scrolling aperture plot with Aqua Cyan trace for /test/tapping."""
    H = frame.shape[0]
    x0 = margin
    y0 = H - margin - h
    _, ap, _ = win.arrays()

    fa = ap[np.isfinite(ap)]
    ap_hi = max(1.5, float(fa.max()) * 1.1) if fa.size else 1.5
    _draw_series_panel(frame, win, ap, x0, y0, w, h, "MDS-UPDRS 3.4  Aperture Kinematics", COLOR_AQUA_CYAN,
                       0.0, ap_hi, ref=RATIO_OPEN)

    cv2.putText(frame, f"{len(win)}/{win.maxlen} ({win.span:4.1f}s)", (x0 + w - 140, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HOT_PINK, 1, cv2.LINE_AA)
    fill = int((w - 8) * len(win) / win.maxlen)
    cv2.rectangle(frame, (x0 + 4, y0 + h - 6), (x0 + 4 + fill, y0 + h - 2), COLOR_AQUA_CYAN, -1)


def draw_tremor_graph(frame, win: RollingWindow, w: int = 460, h: int = 130, margin: int = 20):
    """Scrolling index height plot with Ribbon Pink trace for /test/tremor."""
    H = frame.shape[0]
    x0 = margin
    y0 = H - margin - h
    _, _, iy = win.arrays()

    fi = iy[np.isfinite(iy)]
    if fi.size:
        mid, half = float(np.median(fi)), max(float(np.ptp(fi)) * 0.6, 0.05)
        lo, hi = mid - half, mid + half
    else:
        lo, hi = -1.0, 1.0

    _draw_series_panel(frame, win, iy, x0, y0, w, h, "MDS-UPDRS 3.15  Index Height (Gravity Axis)",
                       COLOR_HOT_PINK, lo, hi)

    cv2.putText(frame, f"{len(win)}/{win.maxlen} ({win.span:4.1f}s)", (x0 + w - 140, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_AQUA_CYAN, 1, cv2.LINE_AA)
    fill = int((w - 8) * len(win) / win.maxlen)
    cv2.rectangle(frame, (x0 + 4, y0 + h - 6), (x0 + 4 + fill, y0 + h - 2), COLOR_HOT_PINK, -1)


def draw_window_graph(frame, win: RollingWindow, w: int = 440, h: int = 110, margin: int = 16):
    """Stacked plots for legacy desktop mode."""
    H = frame.shape[0]
    x0 = margin
    y_ap = H - margin - 2 * h - 8
    y_iy = H - margin - h
    _, ap, iy = win.arrays()

    fa = ap[np.isfinite(ap)]
    ap_hi = max(1.5, float(fa.max()) * 1.1) if fa.size else 1.5
    _draw_series_panel(frame, win, ap, x0, y_ap, w, h, "Stage 1  Aperture", COLOR_AQUA_CYAN,
                       0.0, ap_hi, ref=RATIO_OPEN)

    fi = iy[np.isfinite(iy)]
    if fi.size:
        mid, half = float(np.median(fi)), max(float(np.ptp(fi)) * 0.6, 0.05)
        lo, hi = mid - half, mid + half
    else:
        lo, hi = -1.0, 1.0
    _draw_series_panel(frame, win, iy, x0, y_iy, w, h, "Stage 2  Index Height", COLOR_HOT_PINK, lo, hi)


# ------------------------------------------------------------------ Decoupled HUD Banners & Sleek Scorecards
def draw_tapping_hud(frame: np.ndarray, session: TappingSession, t: float):
    """Dedicated HUD prompt banner, 10s countdown, and sleek completion banner for /test/tapping."""
    H, W = frame.shape[:2]
    bx, by, bw, bh = 30, 14, W - 60, 96
    
    # Frosted Ocean Navy background with Hot Pink border
    roi = frame[by:by + bh, bx:bx + bw]
    navy_box = np.full_like(roi, 0)
    navy_box[:] = COLOR_OCEAN_NAVY
    cv2.addWeighted(roi, 0.25, navy_box, 0.75, 0, roi)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_HOT_PINK, 2, cv2.LINE_AA)

    rem = session.time_remaining(t)
    frac = session.progress_fraction(t)

    if session.state == SessionState.IDLE:
        cv2.putText(frame, "STAGE 1: BRADYKINESIA FINGER TAPPING (MDS-UPDRS 3.4)", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.76, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Click [Start Test] or press [SPACE] to begin 10-second trial",
                    (bx + 20, by + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)

    elif session.state == SessionState.COUNTDOWN:
        cv2.putText(frame, "STAGE 1 — FINGER TAPPING ACTIVE", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_HOT_PINK, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TIME: {rem:4.1f}s", (bx + bw - 190, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, '"Tap thumb and index finger as WIDE and FAST as possible."',
                    (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
        bar_w = int((bw - 40) * (1.0 - frac))
        cv2.rectangle(frame, (bx + 20, by + 82), (bx + 20 + bar_w, by + 88), COLOR_AQUA_CYAN, -1)

    elif session.state == SessionState.COMPLETED:
        cv2.putText(frame, "STAGE 1 TRIAL COMPLETE — AI CLINICAL ANALYSIS ACTIVE", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Review comprehensive Gemini AI Clinical Summary below canvas ↓",
                    (bx + 20, by + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.66, COLOR_HOT_PINK, 2, cv2.LINE_AA)
        draw_tapping_scorecard(frame, session)


def draw_tapping_scorecard(frame: np.ndarray, session: TappingSession):
    """Draw sleek Ocean-Blue & Pink banner on stream without obscuring camera view."""
    if not session.results:
        return
    s1 = session.results
    H, W = frame.shape[:2]
    
    # Sleek bottom banner card
    cw, ch = min(720, W - 60), 125
    cx0, cy0 = (W - cw) // 2, H - ch - 30

    card_roi = frame[cy0:cy0 + ch, cx0:cx0 + cw]
    navy_box = np.full_like(card_roi, 0)
    navy_box[:] = COLOR_OCEAN_NAVY
    cv2.addWeighted(card_roi, 0.20, navy_box, 0.80, 0, card_roi)
    cv2.rectangle(frame, (cx0, cy0), (cx0 + cw, cy0 + ch), COLOR_HOT_PINK, 2, cv2.LINE_AA)

    # 4-Tier color for Score badge
    tier_col = get_tier_color(s1.score)
    tier_label = get_tier_label(s1.score, is_tremor=False)

    # Title & Badge
    cv2.putText(frame, "MDS-UPDRS 3.4 FINGER TAPPING — TRIAL LOCKED", (cx0 + 20, cy0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
    
    badge_str = f"Score {s1.score}/4 ({s1.label})"
    cv2.putText(frame, badge_str, (cx0 + cw - 240, cy0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, tier_col, 2, cv2.LINE_AA)

    cv2.line(frame, (cx0 + 20, cy0 + 44), (cx0 + cw - 20, cy0 + 44), (90, 46, 15), 1, cv2.LINE_AA)

    # Metrics Row
    metrics_str = f"Speed: {s1.rate:.1f} taps/s  |  Fatigue Decr: {s1.decrement * 100:.1f}%  |  Count: {s1.n_taps}  |  CV: {s1.cv:.2f}"
    cv2.putText(frame, metrics_str, (cx0 + 20, cy0 + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)

    # AI Directive Subtitle
    cv2.putText(frame, "AI Clinical Summary generated on web card below. Use buttons to Retry or Return.",
                (cx0 + 20, cy0 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_HOT_PINK, 1, cv2.LINE_AA)


def draw_tremor_hud(frame: np.ndarray, session: TremorSession, t: float):
    """Dedicated HUD prompt banner, 10s countdown, and sleek completion banner for /test/tremor."""
    H, W = frame.shape[:2]
    bx, by, bw, bh = 30, 14, W - 60, 96
    
    # Frosted Ocean Navy background with Hot Pink border
    roi = frame[by:by + bh, bx:bx + bw]
    navy_box = np.full_like(roi, 0)
    navy_box[:] = COLOR_OCEAN_NAVY
    cv2.addWeighted(roi, 0.25, navy_box, 0.75, 0, roi)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_HOT_PINK, 2, cv2.LINE_AA)

    rem = session.time_remaining(t)
    frac = session.progress_fraction(t)

    if session.state == SessionState.IDLE:
        cv2.putText(frame, "STAGE 2: POSTURAL TREMOR SPECTRAL TEST (MDS-UPDRS 3.15)", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.74, COLOR_HOT_PINK, 2, cv2.LINE_AA)
        cv2.putText(frame, "Click [Start Test] or press [SPACE] to begin 10-second trial",
                    (bx + 20, by + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)

    elif session.state == SessionState.COUNTDOWN:
        cv2.putText(frame, "STAGE 2 — POSTURAL TREMOR ACTIVE", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TIME: {rem:4.1f}s", (bx + bw - 190, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_HOT_PINK, 2, cv2.LINE_AA)
        cv2.putText(frame, '"Hold hand flat and completely still in the air."',
                    (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
        bar_w = int((bw - 40) * (1.0 - frac))
        cv2.rectangle(frame, (bx + 20, by + 82), (bx + 20 + bar_w, by + 88), COLOR_HOT_PINK, -1)

    elif session.state == SessionState.COMPLETED:
        cv2.putText(frame, "STAGE 2 TRIAL COMPLETE — AI CLINICAL ANALYSIS ACTIVE", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_HOT_PINK, 2, cv2.LINE_AA)
        cv2.putText(frame, "Review comprehensive Gemini AI Clinical Summary below canvas ↓",
                    (bx + 20, by + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.66, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        draw_tremor_scorecard(frame, session)


def draw_tremor_scorecard(frame: np.ndarray, session: TremorSession):
    """Draw sleek Ocean-Blue & Pink banner on stream without obscuring camera view."""
    if not session.results:
        return
    s2 = session.results
    H, W = frame.shape[:2]
    
    # Sleek bottom banner card
    cw, ch = min(720, W - 60), 125
    cx0, cy0 = (W - cw) // 2, H - ch - 30

    card_roi = frame[cy0:cy0 + ch, cx0:cx0 + cw]
    navy_box = np.full_like(card_roi, 0)
    navy_box[:] = COLOR_OCEAN_NAVY
    cv2.addWeighted(card_roi, 0.20, navy_box, 0.80, 0, card_roi)
    cv2.rectangle(frame, (cx0, cy0), (cx0 + cw, cy0 + ch), COLOR_AQUA_CYAN, 2, cv2.LINE_AA)

    # 4-Tier color for Score badge (Fix Score X/4 denominator!)
    tier_col = get_tier_color(s2.score)
    tier_label = get_tier_label(s2.score, is_tremor=True)

    # Title & Badge
    cv2.putText(frame, "MDS-UPDRS 3.15 POSTURAL TREMOR — TRIAL LOCKED", (cx0 + 20, cy0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_HOT_PINK, 2, cv2.LINE_AA)
    
    badge_str = f"Score {s2.score}/4 ({s2.label})"
    cv2.putText(frame, badge_str, (cx0 + cw - 240, cy0 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, tier_col, 2, cv2.LINE_AA)

    cv2.line(frame, (cx0 + 20, cy0 + 44), (cx0 + cw - 20, cy0 + 44), (90, 46, 15), 1, cv2.LINE_AA)

    # Metrics Row
    metrics_str = f"Peak: {s2.peak_hz:.1f} Hz  |  4-6 Hz Band: {s2.band_ratio * 100:.1f}%  |  RMS: {s2.rms:.4f}  |  {tier_label}"
    cv2.putText(frame, metrics_str, (cx0 + 20, cy0 + 74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 1, cv2.LINE_AA)

    # AI Directive Subtitle
    cv2.putText(frame, "AI Clinical Summary generated on web card below. Use buttons to Retry or Return.",
                (cx0 + 20, cy0 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_AQUA_CYAN, 1, cv2.LINE_AA)


# ------------------------------------------------------------------ Legacy Assessment Prompts & Scorecard
def draw_assessment_hud(frame: np.ndarray, session: AssessmentSession, t: float):
    """Render top instruction banner, 10s countdown bar, and the Stage 3 clinical scorecard."""
    H, W = frame.shape[:2]

    bx, by, bw, bh = 40, 14, W - 80, 96
    roi = frame[by:by + bh, bx:bx + bw]
    navy_box = np.full_like(roi, 0)
    navy_box[:] = COLOR_OCEAN_NAVY
    cv2.addWeighted(roi, 0.25, navy_box, 0.75, 0, roi)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_HOT_PINK, 2, cv2.LINE_AA)

    rem = session.time_remaining(t)
    frac = session.progress_fraction(t)

    if session.state == SessionState.IDLE:
        cv2.putText(frame, "CLINICAL MOTOR ASSESSMENT READY", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press [SPACE] to start Stage 1 (10-second Finger Tapping Trial)",
                    (bx + 20, by + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (240, 240, 240), 2, cv2.LINE_AA)

    elif session.state == SessionState.STAGE1_COUNTDOWN:
        cv2.putText(frame, "STAGE 1 — FINGER TAPPING", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TIME: {rem:4.1f}s", (bx + bw - 180, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_HOT_PINK, 2, cv2.LINE_AA)
        cv2.putText(frame, '"Tap thumb and index finger as WIDE and FAST as possible."',
                    (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
        bar_w = int((bw - 40) * (1.0 - frac))
        cv2.rectangle(frame, (bx + 20, by + 82), (bx + 20 + bar_w, by + 88), COLOR_AQUA_CYAN, -1)

    elif session.state == SessionState.STAGE1_LOCKED:
        s1 = session.stage1_res
        cv2.putText(frame, "STAGE 1 LOCKED IN!  [PREPARING FOR STAGE 2...]", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 184, 166), 2, cv2.LINE_AA)
        stats_str = f"Speed: {s1.rate:.1f} taps/s  |  Fatigue Decr: {s1.decrement * 100:.1f}%  |  Score: {s1.score}/4" if s1 else ""
        cv2.putText(frame, stats_str, (bx + 20, by + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Stage 2 starts in {rem:3.1f}s...", (bx + bw - 240, by + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_HOT_PINK, 1, cv2.LINE_AA)

    elif session.state == SessionState.STAGE2_COUNTDOWN:
        cv2.putText(frame, "STAGE 2 — POSTURAL TREMOR", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_HOT_PINK, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TIME: {rem:4.1f}s", (bx + bw - 180, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, '"Hold hand flat and completely still in the air."',
                    (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (255, 255, 255), 2, cv2.LINE_AA)
        bar_w = int((bw - 40) * (1.0 - frac))
        cv2.rectangle(frame, (bx + 20, by + 82), (bx + 20 + bar_w, by + 88), COLOR_HOT_PINK, -1)

    elif session.state == SessionState.STAGE3_SCORECARD:
        cv2.putText(frame, "STAGE 3 — CLINICAL ASSESSMENT COMPLETE", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.76, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press [SPACE] or [R] to Retest   |   [Q] to Quit", (bx + 20, by + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (220, 220, 220), 2, cv2.LINE_AA)
        draw_clinical_scorecard(frame, session)


def draw_clinical_scorecard(frame: np.ndarray, session: AssessmentSession):
    """Draw Stage 3 Final Clinical Scorecard modal with strict 4-tier color coding."""
    H, W = frame.shape[:2]
    cw, ch = 860, 480
    cx0, cy0 = (W - cw) // 2, (H - ch) // 2 + 30

    card_roi = frame[cy0:cy0 + ch, cx0:cx0 + cw]
    navy_box = np.full_like(card_roi, 0)
    navy_box[:] = COLOR_OCEAN_NAVY
    cv2.addWeighted(card_roi, 0.16, navy_box, 0.84, 0, card_roi)
    cv2.rectangle(frame, (cx0, cy0), (cx0 + cw, cy0 + ch), COLOR_HOT_PINK, 2, cv2.LINE_AA)

    cv2.putText(frame, "MDS-UPDRS MOTOR ASSESSMENT SCORECARD", (cx0 + 36, cy0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx0 + 36, cy0 + 60), (cx0 + cw - 36, cy0 + 60), (90, 46, 15), 1, cv2.LINE_AA)

    badge_y0 = cy0 + 78
    badge_h = 44
    max_score = 0
    if session.stage1_res:
        max_score = max(max_score, session.stage1_res.score)
    if session.stage2_res:
        max_score = max(max_score, session.stage2_res.score)

    badge_bg = get_tier_color(max_score)
    badge_text = f" {get_tier_label(max_score)} (MAX SCORE {max_score}/4) "

    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    badge_w = tw + 40
    badge_x0 = (W - badge_w) // 2
    cv2.rectangle(frame, (badge_x0, badge_y0), (badge_x0 + badge_w, badge_y0 + badge_h), badge_bg, -1)
    cv2.rectangle(frame, (badge_x0, badge_y0), (badge_x0 + badge_w, badge_y0 + badge_h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, badge_text, (badge_x0 + 20, badge_y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    col_w = (cw - 100) // 2
    c1_x0 = cx0 + 36
    c2_x0 = cx0 + 36 + col_w + 28
    box_y0 = cy0 + 144
    box_h = 240

    # Left Column: Stage 1 Finger Tapping
    cv2.rectangle(frame, (c1_x0, box_y0), (c1_x0 + col_w, box_y0 + box_h), (30, 16, 6), -1)
    cv2.rectangle(frame, (c1_x0, box_y0), (c1_x0 + col_w, box_y0 + box_h), COLOR_AQUA_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "STAGE 1: FINGER TAPPING", (c1_x0 + 16, box_y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_AQUA_CYAN, 2, cv2.LINE_AA)

    s1 = session.stage1_res
    if s1:
        cv2.putText(frame, f"Final Tap Speed:     {s1.rate:5.1f} taps/s", (c1_x0 + 16, box_y0 + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Fatigue Decrement:   {s1.decrement * 100:5.1f} %", (c1_x0 + 16, box_y0 + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Fatigue Slope:      {s1.slope:+.4f}/s", (c1_x0 + 16, box_y0 + 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hesitation CV:       {s1.cv:5.2f}", (c1_x0 + 16, box_y0 + 176),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
        s1_col = get_tier_color(s1.score)
        cv2.putText(frame, f"MDS-UPDRS:  Score {s1.score}/4 ({s1.label})", (c1_x0 + 16, box_y0 + 214),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, s1_col, 2, cv2.LINE_AA)

    # Right Column: Stage 2 Postural Tremor
    cv2.rectangle(frame, (c2_x0, box_y0), (c2_x0 + col_w, box_y0 + box_h), (30, 16, 6), -1)
    cv2.rectangle(frame, (c2_x0, box_y0), (c2_x0 + col_w, box_y0 + box_h), COLOR_HOT_PINK, 1, cv2.LINE_AA)
    cv2.putText(frame, "STAGE 2: POSTURAL TREMOR", (c2_x0 + 16, box_y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_HOT_PINK, 2, cv2.LINE_AA)

    s2 = session.stage2_res
    if s2:
        cv2.putText(frame, f"Dominant Peak:       {s2.peak_hz:5.1f} Hz", (c2_x0 + 16, box_y0 + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        band_str = f"4-6 Hz Band Power:   {s2.band_ratio * 100:5.1f} %"
        cv2.putText(frame, band_str, (c2_x0 + 16, box_y0 + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tremor Magnitude:    {s2.rms:5.3f} RMS", (c2_x0 + 16, box_y0 + 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Spatial Excursion:   {s2.excursion:5.3f}",
                    (c2_x0 + 16, box_y0 + 176), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
        s2_col = get_tier_color(s2.score)
        # Note: Fixed Score X/4 denominator bug here (was Score X/3)
        cv2.putText(frame, f"MDS-UPDRS:  Score {s2.score}/4 ({s2.label})", (c2_x0 + 16, box_y0 + 214),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, s2_col, 2, cv2.LINE_AA)

    cv2.putText(frame, "Press [SPACE] or [R] to Retest Assessment  |  [Q] to Exit",
                (cx0 + 180, cy0 + ch - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)



# ------------------------------------------------------------------ Desktop OpenCV Window Mode
def run_desktop_hud():
    cv2.namedWindow("Tremor HUD", cv2.WINDOW_AUTOSIZE)
    win = RollingWindow()
    session = AssessmentSession()

    print("\n[hud] Starting Desktop OpenCV HUD preview (Press [SPACE] to start trial, [Q] to quit)...")
    with HandTracker(num_hands=1) as tracker:
        while True:
            frame, hands, t = tracker.read()
            if frame is None:
                time.sleep(0.03)
                continue

            ratio = float("nan")
            iy = float("nan")
            ix = float("nan")

            if hands:
                hf = hands[0]
                ratio = palm_ratio(hf)
                ix, iy = index_tip_coords(hf)
                win.push(t, ratio, iy)
                tracker.draw(frame, hands, show_pinch=False)
                draw_pinch_cue(frame, hf, ratio)
            else:
                win.push(t, float("nan"), float("nan"))

            session.update(t, ratio, iy, ix)

            if session.state != SessionState.STAGE3_SCORECARD:
                draw_window_graph(frame, win)

            draw_assessment_hud(frame, session, t)

            cv2.putText(frame, f"FPS {tracker.fps:5.1f}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

            cv2.imshow("Tremor HUD", frame)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                session.start(t)
            elif key == ord("r"):
                session.start(t)
            elif key == ord("c"):
                win.clear()
                session.reset()

            # Check if user closed the window
            if cv2.getWindowProperty("Tremor HUD", cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


def main():
    import sys
    import os

    if "--desktop" in sys.argv or "-d" in sys.argv:
        run_desktop_hud()
    else:
        # By default, launch the Multi-Page Web Application
        import webbrowser
        import threading
        import socket

        port = int(os.getenv("PORT", 5001))
        url = f"http://127.0.0.1:{port}"

        print("\n" + "=" * 65)
        print("  TremorTrack MDS-UPDRS Clinical Motor Assessment Web App")
        print("=" * 65)
        print(f"  Launching Multi-Page Web Application on {url}")
        print("  (Tip: To run the legacy desktop OpenCV window instead, use: python hud.py --desktop)")
        print("=" * 65 + "\n")

        # Check if server is already running on this port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        is_port_in_use = (sock.connect_ex(('127.0.0.1', port)) == 0)
        sock.close()

        if is_port_in_use:
            print(f"  --> Web application server is already active on {url}!")
            print(f"  --> Opening browser tab directly...\n")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            return

        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

        import app
        app.app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

