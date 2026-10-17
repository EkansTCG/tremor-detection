# pyrefly: ignore[missing-import]
"""HUD: Three-Stage Clinical Motor Assessment + 300-Sample Sliding Windows.

Assessment Flow:
  1. Stage 1 — Finger Tapping (10s countdown):
     Prompts: "Tap thumb and index finger as WIDE and FAST as possible."
     Locks in: final tap speed, fatigue slope, and hesitation count.
  2. Stage 2 — Postural Tremor (10s countdown):
     Prompts: "Hold hand flat and completely still in the air."
     Watches bottom graph for 4–6 Hz red zone spike & locks in dominant frequency.
  3. Stage 3 — Final Clinical Scorecard:
     Displays locked-in metrics with a Green Normal or Red Elevated Risk badge.

Controls:
  * [SPACE] : Start assessment / Retest from Scorecard
  * [R]     : Reset / restart assessment
  * [C]     : Clear rolling window queues
  * [Q]/ESC : Quit
"""
from __future__ import annotations

from collections import deque

# pyrefly: ignore[missing-import]
import cv2
import numpy as np

from session import AssessmentSession, SessionState
from vision import (HandFrame, HandTracker, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP,
                    THUMB_TIP, WRIST)

WINDOW_FPS = 30.0
WINDOW_SIZE = 300                 # 300 samples / 30 fps = 10 s of history
RATIO_OPEN = 1.0                  # aperture drawn as fully "open" (green) at/above this


def dist3(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 3D points."""
    return float(np.linalg.norm(a - b))


def palm_ratio(hf: HandFrame) -> float:
    """Thumb-index pinch distance / palm size, both measured in 3D world space.

    Palm size averages palm length (wrist -> middle MCP) and palm width
    (index MCP -> pinky MCP), so it stays stable when the hand rotates or tilts.
    """
    w = hf.world
    palm = 0.5 * (dist3(w[WRIST], w[MIDDLE_MCP]) + dist3(w[INDEX_MCP], w[PINKY_MCP]))
    if palm < 1e-6:
        return float("nan")
    return dist3(w[THUMB_TIP], w[INDEX_TIP]) / palm


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
        """Snapshot as numpy arrays (timestamps, aperture, index_y).

        dsp.analyze_finger_tapping(t, aperture) / dsp.analyze_tremor_spectrum(t, index_y)
        """
        return (np.fromiter(self.timestamps, float), np.fromiter(self.aperture, float),
                np.fromiter(self.index_y, float))


# ------------------------------------------------------------------ drawing helpers
def _ratio_color(r: float):
    """BGR color from red (closed) through yellow to green (open)."""
    if not np.isfinite(r):
        return (128, 128, 128)
    k = float(np.clip(r / RATIO_OPEN, 0.0, 1.0))
    return (60, int(80 + 175 * k), int(255 * (1 - k) + 60 * k))


def draw_pinch_cue(frame, hf: HandFrame, ratio: float):
    """Colored thumb-index line plus a ring at its midpoint whose size follows palm_ratio."""
    p1, p2 = hf.pixels[THUMB_TIP], hf.pixels[INDEX_TIP]
    color = _ratio_color(ratio)
    cv2.line(frame, tuple(p1), tuple(p2), color, 4, cv2.LINE_AA)
    mid = tuple(((p1 + p2) // 2).astype(int))
    radius = int(8 + 30 * np.clip(ratio / RATIO_OPEN, 0, 1.5)) if np.isfinite(ratio) else 8
    cv2.circle(frame, mid, radius, color, 2, cv2.LINE_AA)
    cv2.circle(frame, mid, 4, color, -1, cv2.LINE_AA)
    cv2.putText(frame, f"{ratio:.2f}", (mid[0] + radius + 6, mid[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _draw_series_panel(frame, win: RollingWindow, values: np.ndarray, x0: int, y0: int,
                       w: int, h: int, label: str, color, lo: float, hi: float,
                       ref: float | None = None):
    """One translucent scrolling plot of a queue, scaled to [lo, hi]."""
    roi = frame[y0:y0 + h, x0:x0 + w]
    cv2.addWeighted(roi, 0.35, np.full_like(roi, 20), 0.65, 0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (90, 90, 90), 1, cv2.LINE_AA)

    gy0, gy1 = y0 + 26, y0 + h - 18
    span = max(hi - lo, 1e-6)
    to_y = lambda v: gy1 - (v - lo) / span * (gy1 - gy0)

    if ref is not None:
        ry = int(to_y(ref))
        cv2.line(frame, (x0 + 4, ry), (x0 + w - 4, ry), (70, 70, 70), 1, cv2.LINE_AA)

    if len(values) > 1:
        # x is slot index across the full window, so the trace fills left -> right
        xs = x0 + 4 + (np.arange(len(values)) + (win.maxlen - len(values))) * (w - 8) / (win.maxlen - 1)
        ok = np.isfinite(values)
        ys = to_y(np.where(ok, values, lo))
        pts = np.column_stack([xs, np.clip(ys, gy0, gy1)]).astype(np.int32)
        # break the line at gaps (no hand) instead of diving to zero
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

    cur = f"{values[-1]:+.3f}" if len(values) and np.isfinite(values[-1]) else "--"
    cv2.putText(frame, f"{label}  {cur}", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)


def draw_window_graph(frame, win: RollingWindow, w: int = 440, h: int = 110, margin: int = 16):
    """Stacked plots (bottom-left): Stage 1 aperture over Stage 2 index height, plus fill bar."""
    H = frame.shape[0]
    x0 = margin
    y_ap = H - margin - 2 * h - 8
    y_iy = H - margin - h
    _, ap, iy = win.arrays()

    fa = ap[np.isfinite(ap)]
    ap_hi = max(1.5, float(fa.max()) * 1.1) if fa.size else 1.5
    _draw_series_panel(frame, win, ap, x0, y_ap, w, h, "Stage 1  aperture", (255, 210, 80),
                       0.0, ap_hi, ref=RATIO_OPEN)

    # tremor is a small wiggle on a large offset: auto-zoom around the window's own range
    fi = iy[np.isfinite(iy)]
    if fi.size:
        mid, half = float(np.median(fi)), max(float(np.ptp(fi)) * 0.6, 0.05)
        lo, hi = mid - half, mid + half
    else:
        lo, hi = -1.0, 1.0
    _draw_series_panel(frame, win, iy, x0, y_iy, w, h, "Stage 2  index tip height",
                       (120, 160, 255), lo, hi)

    # shared buffer status for all three queues
    cv2.putText(frame, f"{len(win)}/{win.maxlen}  ({win.span:4.1f}s)", (x0 + w - 140, y_ap + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    fill = int((w - 8) * len(win) / win.maxlen)
    bar_color = (80, 220, 120) if win.full else (80, 180, 255)
    cv2.rectangle(frame, (x0 + 4, y_iy + h - 8), (x0 + 4 + fill, y_iy + h - 4), bar_color, -1)


# ------------------------------------------------------------------ Assessment Prompts & Scorecard
def draw_assessment_hud(frame: np.ndarray, session: AssessmentSession, t: float):
    """Render top instruction banner, 10s countdown bar, and the Stage 3 clinical scorecard."""
    H, W = frame.shape[:2]

    # --- TOP PROMPT BANNER ---
    bx, by, bw, bh = 40, 14, W - 80, 96
    roi = frame[by:by + bh, bx:bx + bw]
    cv2.addWeighted(roi, 0.25, np.full_like(roi, 20), 0.75, 0, roi)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), 1, cv2.LINE_AA)

    rem = session.time_remaining(t)
    frac = session.progress_fraction(t)

    if session.state == SessionState.IDLE:
        cv2.putText(frame, "CLINICAL MOTOR ASSESSMENT READY", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press [SPACE] to start Stage 1 (10-second Finger Tapping Trial)",
                    (bx + 20, by + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)

    elif session.state == SessionState.STAGE1_COUNTDOWN:
        # Title + countdown
        cv2.putText(frame, "STAGE 1 — FINGER TAPPING", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"TIME: {rem:4.1f}s", (bx + bw - 180, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 200), 2, cv2.LINE_AA)
        # Prompts the user: "Tap thumb and index finger as WIDE and FAST as possible."
        cv2.putText(frame, '"Tap thumb and index finger as WIDE and FAST as possible."',
                    (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
        # Progress countdown bar
        bar_w = int((bw - 40) * (1.0 - frac))
        cv2.rectangle(frame, (bx + 20, by + 82), (bx + 20 + bar_w, by + 88), (0, 229, 255), -1)

    elif session.state == SessionState.STAGE1_LOCKED:
        s1 = session.stage1_res
        cv2.putText(frame, "STAGE 1 LOCKED IN!  [PREPARING FOR STAGE 2...]", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 120), 2, cv2.LINE_AA)
        stats_str = f"Speed: {s1.rate:.1f} taps/s  |  Fatigue Decrement: {s1.decrement * 100:.1f}%  |  Taps: {s1.n_taps}  |  Score: {s1.label}" if s1 else ""
        cv2.putText(frame, stats_str, (bx + 20, by + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Stage 2 starts in {rem:3.1f}s...", (bx + bw - 240, by + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 1, cv2.LINE_AA)

    elif session.state == SessionState.STAGE2_COUNTDOWN:
        # Title + countdown
        cv2.putText(frame, "STAGE 2 — POSTURAL TREMOR", (bx + 20, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 80, 80), 2, cv2.LINE_AA)
        cv2.putText(frame, f"TIME: {rem:4.1f}s", (bx + bw - 180, by + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 200), 2, cv2.LINE_AA)
        # Prompts the user: "Hold hand flat and completely still in the air."
        cv2.putText(frame, '"Hold hand flat and completely still in the air."',
                    (bx + 20, by + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
        # Progress countdown bar
        bar_w = int((bw - 40) * (1.0 - frac))
        cv2.rectangle(frame, (bx + 20, by + 82), (bx + 20 + bar_w, by + 88), (255, 80, 80), -1)

    elif session.state == SessionState.STAGE3_SCORECARD:
        cv2.putText(frame, "STAGE 3 — CLINICAL ASSESSMENT COMPLETE", (bx + 20, by + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press [SPACE] or [R] to Retest   |   [Q] to Quit", (bx + 20, by + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        # Draw full scorecard modal
        draw_clinical_scorecard(frame, session)


def draw_clinical_scorecard(frame: np.ndarray, session: AssessmentSession):
    """Draw Stage 3 Final Clinical Scorecard modal with Normal or Elevated Risk badge."""
    H, W = frame.shape[:2]
    cw, ch = 860, 480
    cx0, cy0 = (W - cw) // 2, (H - ch) // 2 + 30

    # Modal backdrop
    card_roi = frame[cy0:cy0 + ch, cx0:cx0 + cw]
    cv2.addWeighted(card_roi, 0.12, np.full_like(card_roi, 16), 0.88, 0, card_roi)
    cv2.rectangle(frame, (cx0, cy0), (cx0 + cw, cy0 + ch), (70, 70, 70), 2, cv2.LINE_AA)

    # Header
    cv2.putText(frame, "MDS-UPDRS MOTOR ASSESSMENT SCORECARD", (cx0 + 36, cy0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx0 + 36, cy0 + 60), (cx0 + cw - 36, cy0 + 60), (90, 90, 90), 1, cv2.LINE_AA)

    # Clinical Badge: Green [NORMAL] or Red [ELEVATED RISK / ASYMMETRY DETECTED]
    badge_y0 = cy0 + 78
    badge_h = 44
    if session.is_elevated_risk:
        badge_text = " ELEVATED RISK / ASYMMETRY DETECTED "
        badge_bg = (30, 30, 220)       # Solid Red
        badge_border = (60, 60, 255)
    else:
        badge_text = " NORMAL "
        badge_bg = (40, 160, 40)       # Solid Green
        badge_border = (80, 240, 80)

    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    badge_w = tw + 40
    badge_x0 = (W - badge_w) // 2
    cv2.rectangle(frame, (badge_x0, badge_y0), (badge_x0 + badge_w, badge_y0 + badge_h), badge_bg, -1)
    cv2.rectangle(frame, (badge_x0, badge_y0), (badge_x0 + badge_w, badge_y0 + badge_h), badge_border, 2, cv2.LINE_AA)
    cv2.putText(frame, badge_text, (badge_x0 + 20, badge_y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Two side-by-side metric columns
    col_w = (cw - 100) // 2
    c1_x0 = cx0 + 36
    c2_x0 = cx0 + 36 + col_w + 28
    box_y0 = cy0 + 144
    box_h = 240

    # Left Column: Stage 1 Finger Tapping
    cv2.rectangle(frame, (c1_x0, box_y0), (c1_x0 + col_w, box_y0 + box_h), (35, 35, 35), -1)
    cv2.rectangle(frame, (c1_x0, box_y0), (c1_x0 + col_w, box_y0 + box_h), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "STAGE 1: FINGER TAPPING", (c1_x0 + 16, box_y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2, cv2.LINE_AA)

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
        s1_color = (80, 220, 120) if s1.score <= 1 else (60, 60, 255)
        cv2.putText(frame, f"UPDRS Rating:  Score {s1.score}/4 ({s1.label})", (c1_x0 + 16, box_y0 + 214),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, s1_color, 2, cv2.LINE_AA)

    # Right Column: Stage 2 Postural Tremor
    cv2.rectangle(frame, (c2_x0, box_y0), (c2_x0 + col_w, box_y0 + box_h), (35, 35, 35), -1)
    cv2.rectangle(frame, (c2_x0, box_y0), (c2_x0 + col_w, box_y0 + box_h), (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(frame, "STAGE 2: POSTURAL TREMOR", (c2_x0 + 16, box_y0 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 120, 120), 2, cv2.LINE_AA)

    s2 = session.stage2_res
    if s2:
        cv2.putText(frame, f"Dominant Peak:       {s2.peak_hz:5.1f} Hz", (c2_x0 + 16, box_y0 + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1, cv2.LINE_AA)
        band_str = f"4-6 Hz Band Power:   {s2.band_ratio * 100:5.1f} %"
        b_color = (60, 60, 255) if s2.band_ratio >= 0.30 else (230, 230, 230)
        cv2.putText(frame, band_str, (c2_x0 + 16, box_y0 + 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, b_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tremor Magnitude:    {s2.rms:5.3f} RMS", (c2_x0 + 16, box_y0 + 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tremor Band Active:  {'YES (Warning)' if s2.band_ratio >= 0.30 else 'NO'}",
                    (c2_x0 + 16, box_y0 + 176), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
        s2_color = (80, 220, 120) if s2.score == 0 else (60, 60, 255)
        cv2.putText(frame, f"UPDRS Rating:  Score {s2.score}/3 ({s2.label})", (c2_x0 + 16, box_y0 + 214),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, s2_color, 2, cv2.LINE_AA)

    # Footer
    cv2.putText(frame, "Press [SPACE] or [R] to Retest Assessment  |  [Q] to Exit",
                (cx0 + 180, cy0 + ch - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ Main Loop
def main():
    win = RollingWindow()
    session = AssessmentSession()

    with HandTracker(num_hands=1) as tracker:
        while True:
            frame, hands, t = tracker.read()
            if frame is None:
                print("[hud] camera frame grab failed")
                break

            ratio = float("nan")
            iy = float("nan")

            if hands:
                hf = hands[0]
                ratio = palm_ratio(hf)
                iy = index_tip_height(hf)
                win.push(t, ratio, iy)
                tracker.draw(frame, hands, show_pinch=False)
                draw_pinch_cue(frame, hf, ratio)
            else:
                win.push(t, float("nan"), float("nan"))   # preserve time progression

            # Update Clinical Assessment Session State
            session.update(t, ratio, iy)

            # Draw bottom-left live sliding window graphs (aperture + tremor height)
            if session.state != SessionState.STAGE3_SCORECARD:
                draw_window_graph(frame, win)

            # Draw top prompts, countdown bar, and clinical scorecard
            draw_assessment_hud(frame, session, t)

            # FPS banner
            cv2.putText(frame, f"FPS {tracker.fps:5.1f}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

            cv2.imshow("Tremor HUD", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                # Start / restart trial
                session.start(t)
            elif key == ord("r"):
                # Reset trial
                session.start(t)
            elif key == ord("c"):
                win.clear()
                session.reset()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
