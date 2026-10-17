# pyrefly: ignore[missing-import]
"""HUD: three 300-sample sliding windows (10 s at 30 fps) + visual cues.

Queues (parallel, same length, same index = same frame):
  * timestamps  : seconds since tracker start
  * aperture    : normalized finger aperture (3D thumb-index / palm size) -> Stage 1 tapping
  * index_y     : index fingertip vertical height / hand size in px       -> Stage 2 tremor

Every frame: compute both signals, push them (oldest sample drops automatically),
then draw a pinch cue on the hand and a live scrolling graph for each signal.

Run:  python hud.py   (q / Esc to quit, c to clear the window)
"""
from __future__ import annotations

from collections import deque

# pyrefly: ignore[missing-import]
import cv2
import numpy as np

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


# ------------------------------------------------------------------ drawing
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


def draw_window_graph(frame, win: RollingWindow, w: int = 460, h: int = 120, margin: int = 16):
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
    cv2.putText(frame, f"{len(win)}/{win.maxlen}  ({win.span:4.1f}s)", (x0 + w - 150, y_ap + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    fill = int((w - 8) * len(win) / win.maxlen)
    bar_color = (80, 220, 120) if win.full else (80, 180, 255)
    cv2.rectangle(frame, (x0 + 4, y_iy + h - 8), (x0 + 4 + fill, y_iy + h - 4), bar_color, -1)


def main():
    win = RollingWindow()
    with HandTracker(num_hands=1) as tracker:
        while True:
            frame, hands, t = tracker.read()
            if frame is None:
                print("[hud] camera frame grab failed")
                break
            if hands:
                hf = hands[0]
                ratio = palm_ratio(hf)
                win.push(t, ratio, index_tip_height(hf))
                tracker.draw(frame, hands, show_pinch=False)
                draw_pinch_cue(frame, hf, ratio)
            else:
                win.push(t, float("nan"), float("nan"))   # keep time continuous; dsp drops NaNs
            draw_window_graph(frame, win)
            cv2.putText(frame, f"FPS {tracker.fps:5.1f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.imshow("Tremor HUD", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                win.clear()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
