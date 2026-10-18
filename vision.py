"""Vision tracker: live webcam -> MediaPipe HandLandmarker (21 3D joints) -> per-frame features.

Produces exactly what dsp.py consumes:
  * aperture_norm : thumb-tip <-> index-tip distance / hand size   -> analyze_finger_tapping
  * tremor_y      : vertical (gravity-axis) hand position / hand size -> analyze_tremor_spectrum

Run standalone for a live preview:  python vision.py   (q / Esc to quit)
"""
from __future__ import annotations

import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field

# pyrefly: ignore[missing-import]
import cv2
import numpy as np
# pyrefly: ignore[missing-import]
import mediapipe as mp
# pyrefly: ignore[missing-import]
from mediapipe.tasks import python as mp_python
# pyrefly: ignore[missing-import]
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")

# ---- MediaPipe hand joint indices (21 total) ----
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
)
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass
# Container about one hand in 1 frame
class HandFrame:
    """One detected hand in one frame."""
    t: float                    # seconds since tracker start (monotonic)
    handedness: str             # "Left" / "Right" (as seen by the model)
    score: float                # handedness confidence
    image: np.ndarray           # (21, 3) normalized image coords: x,y in [0,1], z relative depth
    world: np.ndarray           # (21, 3) metric 3D joints in meters, origin at hand center
    pixels: np.ndarray          # (21, 2) int pixel coords for drawing
    hand_size: float = 0.0      # wrist -> middle MCP, meters (world)
    aperture_norm: float = np.nan  # thumb-index tip distance / hand_size (3D, scale-invariant)
    tremor_y: float = np.nan    # image-space vertical position of middle MCP / hand size in px
    extras: dict = field(default_factory=dict)

# Automatically download hand_landmarker in to models/
def ensure_model(path: str = MODEL_PATH) -> str:
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"[vision] downloading hand landmarker model -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Live webcam hand tracker built on the MediaPipe Tasks HandLandmarker.

    Uses VIDEO running mode (synchronous, temporally smoothed) so every returned
    frame is paired with its own landmarks and a precise timestamp for the DSP.
    """

    # Open webcam
    def __init__(self, camera_index: int = 0, num_hands: int = 1, width: int = 1280,
                 height: int = 720, fps: int = 30, mirror: bool = True,
                 min_detection_confidence: float = 0.5, min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, model_path: str = MODEL_PATH):
        self.mirror = mirror
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {camera_index}. On macOS, grant camera "
                               "access to your terminal/IDE in System Settings > Privacy.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self._last_ts_ms = -1
        self._fps = 0.0
        self._last_frame_t = None
        self._baseline_hand_size: float | None = None

    # ------------------------------------------------------------------ core
    # grab one frame
    def read(self):
        """Grab one webcam frame and run the landmarker.

        Returns (frame_bgr, hands: list[HandFrame], t_seconds) or (None, [], t) if the camera fails.
        """
        ok, frame = self.cap.read()
        t = time.perf_counter() - self._t0
        if not ok:
            return None, [], t
        if self.mirror:
            frame = cv2.flip(frame, 1)

        # VIDEO mode requires strictly increasing integer ms timestamps.
        ts_ms = max(int(t * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        if self._last_frame_t is not None:
            dt = t - self._last_frame_t
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        self._last_frame_t = t

        return frame, self._parse(result, frame.shape, t), t

    def _parse(self, result, shape, t) -> list[HandFrame]:
        h, w = shape[:2]
        hands = []
        for i, lms in enumerate(result.hand_landmarks or []):
            image = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
            world_lms = result.hand_world_landmarks[i] if result.hand_world_landmarks else lms
            world = np.array([[p.x, p.y, p.z] for p in world_lms], dtype=float)
            pixels = np.column_stack([image[:, 0] * w, image[:, 1] * h]).astype(int)

            cat = result.handedness[i][0] if result.handedness else None
            hf = HandFrame(t=t, handedness=cat.category_name if cat else "?",
                           score=float(cat.score) if cat else 0.0,
                           image=image, world=world, pixels=pixels)
            self._features(hf, w, h)
            hands.append(hf)
        return hands

    # Calculate two key numbers
    def _features(self, hf: HandFrame, w: int, h: int) -> None:
        # 3D metric hand size (wrist -> middle MCP); robust to camera distance and rotation.
        hf.hand_size = float(np.linalg.norm(hf.world[MIDDLE_MCP] - hf.world[WRIST]))
        if hf.hand_size > 1e-6:
            pinch = np.linalg.norm(hf.world[THUMB_TIP] - hf.world[INDEX_TIP])
            hf.aperture_norm = float(pinch / hf.hand_size)

        # World landmarks are hand-centered, so whole-hand translation (tremor) must come from
        # image space. Normalize by hand size in pixels so it's distance-invariant.
        px = hf.image[:, :2] * np.array([w, h])

        # Invariant 2D palm scale:
        # When hand is held flat towards camera, wrist->middle MCP foreshortens heavily,
        # but knuckle width (index MCP -> pinky MCP) is perpendicular to camera and stays stable.
        palm_w = float(np.linalg.norm(px[INDEX_MCP] - px[PINKY_MCP]))
        palm_h = float(np.linalg.norm(px[MIDDLE_MCP] - px[WRIST]))
        size_px = max(palm_w, palm_h, 25.0)

        # High-inertia baseline stabilization:
        # Prevents the divisor from continuously drifting or oscillating as the hand changes posture
        if self._baseline_hand_size is None:
            self._baseline_hand_size = size_px
        else:
            self._baseline_hand_size = 0.98 * self._baseline_hand_size + 0.02 * size_px

        baseline_size = self._baseline_hand_size
        if baseline_size > 1e-6:
            hf.tremor_y = float(px[MIDDLE_MCP, 1] / baseline_size)
        hf.extras["hand_size_px"] = float(baseline_size)

    # ------------------------------------------------------------- drawing
    @staticmethod
    # Draw the skeleton
    def draw(frame, hands: list[HandFrame], show_pinch: bool = True):
        for hf in hands:
            pts = hf.pixels
            # depth-shaded bones: closer joints (more negative z) are brighter
            z = hf.image[:, 2]
            zn = (z - z.min()) / (np.ptp(z) + 1e-6)
            for a, b in HAND_CONNECTIONS:
                shade = 1.0 - 0.5 * (zn[a] + zn[b]) / 2
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]),
                         (int(255 * shade), int(200 * shade), 60), 2, cv2.LINE_AA)
            for j, (x, y) in enumerate(pts):
                r = 6 if j in FINGERTIPS else 4
                color = (80, 220, 255) if j in FINGERTIPS else (255, 255, 255)
                cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), r, (40, 40, 40), 1, cv2.LINE_AA)
            if show_pinch:
                cv2.line(frame, tuple(pts[THUMB_TIP]), tuple(pts[INDEX_TIP]),
                         (120, 255, 120), 2, cv2.LINE_AA)
            x0, y0 = pts[WRIST]
            cv2.putText(frame, f"{hf.handedness} {hf.score:.2f}  ap={hf.aperture_norm:.2f}",
                        (x0 - 60, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return frame

    @property
    def fps(self) -> float:
        return self._fps

    # ----------------------------------------------------------- lifecycle
    def close(self):
        if getattr(self, "landmarker", None):
            self.landmarker.close()
            self.landmarker = None
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main():
    with HandTracker(num_hands=2) as tracker:
        while True:
            frame, hands, t = tracker.read()
            if frame is None:
                print("[vision] camera frame grab failed")
                break
            tracker.draw(frame, hands)
            cv2.putText(frame, f"FPS {tracker.fps:5.1f}   hands {len(hands)}   joints {21 * len(hands)}",
                        (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2, cv2.LINE_AA)
            if hands:
                wx, wy, wz = hands[0].world[INDEX_TIP]
                cv2.putText(frame, f"index tip 3D (m): {wx:+.3f} {wy:+.3f} {wz:+.3f}",
                            (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Vision Tracker - HandLandmarker", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
