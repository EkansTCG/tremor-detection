"""Camera & Video Stream Manager for Clinical Motor Assessment.

Handles webcam capture, MediaPipe HandLandmarker inference, HUD rendering,
and MJPEG frame generator for /video_feed/tapping and /video_feed/tremor endpoints.
Includes simulated camera fallback if physical webcam is busy or unavailable.
"""
from __future__ import annotations

import time
import math
import threading
from typing import Generator, Optional, Dict, Any
# pyrefly: ignore[missing-import]
import cv2
import numpy as np

from vision import HandTracker, HandFrame, WRIST, INDEX_TIP, THUMB_TIP, MIDDLE_MCP, PINKY_MCP
from hud import (RollingWindow, palm_ratio, index_tip_height, index_tip_coords, draw_pinch_cue,
                 draw_tremor_cue, draw_tapping_graph, draw_tremor_graph,
                 draw_tapping_hud, draw_tremor_hud)
from session import TappingSession, TremorSession, SessionState
import auth_db


class SyntheticCamera:
    """Fallback generator producing realistic synthetic hands if physical webcam is locked."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self._t0 = time.perf_counter()
        self.fps = 30.0

    def read(self):
        t = time.perf_counter() - self._t0
        # Dark clinic backdrop
        frame = np.full((self.height, self.width, 3), 18, dtype=np.uint8)

        # Subtle grid background
        for x in range(0, self.width, 80):
            cv2.line(frame, (x, 0), (x, self.height), (26, 26, 26), 1)
        for y in range(0, self.height, 80):
            cv2.line(frame, (0, y), (self.width, y), (26, 26, 26), 1)

        # Draw a synthetic hand simulation
        cx, cy = self.width // 2, self.height // 2 + 30
        osc = math.sin(t * 2 * math.pi * 2.2)  # 2.2 Hz tapping oscillation
        tremor_wiggle = math.sin(t * 2 * math.pi * 4.8) * 6.0  # 4.8 Hz tremor wiggle

        # Base wrist and palm joints
        wrist = np.array([cx, cy + 160])
        middle_mcp = np.array([cx, cy - 20])
        pinky_mcp = np.array([cx + 90, cy - 10])
        index_mcp = np.array([cx - 80, cy - 15])
        
        # Aperture variation for thumb-index
        pinch_dist = 40 + int(35 * (osc + 1.0))
        thumb_tip = np.array([cx - 50 - pinch_dist // 2, cy - 70])
        index_tip = np.array([cx - 50 + pinch_dist // 2, cy - 110 + int(tremor_wiggle)])

        # Draw synthetic hand bones
        bones = [(wrist, middle_mcp), (wrist, index_mcp), (wrist, pinky_mcp),
                 (index_mcp, index_tip), (wrist, thumb_tip)]
        for p1, p2 in bones:
            cv2.line(frame, tuple(p1), tuple(p2), (180, 200, 70), 3, cv2.LINE_AA)

        for pt in [wrist, middle_mcp, pinky_mcp, index_mcp]:
            cv2.circle(frame, tuple(pt), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(thumb_tip), 8, (80, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(index_tip), 8, (80, 220, 255), -1, cv2.LINE_AA)

        cv2.putText(frame, "[SIMULATED CAMERA STREAM ACTIVE]", (30, self.height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)

        # Synthetic HandFrame representation
        pixels = np.zeros((21, 2), dtype=int)
        pixels[WRIST] = wrist
        pixels[MIDDLE_MCP] = middle_mcp
        pixels[INDEX_MCP] = index_mcp
        pixels[PINKY_MCP] = pinky_mcp
        pixels[THUMB_TIP] = thumb_tip
        pixels[INDEX_TIP] = index_tip

        hf = HandFrame(
            t=t,
            handedness="Right",
            score=0.98,
            image=np.zeros((21, 3)),
            world=np.zeros((21, 3)),
            pixels=pixels,
            hand_size=0.18,
            extras={"hand_size_px": 180.0}
        )

        return frame, [hf], t

    def close(self):
        pass


class StreamManager:
    """Singleton coordinator for video capture and decoupled test sessions."""

    _instance: Optional[StreamManager] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> StreamManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = StreamManager()
            return cls._instance

    def __init__(self):
        self.lock = threading.Lock()
        self.tracker: Optional[Any] = None
        self.is_synthetic = False

        # Independent decoupled test sessions
        self.tapping_session = TappingSession()
        self.tremor_session = TremorSession()

        # Rolling window histories
        self.win_tapping = RollingWindow()
        self.win_tremor = RollingWindow()

        # Active user tracking for auto-saving results
        self.current_user_id: Optional[str] = "demo-patient-001"

        self._init_tracker()

    def _init_tracker(self):
        try:
            self.tracker = HandTracker(camera_index=0, num_hands=1)
            self.is_synthetic = False
            print("[StreamManager] Initialized physical HandTracker webcam")
        except Exception as e:
            print(f"[StreamManager] Hardware camera error ({e}); using SyntheticCamera")
            self.tracker = SyntheticCamera()
            self.is_synthetic = True

    def set_user(self, user_id: str):
        with self.lock:
            self.current_user_id = user_id

    def start_test(self, test_type: str):
        with self.lock:
            if test_type == "tapping":
                self.tapping_session.start()
            elif test_type == "tremor":
                self.tremor_session.start()

    def reset_test(self, test_type: str):
        with self.lock:
            if test_type == "tapping":
                self.tapping_session.reset()
                self.win_tapping.clear()
            elif test_type == "tremor":
                self.tremor_session.reset()
                self.win_tremor.clear()

    def _compute_live_tapping_telemetry(self) -> Dict[str, Any]:
        """Calculates instantaneous tap rate, aperture, running tap count, and recent trace."""
        # Use active trial buffer if countdown is running, else rolling window
        if self.tapping_session.state == SessionState.COUNTDOWN and len(self.tapping_session.t_buf) > 4:
            t_data = list(self.tapping_session.t_buf)
            ap_data = list(self.tapping_session.ap_buf)
        else:
            t_data = list(self.win_tapping.timestamps)
            ap_data = list(self.win_tapping.aperture)

        t_arr = np.asarray(t_data, float)
        ap_arr = np.asarray(ap_data, float)
        ok = np.isfinite(ap_arr)
        t_arr, ap_arr = t_arr[ok], ap_arr[ok]

        cur_val = float(ap_arr[-1]) if len(ap_arr) else 0.0

        if len(ap_arr) < 6:
            return {
                "current_aperture": round(cur_val, 2),
                "rate": 0.0,
                "decrement_pct": 0.0,
                "n_taps": 0,
                "cv": 0.0,
                "recent_trace": [round(float(x), 3) for x in ap_arr[-60:]]
            }

        # Running peak detection
        from scipy.signal import find_peaks
        rng = np.ptp(ap_arr)
        peaks, _ = find_peaks(ap_arr, distance=max(1, int(0.12 * 30.0)), prominence=max(0.12 * rng, 0.04))
        n_taps = len(peaks)
        rate = 0.0
        if n_taps >= 2:
            dur = t_arr[peaks[-1]] - t_arr[peaks[0]]
            if dur > 0:
                rate = round(float((n_taps - 1) / dur), 1)

        decr_pct = 0.0
        cv = 0.0
        if len(ap_arr) >= 45:
            try:
                import dsp
                raw = dsp.analyze_finger_tapping(t_arr, ap_arr)
                rate = round(float(raw.get("rate", rate)), 1)
                decr_pct = round(float(raw.get("decrement", 0.0)) * 100, 1)
                n_taps = int(raw.get("n_taps", n_taps))
                cv = round(float(raw.get("cv", 0.0)), 2)
            except Exception:
                pass

        return {
            "current_aperture": round(cur_val, 2),
            "rate": rate,
            "decrement_pct": decr_pct,
            "n_taps": n_taps,
            "cv": cv,
            "recent_trace": [round(float(x), 3) if np.isfinite(x) else 0.0 for x in ap_arr[-60:]]
        }

    def _compute_live_tremor_telemetry(self) -> Dict[str, Any]:
        """Calculates instantaneous tremor height, 4-6 Hz band ratio, RMS, excursion, and recent trace."""
        if self.tremor_session.state == SessionState.COUNTDOWN and len(self.tremor_session.t_buf) > 4:
            t_data = list(self.tremor_session.t_buf)
            iy_data = list(self.tremor_session.iy_buf)
            ix_data = list(self.tremor_session.ix_buf) if len(self.tremor_session.ix_buf) == len(self.tremor_session.t_buf) else None
        else:
            t_data = list(self.win_tremor.timestamps)
            iy_data = list(self.win_tremor.index_y)
            ix_data = None

        t_arr = np.asarray(t_data, float)
        iy_arr = np.asarray(iy_data, float)
        ok = np.isfinite(iy_arr)
        t_arr, iy_arr = t_arr[ok], iy_arr[ok]

        cur_val = float(iy_arr[-1]) if len(iy_arr) else 0.0

        # Stabilized zero-mean tremor trace (removes slow arm posture drift)
        recent_raw = [float(x) for x in iy_arr[-60:] if np.isfinite(x)]
        if len(recent_raw) >= 10:
            median_y = float(np.median(recent_raw))
            detrended = np.array(recent_raw) - median_y
            clean_trace = [round(float(x), 4) for x in detrended]
        else:
            clean_trace = [round(float(x - cur_val), 4) for x in recent_raw] if recent_raw else [0.0]

        if len(iy_arr) < 15:
            return {
                "current_y": round(cur_val, 3),
                "peak_hz": 0.0,
                "band_ratio_pct": 0.0,
                "rms": 0.0,
                "excursion": 0.0,
                "score": 0,
                "is_elevated": False,
                "status_badge": "NORMAL",
                "tremor_type": "Physiological (Normal)",
                "recent_trace": clean_trace
            }

        peak_hz = 0.0
        band_pct = 0.0
        excursion = round(float(np.ptp(iy_arr)), 4) if len(iy_arr) else 0.0
        rms = round(float(np.std(iy_arr)), 4) if len(iy_arr) else 0.0
        is_elevated = False
        score = 0
        status_badge = "NORMAL (Physiological)"
        tremor_type = "Physiological (Normal)"

        if len(iy_arr) >= 45:
            try:
                import dsp
                raw = dsp.analyze_tremor_spectrum(t_arr, iy_arr, x_coord=ix_data)
                peak_hz = round(float(raw.get("peak_hz", 0.0)), 1)
                band_pct = round(float(raw.get("band_ratio", 0.0)) * 100, 1)
                rms = round(float(raw.get("rms", rms)), 4)
                excursion = round(float(raw.get("excursion", excursion)), 4)
                score = int(raw.get("score", 0))
                is_elevated = bool(raw.get("is_elevated_risk", False))
                status_badge = str(raw.get("risk_badge", "NORMAL"))
                tremor_type = str(raw.get("tremor_type", "Physiological (Normal)"))
            except Exception:
                pass
        else:
            # Baseline check on detrended signal so slow arm lowering isn't treated as shaking
            if len(recent_raw) >= 10:
                rms = round(float(np.sqrt(np.mean(detrended ** 2))), 4)
                excursion = round(float(np.ptp(detrended)), 4)
            is_elevated = bool(rms >= 0.025 or excursion >= 0.08)
            if is_elevated:
                status_badge = "ELEVATED (Active Shaking)"
                score = 2

        return {
            "current_y": round(cur_val, 3),
            "peak_hz": peak_hz,
            "band_ratio_pct": band_pct,
            "rms": rms,
            "excursion": excursion,
            "score": score,
            "is_elevated": is_elevated,
            "status_badge": status_badge,
            "tremor_type": tremor_type,
            "recent_trace": clean_trace
        }

    def get_status(self, test_type: str) -> Dict[str, Any]:
        with self.lock:
            if test_type == "tapping":
                sess = self.tapping_session
                res = sess.results.to_dict() if sess.results else None
                live = self._compute_live_tapping_telemetry()
                return {
                    "test_type": "tapping",
                    "state": sess.state,
                    "time_remaining": round(sess.time_remaining(), 1),
                    "progress": round(sess.progress_fraction(), 3),
                    "is_completed": sess.is_completed(),
                    "is_saved": sess.is_saved,
                    "saved_id": sess.saved_id,
                    "results": res,
                    "live_metrics": live
                }
            elif test_type == "tremor":
                sess = self.tremor_session
                res = sess.results.to_dict() if sess.results else None
                live = self._compute_live_tremor_telemetry()
                return {
                    "test_type": "tremor",
                    "state": sess.state,
                    "time_remaining": round(sess.time_remaining(), 1),
                    "progress": round(sess.progress_fraction(), 3),
                    "is_completed": sess.is_completed(),
                    "is_saved": sess.is_saved,
                    "saved_id": sess.saved_id,
                    "results": res,
                    "live_metrics": live
                }
            return {"error": "Invalid test type"}

    def generate_frames(self, test_type: str) -> Generator[bytes, None, None]:
        """Yields MJPEG stream of frames with live HUD for the specified test page."""
        while True:
            with self.lock:
                if self.tracker is None:
                    self._init_tracker()

                frame, hands, t = self.tracker.read()
                if frame is None:
                    time.sleep(0.033)
                    continue

                if test_type == "tapping":
                    ratio = float("nan")
                    if hands:
                        hf = hands[0]
                        ratio = palm_ratio(hf)
                        if not self.is_synthetic:
                            HandTracker.draw(frame, hands, show_pinch=False)
                        draw_pinch_cue(frame, hf, ratio)
                        self.win_tapping.push(t, ratio, float("nan"))
                    else:
                        self.win_tapping.push(t, float("nan"), float("nan"))

                    # Step tapping session state
                    new_res = self.tapping_session.update(t, ratio)

                    # Auto-persist to patient database record upon completion
                    if self.tapping_session.is_completed() and not self.tapping_session.is_saved and self.tapping_session.results:
                        try:
                            saved_id = auth_db.save_assessment(
                                self.current_user_id or "demo-patient-001",
                                self.tapping_session.results.to_dict()
                            )
                            self.tapping_session.is_saved = True
                            self.tapping_session.saved_id = saved_id
                            print(f"[StreamManager] Persisted Stage 1 Tapping -> ID {saved_id}")
                        except Exception as err:
                            print(f"[StreamManager] Database save error: {err}")

                    # Render Stage 1 graphs & HUD
                    if self.tapping_session.state != SessionState.COMPLETED:
                        draw_tapping_graph(frame, self.win_tapping)
                    draw_tapping_hud(frame, self.tapping_session, t)

                elif test_type == "tremor":
                    iy = float("nan")
                    ix = float("nan")
                    if hands:
                        hf = hands[0]
                        ix, iy = index_tip_coords(hf)
                        if not self.is_synthetic:
                            HandTracker.draw(frame, hands, show_pinch=False)
                        draw_tremor_cue(frame, hf, iy)
                        self.win_tremor.push(t, float("nan"), iy)
                    else:
                        self.win_tremor.push(t, float("nan"), float("nan"))

                    # Step tremor session state with omnidirectional coordinates
                    new_res = self.tremor_session.update(t, iy, ix)

                    # Auto-persist to patient database record upon completion
                    if self.tremor_session.is_completed() and not self.tremor_session.is_saved and self.tremor_session.results:
                        try:
                            saved_id = auth_db.save_assessment(
                                self.current_user_id or "demo-patient-001",
                                self.tremor_session.results.to_dict()
                            )
                            self.tremor_session.is_saved = True
                            self.tremor_session.saved_id = saved_id
                            print(f"[StreamManager] Persisted Stage 2 Tremor -> ID {saved_id}")
                        except Exception as err:
                            print(f"[StreamManager] Database save error: {err}")

                    # Render Stage 2 graphs & HUD
                    if self.tremor_session.state != SessionState.COMPLETED:
                        draw_tremor_graph(frame, self.win_tremor)
                    draw_tremor_hud(frame, self.tremor_session, t)

                # FPS tag
                fps_val = getattr(self.tracker, "fps", 30.0)
                cv2.putText(frame, f"FPS {fps_val:4.1f}", (16, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 200), 2, cv2.LINE_AA)

                # Encode frame to JPEG
                ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                frame_bytes = jpeg.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.015)  # Throttle smoothly to ~30-40 fps
