"""DSP engine: finger tapping (bradykinesia) + postural tremor spectrum.

All thresholds live at the top so you can calibrate them at your demo table.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from scipy.stats import linregress
from scipy.integrate import trapezoid

FS = 30.0  # uniform resample rate (Hz)

# ---- Finger tapping thresholds (calibrate!) ----
TAP_RATE_NORMAL = 2.6     # taps/s
TAP_RATE_SLIGHT = 1.8
TAP_RATE_MILD = 1.2
DECREMENT_SLIGHT = 0.28   # fractional amplitude loss over the trial
DECREMENT_MILD = 0.45

# ---- Tremor thresholds (calibrate!) ----
TREMOR_BAND = (4.0, 6.0)          # parkinsonian rest/postural band
TREMOR_BAND_RATIO_MIN = 0.30      # fraction of 1-12 Hz power inside band
TREMOR_RMS = (0.01, 0.03, 0.08)   # hand-size-normalized RMS cutoffs for 1/2/3

SCORE_LABELS = {0: "Normal", 1: "Slight", 2: "Mild", 3: "Moderate", 4: "Severe"}


def resample_uniform(t, x, fs=FS):
    """Resample an irregularly sampled signal (NaNs dropped) onto a uniform grid."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    if len(t) < 4:
        return np.array([]), np.array([])
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, x)


def analyze_finger_tapping(t, a_norm, fs=FS):
    """Thumb-index normalized aperture -> tap rate, amplitude decrement, score."""
    tu, a = resample_uniform(t, a_norm, fs)
    out = {"t": tu, "a": a, "peaks": np.array([], int), "rate": 0.0,
           "decrement": 0.0, "slope": 0.0, "intercept": 0.0, "score": 4,
           "n_taps": 0, "cv": 0.0}
    if len(a) < fs * 2:
        return out

    # Smooth the signal with a 3-point moving average before peak detection
    a_smooth = np.convolve(a, np.ones(3) / 3.0, mode="same")
    a_smooth[0] = (a[0] + a[1]) / 2.0
    a_smooth[-1] = (a[-2] + a[-1]) / 2.0
    a = a_smooth
    out["a"] = a

    rng = np.ptp(a)
    peaks, _ = find_peaks(a, distance=max(1, int(0.12 * fs)), prominence=max(0.15 * rng, 0.05))
    out["peaks"] = peaks
    out["n_taps"] = len(peaks)

    # Active tapping rate via (len(peaks) - 1) / (tu[peaks[-1]] - tu[peaks[0]])
    if len(peaks) >= 2:
        dur_active = tu[peaks[-1]] - tu[peaks[0]]
        out["rate"] = float((len(peaks) - 1) / dur_active) if dur_active > 0 else 0.0
    else:
        out["rate"] = 0.0

    # Subtract baseline trough levels to isolate swing amplitude
    amps = []
    for i, p in enumerate(peaks):
        left = peaks[i - 1] if i > 0 else max(0, p - int(0.5 * fs))
        right = peaks[i + 1] if i < len(peaks) - 1 else min(len(a), p + int(0.5 * fs))
        trough = min(np.min(a[left:p]), np.min(a[p:right])) if (p > left and right > p) else np.min(a)
        amps.append(max(0.0, a[p] - trough))
    amps = np.array(amps)

    # Linear regression on swing amplitude, dropping first and last peak outliers (peaks[1:-1])
    if len(peaks) >= 4:
        pt = tu[peaks[1:-1]]
        pa = amps[1:-1]
        dur_reg = pt[-1] - pt[0]
        reg = linregress(pt, pa)
        out["slope"], out["intercept"] = float(reg.slope), float(reg.intercept)
        mean_amp = float(np.mean(pa))
        out["decrement"] = max(0.0, -reg.slope * dur_reg / mean_amp) if mean_amp > 0 and dur_reg > 0 else 0.0
    elif len(peaks) >= 3:
        pt = tu[peaks]
        pa = amps
        dur_reg = pt[-1] - pt[0]
        reg = linregress(pt, pa)
        out["slope"], out["intercept"] = float(reg.slope), float(reg.intercept)
        mean_amp = float(np.mean(pa))
        out["decrement"] = max(0.0, -reg.slope * dur_reg / mean_amp) if mean_amp > 0 and dur_reg > 0 else 0.0

    if len(peaks) >= 2:
        intervals = np.diff(tu[peaks])
        if len(intervals) > 0 and np.mean(intervals) > 0:
            out["cv"] = float(np.std(intervals) / np.mean(intervals))

    r, d = out["rate"], out["decrement"]
    if out["n_taps"] < 3:
        score = 4
    elif r >= TAP_RATE_NORMAL and d < DECREMENT_SLIGHT:
        score = 0
    elif r >= TAP_RATE_SLIGHT and d < DECREMENT_MILD:
        score = 1
    elif r >= TAP_RATE_MILD:
        score = 2
    else:
        score = 3
    out["score"] = score
    return out


def analyze_tremor_spectrum(t, y, x_coord=None, fs=FS):
    """Multi-axis hand movement -> Welch PSD, peak freq, band ratio, RMS, excursion, score."""
    tu, y_u = resample_uniform(t, y, fs)
    out = {
        "t": tu, "x": y_u, "f": np.array([]), "psd": np.array([]),
        "peak_hz": 0.0, "band_ratio": 0.0, "broad_ratio": 0.0,
        "rms": 0.0, "excursion": 0.0, "velocity_rms": 0.0,
        "score": 0, "label": "Normal", "is_elevated_risk": False,
        "risk_badge": "NORMAL", "tremor_type": "Physiological (Normal)"
    }
    if len(y_u) < fs * 3:
        return out

    # Detrend and 1-12 Hz 4th-order bandpass filter for Y
    b, a = butter(4, [1.0 / (fs / 2), 12.0 / (fs / 2)], btype="band")
    yf = filtfilt(b, a, detrend(y_u))

    has_x = False
    if x_coord is not None and len(x_coord) == len(t):
        _, x_u = resample_uniform(t, x_coord, fs)
        if len(x_u) == len(y_u):
            xf = filtfilt(b, a, detrend(x_u))
            has_x = True

    if has_x:
        rms_x = float(np.sqrt(np.mean(xf ** 2)))
        rms_y = float(np.sqrt(np.mean(yf ** 2)))
        rms = float(np.hypot(rms_x, rms_y))

        exc_x = float(np.ptp(xf))
        exc_y = float(np.ptp(yf))
        excursion = float(np.hypot(exc_x, exc_y))

        dt = 1.0 / fs
        vel_x = np.diff(xf) / dt
        vel_y = np.diff(yf) / dt
        vel_rms = float(np.sqrt(np.mean(vel_x ** 2 + vel_y ** 2))) if len(vel_x) else 0.0

        sig = xf if rms_x > rms_y else yf
    else:
        sig = yf
        rms = float(np.sqrt(np.mean(yf ** 2)))
        excursion = float(np.ptp(yf))
        dt = 1.0 / fs
        vel = np.diff(yf) / dt
        vel_rms = float(np.sqrt(np.mean(vel ** 2))) if len(vel) else 0.0

    out["x"] = sig
    out["rms"] = rms
    out["excursion"] = excursion
    out["velocity_rms"] = vel_rms

    # Power Spectral Density via Welch
    f, p = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    out["f"], out["psd"] = f, p

    full = (f >= 1.0) & (f <= 12.0)
    park_band = (f >= 4.0) & (f <= 6.0)
    broad_band = (f >= 3.5) & (f <= 9.0)

    total = float(trapezoid(p[full], f[full])) if full.any() else 0.0
    out["band_ratio"] = float(trapezoid(p[park_band], f[park_band]) / total) if total > 0 else 0.0
    out["broad_ratio"] = float(trapezoid(p[broad_band], f[broad_band]) / total) if total > 0 else 0.0

    if full.any():
        out["peak_hz"] = float(f[full][np.argmax(p[full])])

    ratio = out["band_ratio"]
    peak_hz = out["peak_hz"]

    # 4-6 Hz Parkinsonian tremor condition: true 4-6 Hz dominant peak with concentrated power (>= 25%) and measurable RMS
    is_parkinsonian = (ratio >= 0.25 and 4.0 <= peak_hz <= 6.5 and rms >= 0.010)

    # Determine strict 4-tier score (0 to 4)
    if rms >= 0.090 or vel_rms >= 1.20 or excursion >= 0.25:
        score = 4
    elif rms >= 0.050 or vel_rms >= 0.60 or excursion >= 0.14:
        score = 3
    elif is_parkinsonian or rms >= 0.025 or vel_rms >= 0.35 or excursion >= 0.08:
        score = 2
    elif rms >= 0.018 or vel_rms >= 0.20 or excursion >= 0.04:
        score = 1
    else:
        score = 0

    label = SCORE_LABELS.get(score, "Normal")
    is_elevated = bool(score >= 3 or (score >= 2 and is_parkinsonian))

    if score == 0:
        risk_badge = "NORMAL (CALM WATERS)"
        tremor_type = "Physiological (Normal)"
    elif score == 1:
        risk_badge = "SLIGHT / MILD POSTURAL EMERGENCE"
        tremor_type = "Mild Postural Emergence"
    elif score == 2:
        if is_parkinsonian:
            risk_badge = "MILD POSTURAL EMERGENCE (4-6 HZ BAND)"
            tremor_type = "Mild Parkinsonian Postural Tremor (4-6 Hz)"
        else:
            risk_badge = "MILD POSTURAL MOVEMENT DETECTED"
            tremor_type = "Mild Kinematic Excursion"
    else:
        if is_parkinsonian:
            risk_badge = "ELEVATED RISK / 4-6 HZ PARKINSONIAN BAND"
            tremor_type = "Parkinsonian Postural Tremor (4-6 Hz)"
        else:
            risk_badge = "ELEVATED RISK / HIGH-AMPLITUDE MOVEMENT"
            tremor_type = "High-Amplitude Kinetic Shaking / Tremor"

    out["score"] = score
    out["label"] = label
    out["is_elevated_risk"] = is_elevated
    out["risk_badge"] = risk_badge
    out["tremor_type"] = tremor_type
    return out
