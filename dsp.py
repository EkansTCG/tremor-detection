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

    # If X coordinates are provided, evaluate X axis as well for omni-directional shaking
    if x_coord is not None and len(x_coord) == len(t):
        _, x_u = resample_uniform(t, x_coord, fs)
        if len(x_u) == len(y_u):
            xf = filtfilt(b, a, detrend(x_u))
            # Pick primary oscillation axis with highest kinetic excursion
            rms_x = float(np.sqrt(np.mean(xf ** 2)))
            rms_y = float(np.sqrt(np.mean(yf ** 2)))
            sig = xf if rms_x > rms_y else yf
        else:
            sig = yf
    else:
        sig = yf

    out["x"] = sig

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

    # Kinematic excursion & RMS
    rms = float(np.sqrt(np.mean(sig ** 2)))
    excursion = float(np.ptp(sig))
    dt = 1.0 / fs
    vel = np.diff(sig) / dt
    vel_rms = float(np.sqrt(np.mean(vel ** 2))) if len(vel) else 0.0

    out["rms"] = rms
    out["excursion"] = excursion
    out["velocity_rms"] = vel_rms

    ratio = out["band_ratio"]
    peak_hz = out["peak_hz"]

    # Accurate MDS-UPDRS 3.15 Physical Scoring:
    # 0: Normal (<0.012 RMS, minimal excursion)
    # 1: Slight (0.012 - 0.028 RMS, or emerging 4-6 Hz band concentration)
    # 2: Mild (0.028 - 0.065 RMS, or distinct 4-6 Hz tremor)
    # 3: Moderate (0.065 - 0.120 RMS, or prominent kinetic shaking)
    # 4: Severe (>=0.120 RMS or >=0.30 excursion, violent/high-amplitude shaking)
    if rms < 0.012 and excursion < 0.045:
        score = 0
    elif rms < 0.028 and excursion < 0.10:
        score = 1
    elif rms < 0.065 and excursion < 0.20:
        score = 2
    elif rms < 0.120 and excursion < 0.35:
        score = 3
    else:
        score = 4

    # If prominent 4-6 Hz Parkinsonian oscillation is present with sufficient amplitude
    is_parkinsonian = (ratio >= 0.25 and 4.0 <= peak_hz <= 6.5 and rms >= 0.010)
    if is_parkinsonian and score < 1:
        score = 1

    # Elevated risk determination:
    # Triggered if score >= 2 OR Parkinsonian band concentrated OR active kinetic shaking detected
    is_elevated = bool(
        score >= 2 or
        is_parkinsonian or
        rms >= 0.028 or
        excursion >= 0.09
    )

    if not is_elevated:
        risk_badge = "NORMAL"
        tremor_type = "Physiological (Normal)"
        label = "Normal"
        score = 0
    elif is_parkinsonian:
        risk_badge = "ELEVATED RISK / 4-6 HZ PARKINSONIAN BAND"
        tremor_type = "Parkinsonian Postural Tremor (4-6 Hz)"
        label = SCORE_LABELS.get(score, "Mild")
    else:
        risk_badge = "ELEVATED RISK / ACTIVE SHAKING DETECTED"
        tremor_type = "High-Amplitude Kinetic Shaking / Tremor"
        score = max(score, 2)
        label = SCORE_LABELS.get(score, "Moderate")

    out["score"] = score
    out["label"] = label
    out["is_elevated_risk"] = is_elevated
    out["risk_badge"] = risk_badge
    out["tremor_type"] = tremor_type
    return out
