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


def analyze_tremor_spectrum(t, y, fs=FS):
    """Vertical (gravity-axis) hand position -> Welch PSD, peak freq, band ratio, score."""
    tu, x = resample_uniform(t, y, fs)
    out = {"t": tu, "x": x, "f": np.array([]), "psd": np.array([]), "peak_hz": 0.0,
           "band_ratio": 0.0, "rms": 0.0, "score": 0}
    if len(x) < fs * 3:
        return out
    x = detrend(x)
    b, a = butter(4, [1.0 / (fs / 2), 12.0 / (fs / 2)], btype="band")
    xf = filtfilt(b, a, x)
    out["x"] = xf
    f, p = welch(xf, fs=fs, nperseg=min(256, len(xf)))
    out["f"], out["psd"] = f, p
    full = (f >= 1) & (f <= 12)
    band = (f >= TREMOR_BAND[0]) & (f <= TREMOR_BAND[1])
    total = trapezoid(p[full], f[full]) if full.any() else 0.0
    out["band_ratio"] = float(trapezoid(p[band], f[band]) / total) if total > 0 else 0.0
    if full.any():
        out["peak_hz"] = float(f[full][np.argmax(p[full])])
    out["rms"] = float(np.sqrt(np.mean(xf ** 2)))

    rms, ratio = out["rms"], out["band_ratio"]
    if ratio < TREMOR_BAND_RATIO_MIN or rms < TREMOR_RMS[0]:
        score = 0
    elif rms < TREMOR_RMS[1]:
        score = 1
    elif rms < TREMOR_RMS[2]:
        score = 2
    else:
        score = 3
    out["score"] = score
    return out
