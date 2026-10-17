"""DSP engine: finger tapping (bradykinesia) + postural tremor spectrum.

All thresholds live at the top so you can calibrate them at your demo table.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch, detrend
from scipy.stats import linregress

FS = 30.0  # uniform resample rate (Hz)

# ---- Finger tapping thresholds (calibrate!) ----
TAP_RATE_NORMAL = 3.5     # taps/s
TAP_RATE_SLIGHT = 2.5
TAP_RATE_MILD = 1.5
DECREMENT_SLIGHT = 0.20   # fractional amplitude loss over the trial
DECREMENT_MILD = 0.35

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
    rng = np.ptp(a)
    peaks, _ = find_peaks(a, distance=max(1, int(0.12 * fs)), prominence=max(0.15 * rng, 0.05))
    out["peaks"] = peaks
    out["n_taps"] = len(peaks)
    dur = tu[-1] - tu[0]
    out["rate"] = len(peaks) / dur if dur > 0 else 0.0
    if len(peaks) >= 3:
        pt, pa = tu[peaks], a[peaks]
        reg = linregress(pt, pa)
        out["slope"], out["intercept"] = reg.slope, reg.intercept
        mean_amp = np.mean(pa)
        out["decrement"] = max(0.0, -reg.slope * dur / mean_amp) if mean_amp > 0 else 0.0
        intervals = np.diff(pt)
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
    total = np.trapz(p[full], f[full]) if full.any() else 0.0
    out["band_ratio"] = float(np.trapz(p[band], f[band]) / total) if total > 0 else 0.0
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
