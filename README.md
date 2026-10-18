# TremorTrack: Clinical Motor Assessment Web Application (MDS-UPDRS)

A multi-page clinical motor assessment web application featuring real-time MediaPipe hand tracking, digital signal processing (DSP) kinematics, persistent database storage, and decoupled test protocols.

---

## 🌟 Key Architecture & Features

### 1. Authentication & Persistent Database Layer (`auth_db.py`)
- **Dual Backend**: Supports Firebase Authentication & Firestore if credentials (`FIREBASE_PROJECT_ID`, `FIREBASE_API_KEY`, or `FIREBASE_SERVICE_ACCOUNT_KEY`) are present in `.env`.
- **Automatic SQLite Fallback**: Automatically creates and persists to a local SQLite database (`patients.db`) when Firebase is not configured, requiring zero setup.
- **1-Click Demo Patient Login**: High-visibility button on the login screen for judges to bypass credentials during live demonstrations, pre-loading **Eleanor Vance (PT-88219)** with longitudinal clinical history.

### 2. Clinical Home Dashboard (`/home`)
- **Patient Profile Banner**: Displays patient demographic metadata (MRN, age, gender, clinical indication, overall risk badge).
- **Decoupled Protocol Navigation Cards**:
  - **Card A (MDS-UPDRS 3.4)**: Bradykinesia Finger Tapping Test &rarr; `/test/tapping`
  - **Card B (MDS-UPDRS 3.15)**: Postural Tremor Spectral Test &rarr; `/test/tremor`
- **Longitudinal Trend Analytics**: Interactive **Chart.js** visualizations tracking tap speed (Hz), fatigue decrement (%), dominant tremor peak (Hz), and 4–6 Hz power ratio over time.
- **Patient Assessment History Table**: Full audit log of past trials with UPDRS ratings (0–4), kinematic metrics, and clinical risk flags.

### 3. Decoupled Test Pages (`/test/tapping` & `/test/tremor`)
- **Stage 1: Finger Tapping (`/test/tapping`)**:
  - Streams MediaPipe + OpenCV webcam HUD running **only** the 10-second Stage 1 trial.
  - Computes scale-invariant thumb-index aperture (`palm_ratio`), real-time scrolling waveform, tap rate, fatigue slope, and rhythm variability via `dsp.analyze_finger_tapping`.
  - When the 10s timer finishes, displays the Stage 1 Scorecard modal and auto-saves the result to the user's database record.
  - Provides `[Retry Test]` and `[Back to Home]` buttons.
- **Stage 2: Postural Tremor (`/test/tremor`)**:
  - Streams MediaPipe + OpenCV webcam HUD running **only** the 10-second Stage 2 trial.
  - Computes vertical index fingertip displacement (`index_tip_height`), Welch Power Spectral Density (PSD), 4–6 Hz parkinsonian tremor band ratio, and RMS magnitude via `dsp.analyze_tremor_spectrum`.
  - When the 10s timer finishes, displays the Stage 2 Scorecard modal with clinical risk badge and auto-saves the result to the user's database record.
  - Provides `[Retry Test]` and `[Back to Home]` buttons.
- **MJPEG Streaming**: Clean OpenCV HUD frames served via `/video_feed/<test_type>`.
- **Intelligent Camera Fallback**: Automatically falls back to a synthetic kinematic stream if the physical webcam is busy or unavailable.

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```
Open your browser to:
```
http://127.0.0.1:5001
```

### 3. Live Judge Testing
1. On the login screen, click **"⚡ 1-Click Demo Patient Login"**.
2. Explore Eleanor Vance's longitudinal trend charts and assessment history on `/home`.
3. Click **"Launch Tapping Test"** to run Stage 1. Press `[SPACE]` to start the 10s countdown.
4. When finished, review the scorecard and click **"Back to Home"** to see your new trial recorded in the history table!
5. Click **"Launch Tremor Test"** to run Stage 2. Press `[SPACE]` to start the 10s countdown.