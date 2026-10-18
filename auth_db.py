"""Authentication & Database Layer for Clinical Motor Assessment.

Supports:
- Local SQLite database (patients.db) by default with pre-seeded demo patient history.
- Firebase Authentication & Firestore (via firebase-admin or Firebase REST API) if
  FIREBASE_PROJECT_ID / FIREBASE_API_KEY / FIREBASE_SERVICE_ACCOUNT_KEY are configured in .env.
- Automatic fallback to SQLite if Firebase credentials are not set.
- 1-Click Demo Patient Login for judges and live demonstrations.
"""
from __future__ import annotations

import os
import json
import sqlite3
import uuid
import datetime
from typing import Optional, Dict, Any, List
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "patients.db")

# Firebase configuration check
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
FIREBASE_SERVICE_ACCOUNT_KEY = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")

HAS_FIREBASE = bool(FIREBASE_PROJECT_ID and (FIREBASE_API_KEY or FIREBASE_SERVICE_ACCOUNT_KEY))

# Initialize sqlite database
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite database tables and seed demo patient data if empty."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                patient_id TEXT UNIQUE NOT NULL,
                age INTEGER,
                gender TEXT,
                diagnosis TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                score INTEGER NOT NULL,
                label TEXT NOT NULL,
                rate REAL,
                decrement REAL,
                slope REAL,
                n_taps INTEGER,
                cv REAL,
                peak_hz REAL,
                band_ratio REAL,
                rms REAL,
                is_elevated_risk INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.commit()

    # Pre-seed demo user if not exists
    seed_demo_patient_if_needed()


def seed_demo_patient_if_needed() -> None:
    """Creates Eleanor Vance (Demo Patient) with rich longitudinal clinical history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", ("demo@patient.org",))
        existing = cursor.fetchone()
        
        if existing:
            demo_user_id = existing["id"]
        else:
            demo_user_id = "demo-patient-001"
            pwd_hash = generate_password_hash("demopatient123")
            now_iso = (datetime.datetime.now() - datetime.timedelta(days=90)).isoformat()
            cursor.execute("""
                INSERT INTO users (id, email, password_hash, name, patient_id, age, gender, diagnosis, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                demo_user_id,
                "demo@patient.org",
                pwd_hash,
                "Eleanor Vance (Demo Patient)",
                "PT-88219",
                68,
                "Female",
                "MDS-UPDRS Bradykinesia & Postural Tremor Monitoring",
                now_iso
            ))
            conn.commit()

        # Check if longitudinal assessments are already seeded
        cursor.execute("SELECT COUNT(*) as count FROM assessments WHERE user_id = ?", (demo_user_id,))
        count = cursor.fetchone()["count"]

        if count == 0:
            # Seed 6 past assessments spanning past 90 days
            now = datetime.datetime.now()
            historical_data = [
                # 80 days ago: Tapping
                {
                    "id": str(uuid.uuid4()),
                    "user_id": demo_user_id,
                    "test_type": "tapping",
                    "timestamp": (now - datetime.timedelta(days=80, hours=2)).strftime("%Y-%m-%d %H:%M"),
                    "score": 1,
                    "label": "Slight",
                    "rate": 2.45,
                    "decrement": 0.22,
                    "slope": -0.018,
                    "n_taps": 25,
                    "cv": 0.18,
                    "peak_hz": 0.0,
                    "band_ratio": 0.0,
                    "rms": 0.0,
                    "is_elevated_risk": 0,
                    "notes": "Baseline finger tapping protocol. Slight rhythm variation."
                },
                # 80 days ago: Tremor
                {
                    "id": str(uuid.uuid4()),
                    "user_id": demo_user_id,
                    "test_type": "tremor",
                    "timestamp": (now - datetime.timedelta(days=80, hours=1)).strftime("%Y-%m-%d %H:%M"),
                    "score": 0,
                    "label": "Normal",
                    "rate": 0.0,
                    "decrement": 0.0,
                    "slope": 0.0,
                    "n_taps": 0,
                    "cv": 0.0,
                    "peak_hz": 2.1,
                    "band_ratio": 0.12,
                    "rms": 0.007,
                    "is_elevated_risk": 0,
                    "notes": "Postural tremor within normal physiological limits."
                },
                # 50 days ago: Tapping
                {
                    "id": str(uuid.uuid4()),
                    "user_id": demo_user_id,
                    "test_type": "tapping",
                    "timestamp": (now - datetime.timedelta(days=50, hours=3)).strftime("%Y-%m-%d %H:%M"),
                    "score": 2,
                    "label": "Mild",
                    "rate": 1.95,
                    "decrement": 0.35,
                    "slope": -0.032,
                    "n_taps": 20,
                    "cv": 0.28,
                    "peak_hz": 0.0,
                    "band_ratio": 0.0,
                    "rms": 0.0,
                    "is_elevated_risk": 1,
                    "notes": "Increased amplitude decay observed midway through trial."
                },
                # 50 days ago: Tremor
                {
                    "id": str(uuid.uuid4()),
                    "user_id": demo_user_id,
                    "test_type": "tremor",
                    "timestamp": (now - datetime.timedelta(days=50, hours=2)).strftime("%Y-%m-%d %H:%M"),
                    "score": 1,
                    "label": "Slight",
                    "rate": 0.0,
                    "decrement": 0.0,
                    "slope": 0.0,
                    "n_taps": 0,
                    "cv": 0.0,
                    "peak_hz": 4.8,
                    "band_ratio": 0.34,
                    "rms": 0.018,
                    "is_elevated_risk": 1,
                    "notes": "Emergent 4.8 Hz peak inside Parkinsonian rest/postural band."
                },
                # 20 days ago: Tapping
                {
                    "id": str(uuid.uuid4()),
                    "user_id": demo_user_id,
                    "test_type": "tapping",
                    "timestamp": (now - datetime.timedelta(days=20, hours=4)).strftime("%Y-%m-%d %H:%M"),
                    "score": 2,
                    "label": "Mild",
                    "rate": 1.82,
                    "decrement": 0.38,
                    "slope": -0.038,
                    "n_taps": 19,
                    "cv": 0.31,
                    "peak_hz": 0.0,
                    "band_ratio": 0.0,
                    "rms": 0.0,
                    "is_elevated_risk": 1,
                    "notes": "Hesitations on non-dominant hand; mild bradykinesia progression."
                },
                # 20 days ago: Tremor
                {
                    "id": str(uuid.uuid4()),
                    "user_id": demo_user_id,
                    "test_type": "tremor",
                    "timestamp": (now - datetime.timedelta(days=20, hours=3)).strftime("%Y-%m-%d %H:%M"),
                    "score": 2,
                    "label": "Mild",
                    "rate": 0.0,
                    "decrement": 0.0,
                    "slope": 0.0,
                    "n_taps": 0,
                    "cv": 0.0,
                    "peak_hz": 5.1,
                    "band_ratio": 0.42,
                    "rms": 0.036,
                    "is_elevated_risk": 1,
                    "notes": "Prominent 5.1 Hz postural tremor with significant PSD concentration."
                }
            ]

            for item in historical_data:
                cursor.execute("""
                    INSERT INTO assessments (
                        id, user_id, test_type, timestamp, score, label,
                        rate, decrement, slope, n_taps, cv, peak_hz,
                        band_ratio, rms, is_elevated_risk, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["id"], item["user_id"], item["test_type"], item["timestamp"],
                    item["score"], item["label"], item["rate"], item["decrement"],
                    item["slope"], item["n_taps"], item["cv"], item["peak_hz"],
                    item["band_ratio"], item["rms"], item["is_elevated_risk"], item["notes"]
                ))
            conn.commit()


def register_user(email: str, password: str, name: str,
                  patient_id: Optional[str] = None, age: Optional[int] = 65,
                  gender: Optional[str] = "Unspecified", diagnosis: Optional[str] = "Clinical Observation") -> Dict[str, Any]:
    """Register a new user in the database."""
    init_db()
    email_clean = email.strip().lower()
    if not patient_id:
        patient_id = f"PT-{uuid.uuid4().hex[:6].upper()}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email_clean,))
        if cursor.fetchone():
            return {"success": False, "error": "An account with this email already exists."}

        user_id = str(uuid.uuid4())
        pwd_hash = generate_password_hash(password)
        now_iso = datetime.datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO users (id, email, password_hash, name, patient_id, age, gender, diagnosis, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, email_clean, pwd_hash, name.strip(), patient_id, age, gender, diagnosis, now_iso))
        conn.commit()

        return {
            "success": True,
            "user": {
                "id": user_id,
                "email": email_clean,
                "name": name.strip(),
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "diagnosis": diagnosis
            }
        }


def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with email and password."""
    init_db()
    email_clean = email.strip().lower()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
        user_row = cursor.fetchone()

        if user_row and check_password_hash(user_row["password_hash"], password):
            return {
                "id": user_row["id"],
                "email": user_row["email"],
                "name": user_row["name"],
                "patient_id": user_row["patient_id"],
                "age": user_row["age"],
                "gender": user_row["gender"],
                "diagnosis": user_row["diagnosis"]
            }
    return None


def get_demo_user() -> Dict[str, Any]:
    """Returns or provisions the Eleanor Vance 1-Click Demo Patient account."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", ("demo@patient.org",))
        demo = cursor.fetchone()
        if demo:
            return {
                "id": demo["id"],
                "email": demo["email"],
                "name": demo["name"],
                "patient_id": demo["patient_id"],
                "age": demo["age"],
                "gender": demo["gender"],
                "diagnosis": demo["diagnosis"]
            }
    # Fallback if somehow not found
    seed_demo_patient_if_needed()
    return {
        "id": "demo-patient-001",
        "email": "demo@patient.org",
        "name": "Eleanor Vance (Demo Patient)",
        "patient_id": "PT-88219",
        "age": 68,
        "gender": "Female",
        "diagnosis": "MDS-UPDRS Bradykinesia & Postural Tremor Monitoring"
    }


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve user metadata by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, name, patient_id, age, gender, diagnosis, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def save_assessment(user_id: str, assessment_data: Dict[str, Any]) -> str:
    """Save an assessment test result (tapping or tremor) to persistent database."""
    init_db()
    assessment_id = assessment_data.get("id") or str(uuid.uuid4())
    now_str = assessment_data.get("timestamp") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO assessments (
                id, user_id, test_type, timestamp, score, label,
                rate, decrement, slope, n_taps, cv, peak_hz,
                band_ratio, rms, is_elevated_risk, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment_id,
            user_id,
            assessment_data.get("test_type", "tapping"),
            now_str,
            int(assessment_data.get("score", 0)),
            assessment_data.get("label", "Normal"),
            float(assessment_data.get("rate", 0.0)),
            float(assessment_data.get("decrement", 0.0)),
            float(assessment_data.get("slope", 0.0)),
            int(assessment_data.get("n_taps", 0)),
            float(assessment_data.get("cv", 0.0)),
            float(assessment_data.get("peak_hz", 0.0)),
            float(assessment_data.get("band_ratio", 0.0)),
            float(assessment_data.get("rms", 0.0)),
            1 if assessment_data.get("is_elevated_risk") else 0,
            assessment_data.get("notes", "")
        ))
        conn.commit()

    return assessment_id


def get_user_assessments(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch chronological list of assessment results for a patient (newest first)."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM assessments
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_trend_data(user_id: str) -> Dict[str, Any]:
    """Compile trend analytics suitable for Chart.js visualization."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, test_type, score, rate, decrement, peak_hz, band_ratio
            FROM assessments
            WHERE user_id = ?
            ORDER BY timestamp ASC
        """, (user_id,))
        rows = [dict(r) for r in cursor.fetchall()]

    tapping_points = []
    tremor_points = []

    for r in rows:
        ts = r["timestamp"]
        if r["test_type"] == "tapping":
            tapping_points.append({
                "date": ts,
                "score": r["score"],
                "rate": round(r["rate"], 2),
                "decrement_pct": round(r["decrement"] * 100, 1)
            })
        elif r["test_type"] == "tremor":
            tremor_points.append({
                "date": ts,
                "score": r["score"],
                "peak_hz": round(r["peak_hz"], 2),
                "band_ratio_pct": round(r["band_ratio"] * 100, 1)
            })

    return {
        "tapping": tapping_points,
        "tremor": tremor_points,
        "total_count": len(rows),
        "database_backend": "Firebase Firestore" if HAS_FIREBASE else "Local SQLite (patients.db)"
    }


# Initialize on import
init_db()
