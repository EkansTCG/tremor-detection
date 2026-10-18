"""Multi-Page Clinical Motor Assessment Web Application.

Endpoints:
- /: Welcome redirect -> /home (if authenticated) or /login
- /login: Clinician/patient sign in + 1-Click Demo Patient Login for judges
- /demo-login: Instant 1-click authentication as Eleanor Vance (PT-88219)
- /signup: New patient registration
- /logout: Sign out session
- /home: Patient dashboard, profile banner, two test cards, longitudinal Chart.js trends, history table
- /test/tapping: Decoupled Stage 1 Finger Tapping (MDS-UPDRS 3.4) MediaPipe video feed + scorecard
- /test/tremor: Decoupled Stage 2 Postural Tremor (MDS-UPDRS 3.15) MediaPipe video feed + scorecard
- /video_feed/<test_type>: MJPEG video streaming endpoint
- /api/start_test/<test_type>: Trigger 10-second countdown
- /api/reset_test/<test_type>: Reset trial state
- /api/test_status/<test_type>: Live status, remaining time, results, saved flag
- /api/patient_trends: Longitudinal time-series data for Chart.js
"""
from __future__ import annotations

import os
import functools
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, Response, flash)
from dotenv import load_dotenv

import auth_db
import gemini_service
from camera_stream import StreamManager

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "clin-motor-assessment-technica-secret-2026")

stream_manager = StreamManager.get_instance()


def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_globals():
    backend_name = "Firebase Firestore" if auth_db.HAS_FIREBASE else "Local SQLite (patients.db)"
    return dict(db_backend=backend_name)


# ------------------------------------------------------------------ Authentication Routes
@app.route("/")
def index():
    if session.get("user"):
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        user = auth_db.authenticate_user(email, password)
        if user:
            session["user"] = user
            stream_manager.set_user(user["id"])
            flash(f"Welcome back, {user['name']}.", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid email or password. Please verify credentials or use Demo Login.", "error")
            return render_template("login.html")

    return render_template("login.html")


@app.route("/demo-login", methods=["GET", "POST"])
def demo_login():
    """1-Click Demo Patient Login for judges to bypass credentials during live testing."""
    demo_user = auth_db.get_demo_user()
    session["user"] = demo_user
    stream_manager.set_user(demo_user["id"])
    flash(f"Signed in as Demo Patient: {demo_user['name']} ({demo_user['patient_id']})", "success")
    return redirect(url_for("home"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        patient_id = request.form.get("patient_id", "").strip()
        password = request.form.get("password", "")
        age = request.form.get("age", type=int) or 65
        gender = request.form.get("gender", "Unspecified")
        diagnosis = request.form.get("diagnosis", "Clinical Motor Evaluation")

        res = auth_db.register_user(
            email=email,
            password=password,
            name=name,
            patient_id=patient_id if patient_id else None,
            age=age,
            gender=gender,
            diagnosis=diagnosis
        )

        if res.get("success"):
            user = res.get("user")
            session["user"] = user
            stream_manager.set_user(user["id"])
            flash("Patient account registered successfully.", "success")
            return redirect(url_for("home"))
        else:
            flash(res.get("error", "Registration error."), "error")
            return render_template("signup.html")

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


# ------------------------------------------------------------------ Core Application Pages
@app.route("/home")
@login_required
def home():
    user = session["user"]
    stream_manager.set_user(user["id"])
    assessments = auth_db.get_user_assessments(user["id"], limit=50)
    return render_template("dashboard.html", user=user, assessments=assessments)


@app.route("/profile")
@login_required
def profile():
    user = session["user"]
    stream_manager.set_user(user["id"])
    full_profile = auth_db.get_user_profile(user["id"]) or user
    session["user"] = full_profile
    assessments = auth_db.get_user_assessments(user["id"], limit=100)
    return render_template("profile.html", user=full_profile, assessments=assessments)


@app.route("/api/profile/update", methods=["POST"])
@login_required
def api_update_profile():
    user = session["user"]
    data = request.get_json() or {}
    ok = auth_db.update_user_profile(user["id"], data)
    if ok:
        updated = auth_db.get_user_profile(user["id"]) or user
        session["user"] = updated
        return jsonify({"success": True, "user": updated})
    return jsonify({"success": False, "error": "Unable to update profile"}), 400


@app.route("/api/gemini_summary/tremor", methods=["POST"])
@login_required
def api_gemini_tremor_summary():
    user = session.get("user", {})
    patient_name = user.get("name", "Patient")
    data = request.get_json() or {}
    
    # AI directly analyzes raw kinematic data and determines clinical score, severity & recommendations
    ai_result = gemini_service.analyze_tremor_data_with_ai(data, patient_name=patient_name)
    
    # Sync session results with the AI output so OpenCV HUD and Web UI display identical metrics
    stream_manager.sync_ai_tremor_result(ai_result)

    # Persist the AI-determined clinical score and classification into the patient's record
    if user.get("id"):
        auth_db.update_latest_tremor_assessment(user["id"], {
            "score": ai_result.get("score"),
            "label": ai_result.get("label"),
            "is_elevated_risk": ai_result.get("is_elevated_risk"),
            "risk_badge": ai_result.get("risk_badge"),
            "tremor_type": ai_result.get("tremor_type")
        })
    
    return jsonify({"success": True, "summary": ai_result})


@app.route("/api/gemini_summary/patient", methods=["GET"])
@login_required
def api_gemini_patient_overview():
    user = session["user"]
    full_profile = auth_db.get_user_profile(user["id"]) or user
    assessments = auth_db.get_user_assessments(user["id"], limit=100)
    overview = gemini_service.generate_patient_ai_overview(full_profile, assessments)
    return jsonify({"success": True, "overview": overview})


@app.route("/test/tapping")
@login_required
def test_tapping():
    user = session["user"]
    stream_manager.set_user(user["id"])
    return render_template("test_tapping.html", user=user)


@app.route("/test/tremor")
@login_required
def test_tremor():
    user = session["user"]
    stream_manager.set_user(user["id"])
    return render_template("test_tremor.html", user=user)


# ------------------------------------------------------------------ Video Streaming & APIs
@app.route("/video_feed/<test_type>")
@login_required
def video_feed(test_type: str):
    if test_type not in ("tapping", "tremor"):
        return "Invalid test type", 400

    user = session.get("user")
    if user:
        stream_manager.set_user(user["id"])

    return Response(
        stream_manager.generate_frames(test_type),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/start_test/<test_type>", methods=["POST"])
@login_required
def api_start_test(test_type: str):
    if test_type not in ("tapping", "tremor"):
        return jsonify({"error": "Invalid test type"}), 400
    user = session.get("user")
    if user:
        stream_manager.set_user(user["id"])
    stream_manager.start_test(test_type)
    return jsonify(stream_manager.get_status(test_type))


@app.route("/api/reset_test/<test_type>", methods=["POST"])
@login_required
def api_reset_test(test_type: str):
    if test_type not in ("tapping", "tremor"):
        return jsonify({"error": "Invalid test type"}), 400
    stream_manager.reset_test(test_type)
    return jsonify(stream_manager.get_status(test_type))


@app.route("/api/test_status/<test_type>")
def api_test_status(test_type: str):
    if test_type not in ("tapping", "tremor"):
        return jsonify({"error": "Invalid test type"}), 400
    return jsonify(stream_manager.get_status(test_type))


@app.route("/api/patient_trends")
@login_required
def api_patient_trends():
    user = session["user"]
    data = auth_db.get_trend_data(user["id"])
    return jsonify(data)


if __name__ == "__main__":
    import socket
    import webbrowser
    import threading
    import time

    default_port = int(os.getenv("PORT", 5001))
    port = default_port

    # Verify if port is available, or find next open port
    for p in range(default_port, default_port + 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break

    url = f"http://127.0.0.1:{port}"
    print(f"\n=========================================================")
    print(f" TremorTrack MDS-UPDRS Web Application")
    print(f" Running at: {url}")
    print(f" Persistent Database: {auth_db.SQLITE_DB_PATH}")
    print(f"=========================================================\n")

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

