
# ==============================================================================
#  Driver Drowsiness Detector & SOS System
#  Stable Session State & Persistent Performance Optimization
#  ==============================================================================

import streamlit as st
import cv2
import dlib
import numpy as np
import time
import datetime
import threading
import os
import csv
import json
from imutils import face_utils
from collections import deque
from pathlib import Path
import queue as _queue_module

# ── streamlit-webrtc (browser webcam — replaces server-side cv2.VideoCapture) ──
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
    _RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
except ImportError:
    WEBRTC_AVAILABLE = False
    _RTC_CONFIGURATION = None

# ── Dependency Check ──────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


try:
    import geocoder
    GEOCODER_AVAILABLE = True
except ImportError:
    GEOCODER_AVAILABLE = False

try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    TwilioRestException = Exception

# ── Load .env file early so os.getenv() sees credentials ─────────────────
try:
    from dotenv import load_dotenv as _dotenv_load
    _dotenv_load(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
    print("[Config] .env file loaded (if present)")
except ImportError:
    print("[Config] python-dotenv not installed; run: pip install python-dotenv")

# ── SOS module imports ────────────────────────────────────────────────────
try:
    from sos.config import (
        is_sos_configured,
        get_twilio_account_sid as _sos_cfg_sid,
        get_twilio_auth_token as _sos_cfg_token,
        get_twilio_phone_number as _sos_cfg_from,
        get_emergency_phone_number as _sos_cfg_emergency,
        get_config_status as _sos_config_status,
    )
    from sos.twilio_service import (
        compose_auto_sos_message,
        compose_manual_sos_message,
        compose_test_message,
        send_sms as _twilio_send_sms,
        compose_auto_voice_twiml,
        compose_manual_voice_twiml,
        compose_test_voice_twiml,
        make_voice_call as _twilio_make_voice_call,
    )
    from sos.sos_manager import SosManager
    SOS_MODULE_AVAILABLE = True
except Exception as _sos_import_err:  # noqa: BLE001
    print(f"[SOS] Module import failed: {_sos_import_err}")
    SOS_MODULE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
ALARM_SOUND_PATH = str(BASE_DIR / "assets" / "sounds" / "alarm.mp3")
EMERGENCY_CONTACTS_PATH = BASE_DIR / "emergency_contact.json"
ALARM_RELEASE_SECONDS = 1.0
TARGET_FPS = 20
DROWSY_FRAMES = 15
SLEEP_FRAMES = 30
SOS_FRAMES = TARGET_FPS * 5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG & SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Driver Drowsiness Detector & SOS System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State safely
if "run" not in st.session_state:
    st.session_state.run = False
if "blink_count" not in st.session_state:
    st.session_state.blink_count = 0
if "ear_hist" not in st.session_state:
    st.session_state.ear_hist = deque(maxlen=100)  # Expanded for better graph history
if "time_hist" not in st.session_state:
    st.session_state.time_hist = deque(maxlen=100)
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "status" not in st.session_state:
    st.session_state.status = "Idle"
if "sos_triggered" not in st.session_state:
    st.session_state.sos_triggered = False
if "emergency_number" not in st.session_state:
    st.session_state.emergency_number = ""
if "emergency_number_input" not in st.session_state:
    st.session_state.emergency_number_input = ""
if "sos_debug_log" not in st.session_state:
    st.session_state.sos_debug_log = []
if "last_sos_result" not in st.session_state:
    st.session_state.last_sos_result = {}

# ── SOS State Machine session state ──────────────────────────────────────
if "sos_auto_enabled" not in st.session_state:
    st.session_state.sos_auto_enabled = True
if "sos_manual_confirm_pending" not in st.session_state:
    st.session_state.sos_manual_confirm_pending = False
if "sos_last_sent_time" not in st.session_state:
    st.session_state.sos_last_sent_time = None
if "sos_last_sent_type" not in st.session_state:
    st.session_state.sos_last_sent_type = None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREMIUM STYLING (CSS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {
    --bg-primary: #060b16;
    --bg-secondary: #0c1424;
    --panel: rgba(9, 17, 31, 0.82);
    --panel-strong: rgba(13, 24, 43, 0.94);
    --panel-soft: rgba(12, 20, 36, 0.72);
    --border: rgba(125, 211, 252, 0.16);
    --border-strong: rgba(96, 165, 250, 0.35);
    --text: #f8fbff;
    --muted: #8fa6c2;
    --cyan: #6ee7f9;
    --blue: #60a5fa;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #fb7185;
    --shadow: 0 24px 60px rgba(0, 0, 0, 0.42);
}

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 32%),
        radial-gradient(circle at 85% 18%, rgba(110, 231, 249, 0.14), transparent 22%),
        linear-gradient(180deg, #040814 0%, #07111f 48%, #040814 100%);
}

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 36px 36px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(4, 10, 22, 0.98), rgba(8, 16, 30, 0.96));
    border-right: 1px solid rgba(125, 211, 252, 0.12);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem;
}

[data-testid="block-container"] {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1480px;
}

#MainMenu, footer, header {
    visibility: hidden;
}

h1, h2, h3 {
    letter-spacing: -0.03em;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 1.6rem 1.8rem;
    border-radius: 28px;
    margin-bottom: 1.25rem;
    background:
        linear-gradient(135deg, rgba(10, 19, 35, 0.95), rgba(8, 14, 27, 0.82)),
        radial-gradient(circle at top right, rgba(110, 231, 249, 0.16), transparent 36%);
    border: 1px solid rgba(125, 211, 252, 0.14);
    box-shadow: var(--shadow);
}

.hero-shell::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 0%, rgba(110, 231, 249, 0.04) 35%, transparent 70%);
    pointer-events: none;
}

.hero-kicker {
    color: var(--cyan);
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.28em;
    font-weight: 700;
    margin-bottom: 0.65rem;
}

.hero-title {
    font-size: clamp(2rem, 4vw, 3.3rem);
    font-weight: 700;
    line-height: 1;
    margin: 0;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 1rem;
    max-width: 48rem;
    margin: 0.65rem 0 0;
}

.hero-stat {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    height: 100%;
}

.micro-card,
.glass-card,
.camera-shell {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
}

.micro-card {
    min-height: 118px;
    padding: 1rem 1.1rem;
}

.glass-card {
    padding: 1.15rem 1.2rem 1.2rem;
    margin-bottom: 1rem;
}

.camera-shell {
    padding: 1.05rem;
    margin-bottom: 1rem;
}

.camera-shell::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 24px;
    box-shadow: inset 0 0 0 1px rgba(110, 231, 249, 0.08), 0 0 32px rgba(96, 165, 250, 0.12);
    pointer-events: none;
}

.camera-header,
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.85rem;
}

.panel-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
}

.panel-subtitle {
    margin: 0.2rem 0 0;
    font-size: 0.85rem;
    color: var(--muted);
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.38rem 0.7rem;
    border-radius: 999px;
    background: rgba(110, 231, 249, 0.08);
    border: 1px solid rgba(110, 231, 249, 0.18);
    color: var(--cyan);
    font-size: 0.73rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 700;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.18em;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 0.55rem;
}

.metric-value.metric-small {
    font-size: 1.1rem;
}

.metric-note {
    margin-top: 0.35rem;
    color: var(--muted);
    font-size: 0.82rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.78rem 1.1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    border: 1px solid transparent;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
}

.status-pill::before {
    content: "";
    width: 0.65rem;
    height: 0.65rem;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 16px currentColor;
}

.status-active {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(16, 185, 129, 0.08));
    color: var(--green);
    border-color: rgba(52, 211, 153, 0.3);
}

.status-drowsy {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.18), rgba(251, 191, 36, 0.08));
    color: var(--amber);
    border-color: rgba(251, 191, 36, 0.3);
}

.status-sleeping {
    background: linear-gradient(135deg, rgba(251, 113, 133, 0.18), rgba(239, 68, 68, 0.12));
    color: #fca5a5;
    border-color: rgba(251, 113, 133, 0.42);
    animation: pulse-red 1.5s infinite;
}

.status-sos-triggered {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.86), rgba(239, 68, 68, 0.28));
    color: #ffe2e2;
    border-color: rgba(251, 113, 133, 0.62);
    animation: sos-blink 1s infinite;
}

.status-sos-countdown {
    background: linear-gradient(135deg, rgba(120, 53, 15, 0.88), rgba(245, 158, 11, 0.28));
    color: #fef3c7;
    border-color: rgba(245, 158, 11, 0.62);
    animation: pulse-amber 1s infinite;
}

.status-idle,
.status-no-face {
    background: linear-gradient(135deg, rgba(148, 163, 184, 0.16), rgba(148, 163, 184, 0.06));
    color: #d6deea;
    border-color: rgba(148, 163, 184, 0.28);
}

.fps-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: rgba(110, 231, 249, 0.09);
    padding: 0.5rem 0.75rem;
    border-radius: 999px;
    color: var(--cyan);
    border: 1px solid rgba(110, 231, 249, 0.16);
}

.alert-bar {
    background: linear-gradient(90deg, rgba(127, 29, 29, 0.95), rgba(251, 113, 133, 0.95), rgba(127, 29, 29, 0.95));
    color: white;
    padding: 0.95rem 1rem;
    border-radius: 18px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.18em;
    font-size: 1rem;
    margin-top: 0.25rem;
    box-shadow: 0 0 34px rgba(251, 113, 133, 0.24);
}

.sos-banner {
    background: linear-gradient(90deg, rgba(127, 29, 29, 0.98), rgba(239, 68, 68, 0.92), rgba(127, 29, 29, 0.98));
    color: #fff;
    padding: 1rem 1.2rem;
    border-radius: 18px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.18em;
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.35);
    animation: sos-blink 0.9s infinite;
}

.fatigue-track {
    width: 100%;
    height: 10px;
    margin-top: 0.7rem;
    background: rgba(148, 163, 184, 0.14);
    border-radius: 999px;
    overflow: hidden;
}

.fatigue-fill {
    height: 100%;
    border-radius: 999px;
    box-shadow: 0 0 20px currentColor;
}

[data-testid="stImage"] img {
    border-radius: 18px;
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(251, 113, 133, 0.36); }
    70% { box-shadow: 0 0 0 16px rgba(251, 113, 133, 0); }
    100% { box-shadow: 0 0 0 0 rgba(251, 113, 133, 0); }
}

@keyframes sos-blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.62; transform: scale(1.02); }
}

@keyframes pulse-amber {
    0%, 100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4); }
    50% { box-shadow: 0 0 0 14px rgba(245, 158, 11, 0); }
}

.countdown-banner {
    background: linear-gradient(90deg, rgba(120, 53, 15, 0.97), rgba(245, 158, 11, 0.85), rgba(120, 53, 15, 0.97));
    color: #fff;
    padding: 1rem 1.2rem;
    border-radius: 18px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.12em;
    box-shadow: 0 0 36px rgba(245, 158, 11, 0.32);
    animation: pulse-amber 1.2s infinite;
    margin-top: 0.25rem;
}

.sos-sent-banner {
    background: linear-gradient(90deg, rgba(127, 29, 29, 0.98), rgba(239, 68, 68, 0.92), rgba(127, 29, 29, 0.98));
    color: #fff;
    padding: 1rem 1.2rem;
    border-radius: 18px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.12em;
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.35);
    margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITIES & CONCURRENCY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThreadedCamera:
    """Threaded camera capture to maximize FPS and minimize script blocking."""
    def __init__(self, src=0, width=640, height=480):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

class PerformanceTracker:
    def __init__(self):
        self.times = deque(maxlen=20)
        self.last_time = time.time()
    def tick(self):
        cur = time.time()
        self.times.append(cur - self.last_time)
        self.last_time = cur
    def get_fps(self):
        """Calculates stable average FPS with zero-division protection."""
        try:
            if not self.times or len(self.times) == 0:
                return 0.0
            
            avg_time = sum(self.times) / len(self.times)
            
            # Prevent division by zero if average time is 0 (extremely fast processing)
            if avg_time <= 0:
                return 60.0 # Upper bound fallback for instantaneous processing
            
            fps = 1.0 / avg_time
            # Clamp FPS to a realistic range (0 - 120) for dashboard stability
            return min(120.0, max(0.0, fps))
        except:
            return 0.0

# ── Sound Alert System ───────────────────────────────────────────────────────
# Uses @st.cache_resource so the engine object SURVIVES Streamlit script
# Sound alert system
# Uses @st.cache_resource so the engine object survives Streamlit reruns.


def _read_setting(name: str, default: str = "") -> str:
    """Read a setting from environment variables first, then Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        secret_value = st.secrets.get(name, default)
    except Exception:
        secret_value = default

    return str(secret_value).strip() if secret_value else default


def _sos_log(message: str, *, level: str = "info") -> None:
    """Print SOS logs and mirror them into Streamlit session state."""
    formatted = f"[SOS] {message}"
    print(formatted)

    log_buffer = list(st.session_state.sos_debug_log)
    log_buffer.append(formatted)
    st.session_state.sos_debug_log = log_buffer[-30:]

    st.session_state.last_sos_result = {
        **st.session_state.last_sos_result,
        "level": level,
        "last_log": formatted,
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _mask_secret(value: str) -> str:
    """Show only a small part of sensitive values in logs."""
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _explain_twilio_issue(error_code: int | None, error_message: str, to_number: str) -> str:
    """Return a beginner-friendly explanation for common Twilio failures."""
    message_lower = (error_message or "").lower()

    if error_code == 20003 or "authenticate" in message_lower or "auth" in message_lower:
        return "Twilio rejected the Account SID/Auth Token. Re-check your credentials in environment variables or Streamlit secrets."
    if error_code == 21608:
        return "Trial accounts can only send SMS to verified recipient numbers. Verify the destination number in the Twilio Console."
    if error_code == 21606:
        return "The Twilio sender number is invalid or not SMS-capable for this account."
    if error_code == 21614:
        return "The destination number is not a valid mobile number for SMS delivery."
    if error_code == 30007:
        return "The carrier likely filtered or blocked the SMS content."
    if error_code == 30008:
        return "The destination handset may be unreachable, roaming, or temporarily unavailable."
    if error_code == 30034:
        return "The carrier blocked the message because the sender is not properly registered for that route."
    if to_number.startswith("+91"):
        return (
            "India delivery can be restricted by carrier and sender rules. Trial accounts add extra limits, "
            "so check Twilio Console messaging logs and keep a fallback alert ready."
        )
    return "Check the Twilio error code, Console messaging logs, recipient number format, and carrier restrictions."


def _normalize_phone_number(raw_number: str) -> str:
    """Return a Twilio-friendly phone number in E.164-like format when possible."""
    cleaned = raw_number.strip().replace(" ", "")
    if not cleaned:
        return ""
    if cleaned.startswith("whatsapp:"):
        cleaned = cleaned.replace("whatsapp:", "", 1)
    if cleaned.startswith("+"):
        prefix = "+"
        digits = "".join(ch for ch in cleaned if ch.isdigit())
        return f"{prefix}{digits}" if digits else ""
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return f"+{digits}" if digits else ""


def load_emergency_number() -> str:
    """Load a persisted emergency number from disk."""
    if not EMERGENCY_CONTACTS_PATH.exists():
        return ""

    try:
        with EMERGENCY_CONTACTS_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return _normalize_phone_number(payload.get("emergency_number", ""))
    except Exception as exc:
        print(f"[SOS] Failed to load emergency number: {exc}")
        return ""


def save_emergency_number(number: str) -> bool:
    """Persist the emergency number to disk for future sessions."""
    try:
        with EMERGENCY_CONTACTS_PATH.open("w", encoding="utf-8") as file:
            json.dump({"emergency_number": number}, file, indent=2)
        return True
    except Exception as exc:
        print(f"[SOS] Failed to save emergency number: {exc}")
        return False


if not st.session_state.emergency_number:
    loaded_emergency_number = load_emergency_number()
    if loaded_emergency_number:
        st.session_state.emergency_number = loaded_emergency_number
        st.session_state.emergency_number_input = loaded_emergency_number


def get_location() -> tuple[float | None, float | None, str]:
    """Return the current approximate latitude, longitude and Google Maps link."""
    if not GEOCODER_AVAILABLE:
        print("[SOS] geocoder is not installed; location unavailable")
        return None, None, "Location unavailable"

    try:
        result = geocoder.ip("me")
        if not result or not result.ok or not result.latlng:
            print("[SOS] Unable to resolve current location")
            return None, None, "Location unavailable"

        lat, lon = result.latlng
        maps_link = f"https://www.google.com/maps?q={lat},{lon}"
        print(f"[SOS] Location resolved | lat={lat} | lon={lon}")
        return float(lat), float(lon), maps_link
    except Exception as exc:
        print(f"[SOS] Location lookup failed: {exc}")
        return None, None, "Location unavailable"


# Removed duplicate legacy send_sos() implementation so only the
# diagnostic-aware version below remains active.


def send_sos(trigger_source: str = "manual") -> bool:
    """Send one SOS SMS alert with detailed logging and Twilio diagnostics.

    Credentials are loaded from environment variables (TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER).  Auth token is never logged or
    displayed.  The function never raises — all errors are caught and logged.
    """
    print(f"[SOS] send_sos called | trigger={trigger_source}")

    if not TWILIO_AVAILABLE:
        _sos_log("Twilio SDK is not installed; cannot send SOS messages", level="error")
        return False

    # ── Load credentials from environment (set in .env or system env) ────
    account_sid = _read_setting("TWILIO_ACCOUNT_SID")
    auth_token = _read_setting("TWILIO_AUTH_TOKEN")
    sms_from = _normalize_phone_number(_read_setting("TWILIO_PHONE_NUMBER"))

    # Emergency number: prefer session state (user-configured), fall back to env
    emergency_number = _normalize_phone_number(
        st.session_state.get("emergency_number", "")
        or _read_setting("EMERGENCY_PHONE_NUMBER")
    )

    if not account_sid or not auth_token or not sms_from:
        _sos_log(
            "Missing Twilio settings | "
            f"sid={'SET' if account_sid else 'MISSING'} | "
            f"token={'SET' if auth_token else 'MISSING'} | "
            f"from={sms_from or '<missing>'}",
            level="error",
        )
        return False

    if not emergency_number:
        warning_text = (
            "No emergency number saved. Add a mobile number in the sidebar "
            "and click 'Save Emergency Number' to enable SOS alerts."
        )
        _sos_log(warning_text, level="error")
        st.warning(warning_text, icon="⚠️")
        return False

    # ── Location (best-effort, not required) ──────────────────────────────
    lat, lon, maps_link = get_location()
    location_text = maps_link if (lat is not None and lon is not None) else None

    # ── Compose appropriate message body ──────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if SOS_MODULE_AVAILABLE:
        if trigger_source in ("manual_button", "manual"):
            message_body = compose_manual_sos_message(ts, location_text)
        elif trigger_source in ("manual_test_button", "test"):
            message_body = compose_test_message(ts)
        else:
            message_body = compose_auto_sos_message(ts, location_text)
    else:
        # Fallback inline message when sos module is unavailable
        loc_line = f"\nLocation: {maps_link}" if location_text else "\nLocation: unavailable"
        message_body = (
            f"🚨 DRIVER EMERGENCY ALERT\n\n"
            f"Severe drowsiness detected. Driver did not respond to warnings.\n"
            f"Please check the driver immediately.\n\n"
            f"Time: {ts}{loc_line}\n\n"
            f"Source: {trigger_source} — Driver Drowsiness Detector & SOS System"
        )

    _sos_log(
        f"Sending SMS | trigger={trigger_source} | to={emergency_number} "
        f"| from={sms_from} | location={'yes' if location_text else 'no'}"
    )

    # ── Send via Twilio (delegate to sos module or direct SDK) ────────────
    if SOS_MODULE_AVAILABLE:
        success, friendly_msg, response_dict = _twilio_send_sms(
            account_sid, auth_token, sms_from, emergency_number, message_body
        )
        st.session_state.last_sos_result = {
            **st.session_state.last_sos_result,
            "success": success,
            "trigger_source": trigger_source,
            "timestamp": ts,
            "response": response_dict,
        }
        if success:
            st.session_state.sos_last_sent_time = ts
            st.session_state.sos_last_sent_type = trigger_source
            _sos_log(
                f"SMS sent successfully | "
                f"SID={response_dict.get('sid')} | status={response_dict.get('status')}"
            )
        else:
            _sos_log(f"SMS failed: {friendly_msg}", level="error")
        return success

    # ── Legacy direct Twilio path (sos module unavailable) ────────────────
    try:
        client = Client(account_sid, auth_token)
        sms_message = client.messages.create(
            body=message_body,
            from_=sms_from,
            to=emergency_number,
        )
        fetched_message = client.messages(sms_message.sid).fetch()
        response_summary = {
            "sid": fetched_message.sid,
            "status": fetched_message.status,
            "to": fetched_message.to,
            "from": fetched_message.from_,
            "error_code": fetched_message.error_code,
            "error_message": fetched_message.error_message,
            "date_created": str(fetched_message.date_created),
            "date_sent": str(fetched_message.date_sent),
        }
        st.session_state.last_sos_result = {
            **st.session_state.last_sos_result,
            "success": True,
            "trigger_source": trigger_source,
            "timestamp": ts,
            "response": response_summary,
        }
        st.session_state.sos_last_sent_time = ts
        st.session_state.sos_last_sent_type = trigger_source
        _sos_log(
            f"SMS sent (legacy path) | SID={fetched_message.sid} "
            f"| status={fetched_message.status}"
        )
        return True
    except TwilioRestException as exc:
        error_code = getattr(exc, "code", None)
        error_message = str(exc)
        # Redact auth token from error string
        if auth_token and auth_token in error_message:
            error_message = error_message.replace(auth_token, "[REDACTED]")
        extra_help = _explain_twilio_issue(error_code, error_message, emergency_number)
        st.session_state.last_sos_result = {
            **st.session_state.last_sos_result,
            "success": False,
            "trigger_source": trigger_source,
            "timestamp": ts,
            "response": {
                "status": getattr(exc, "status", None),
                "error_code": error_code,
                "error_message": error_message,
                "more_info": getattr(exc, "more_info", None),
            },
        }
        _sos_log(
            f"Failed | code={error_code} | more_info={getattr(exc, 'more_info', None)}",
            level="error",
        )
        _sos_log(f"Likely cause: {extra_help}", level="error")
        return False
    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        if auth_token and auth_token in err_msg:
            err_msg = err_msg.replace(auth_token, "[REDACTED]")
        st.session_state.last_sos_result = {
            **st.session_state.last_sos_result,
            "success": False,
            "trigger_source": trigger_source,
            "timestamp": ts,
            "response": {"error_message": err_msg},
        }
        _sos_log(f"Failed: {err_msg}", level="error")
        return False


def make_sos_call(trigger_source: str = "manual") -> bool:
    """Make one SOS Voice call with detailed logging and Twilio diagnostics."""
    print(f"[SOS] make_sos_call called | trigger={trigger_source}")

    if not TWILIO_AVAILABLE:
        _sos_log("Twilio SDK is not installed; cannot make SOS calls", level="error")
        return False

    account_sid = _read_setting("TWILIO_ACCOUNT_SID")
    auth_token = _read_setting("TWILIO_AUTH_TOKEN")
    sms_from = _normalize_phone_number(_read_setting("TWILIO_PHONE_NUMBER"))
    emergency_number = _normalize_phone_number(
        st.session_state.get("emergency_number", "")
        or _read_setting("EMERGENCY_PHONE_NUMBER")
    )

    if not account_sid or not auth_token or not sms_from:
        _sos_log("Missing Twilio settings for voice call", level="error")
        return False

    if not emergency_number:
        _sos_log("No emergency number saved for voice call", level="error")
        return False

    lat, lon, maps_link = get_location()
    location_text = maps_link if (lat is not None and lon is not None) else None
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if SOS_MODULE_AVAILABLE:
        if trigger_source in ("manual_button", "manual"):
            twiml_body = compose_manual_voice_twiml(ts, location_text)
        elif trigger_source in ("manual_test_button", "test"):
            twiml_body = compose_test_voice_twiml(ts)
        else:
            twiml_body = compose_auto_voice_twiml(ts, location_text)
    else:
        # Fallback TwiML
        twiml_body = (
            f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            f"<Response><Say voice=\"alice\">"
            f"Emergency Alert from Drowse Guard A I. "
            f"Please check the driver immediately."
            f"</Say></Response>"
        )

    _sos_log(
        f"Making Voice Call | trigger={trigger_source} | to={emergency_number} | from={sms_from}"
    )

    if SOS_MODULE_AVAILABLE:
        success, friendly_msg, response_dict = _twilio_make_voice_call(
            account_sid, auth_token, sms_from, emergency_number, twiml_body
        )
        st.session_state.last_sos_call_result = {
            "success": success,
            "trigger_source": trigger_source,
            "timestamp": ts,
            "response": response_dict,
        }
        if success:
            _sos_log(
                f"Voice Call initiated successfully | "
                f"SID={response_dict.get('sid')} | status={response_dict.get('status')}"
            )
        else:
            _sos_log(f"Voice Call failed: {friendly_msg}", level="error")
        return success
    
    # Legacy direct Twilio path if sos module unavailable
    try:
        client = Client(account_sid, auth_token)
        call = client.calls.create(twiml=twiml_body, to=emergency_number, from_=sms_from)
        response_summary = {
            "sid": call.sid,
            "status": call.status,
            "to": call.to,
            "from": call.from_,
            "date_created": str(call.date_created),
        }
        st.session_state.last_sos_call_result = {
            "success": True,
            "trigger_source": trigger_source,
            "timestamp": ts,
            "response": response_summary,
        }
        _sos_log(f"Voice Call initiated (legacy path) | SID={call.sid}")
        return True
    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        if auth_token and auth_token in err_msg:
            err_msg = err_msg.replace(auth_token, "[REDACTED]")
        st.session_state.last_sos_call_result = {
            "success": False,
            "trigger_source": trigger_source,
            "timestamp": ts,
            "response": {"error_message": err_msg},
        }
        _sos_log(f"Voice Call Failed: {err_msg}", level="error")
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SOS STATE MACHINE HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner=False)
def _get_sos_manager() -> "SosManager":
    """Singleton SOS state machine — persists across all Streamlit reruns."""
    if SOS_MODULE_AVAILABLE:
        sm = SosManager()
        sm.countdown_seconds = 0
        return sm
    # Stub when sos module failed to import (prevents crashes)
    class _StubSosManager:
        state = "IDLE"
        trigger_count = 0
        def notify_severe_drowsiness(self, now=None): return None
        def notify_driver_active(self, now=None): return None
        def cancel_countdown(self): return False
        def enter_cooldown(self, now=None): pass
        def trigger_manual(self, now=None): return False, "SOS module unavailable"
        def get_countdown_remaining(self, now=None): return 0.0
        def get_cooldown_remaining(self, now=None): return 0.0
        def is_in_countdown(self): return False
        def is_in_cooldown(self): return False
        def reset(self): pass
    return _StubSosManager()


def _sos_notify_severe_drowsiness(now: float) -> None:
    """Notify the SOS state machine of severe/sustained drowsiness.

    Called every frame when ``closed_counter >= SOS_FRAMES``.
    Starts the countdown if IDLE; triggers SMS when countdown expires.
    """
    sm = _get_sos_manager()
    action = sm.notify_severe_drowsiness(now)

    if action == "COUNTDOWN_STARTED":
        cd = getattr(sm, "countdown_seconds", SosManager.DEFAULT_COUNTDOWN_SECONDS if SOS_MODULE_AVAILABLE else 10)
        _sos_log(f"SOS countdown started | duration={cd}s | press 'CANCEL SOS' to abort")

    elif action == "TRIGGER_SOS":
        _sos_log("Eyes continuously closed for 6 seconds")
        _sos_log("Automatic SOS triggered")
        success_sms = send_sos(trigger_source="auto_countdown_expired")
        if success_sms:
            _sos_log("SMS sent")
        success_call = make_sos_call(trigger_source="auto_countdown_expired")
        if success_call:
            _sos_log("Voice call initiated")
        success = success_sms or success_call
        sm.enter_cooldown(now)
        cooldown_s = getattr(sm, "_cooldown_seconds", 300)
        if success:
            st.session_state.sos_triggered = True
            _sos_log(
                f"Automatic SOS triggered | cooldown={cooldown_s}s "
                "(no new auto-SOS during cooldown)"
            )
        else:
            # Still enter cooldown to prevent spam on repeated failures
            _sos_log(
                "Automatic SOS FAILED (both SMS and Call) | entering cooldown to prevent spam",
                level="error",
            )


def _sos_notify_driver_active() -> None:
    """Notify the SOS state machine that the driver is awake.

    Called whenever ``closed_counter`` resets to 0 (eyes opened).
    Cancels an active countdown if one is running.
    """
    sm = _get_sos_manager()
    action = sm.notify_driver_active()
    if action == "COUNTDOWN_CANCELLED":
        _sos_log("SOS countdown cancelled — driver became active (eyes opened)")
        st.session_state.sos_triggered = False
    elif action == "COOLDOWN_ENDED":
        _sos_log("SOS cooldown period ended — system ready for new monitoring")
        st.session_state.sos_triggered = False


def _sos_cancel_countdown() -> None:
    """Cancel the active SOS countdown when the driver clicks 'Cancel SOS'."""
    sm = _get_sos_manager()
    if sm.cancel_countdown():
        _sos_log("SOS countdown manually cancelled by driver")
        st.session_state.sos_triggered = False


def _sos_trigger_manual() -> bool:
    """Trigger a manual SOS alert with cooldown protection.

    Returns True if the SMS or Call was dispatched (or attempted).
    """
    sm = _get_sos_manager()
    can_send, result = sm.trigger_manual()
    if not can_send:
        _sos_log(f"Manual SOS blocked: {result}", level="error")
        return False
    _sos_log("Manual SOS triggered by driver")
    success_sms = send_sos(trigger_source="manual_button")
    success_call = make_sos_call(trigger_source="manual_button")
    success = success_sms or success_call
    sm.enter_cooldown()
    if success:
        st.session_state.sos_triggered = True
    return success


def _sos_get_countdown_remaining(now: float) -> float:
    """Return seconds remaining in the SOS countdown (0.0 if not counting)."""
    return _get_sos_manager().get_countdown_remaining(now)


def _sos_get_state() -> str:
    """Return the current SOS state string."""
    return _get_sos_manager().state


import base64

@st.cache_data(show_spinner=False)
def _get_alarm_base64() -> str:
    """Load the MP3 file into a base64 string for HTML5 audio."""
    if not os.path.exists(ALARM_SOUND_PATH):
        return ""
    try:
        with open(ALARM_SOUND_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

def start_alert():
    """Trigger the browser-based alarm."""
    if not st.session_state.get("alarm_active", False):
        print("[AlertAPI] start_alert() called")
        st.session_state.alarm_active = True

def stop_alert():
    """Stop the browser-based alarm."""
    if st.session_state.get("alarm_active", False):
        print("[AlertAPI] stop_alert() called")
        st.session_state.alarm_active = False

def render_browser_alarm():
    """Injects an invisible HTML5 audio element into the DOM if the alarm is active."""
    if st.session_state.get("alarm_active", False):
        b64 = _get_alarm_base64()
        if b64:
            st.markdown(
                f'<audio autoplay loop style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
                unsafe_allow_html=True,
            )

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return detector, predictor
    except Exception as exc:
        print(f"[ModelLoad] Failed to load models: {exc}")
        return None, None

DETECTOR, PREDICTOR = load_models()

# ── EAR calculation (moved to module-level so DrowsinessTransformer can use it) ─
def eye_ear(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def _make_proc_state():
    """Return a fresh detection-state dict for a new monitoring session."""
    return {
        "f_idx": 0,
        "closed_counter": 0,
        "closure_start_time": 0.0,
        "last_sleep_detected_at": 0.0,
        "was_closed": False,
        "perf": PerformanceTracker(),
    }


# ── Browser-webcam frame processor (replaces ThreadedCamera + while-loop) ──────
# Receives frames from the user's browser webcam via streamlit-webrtc,
# runs the existing dlib detection pipeline, annotates frames, and places
# results in a thread-safe queue for the Streamlit main thread to consume.
if WEBRTC_AVAILABLE:
    class DrowsinessTransformer(VideoProcessorBase):
        """VideoProcessor that runs the dlib drowsiness pipeline on browser frames."""

        def __init__(self):
            self.result_queue: _queue_module.Queue = _queue_module.Queue(maxsize=4)
            # Mirrored from sidebar settings — written by main thread, read by recv()
            self.draw_box: bool = True
            self.draw_landmarks: bool = False
            self.ear_threshold: float = 0.25
            self.frame_skip: int = 2
            self._f_idx: int = 0
            # Last valid detection (for per-frame annotation between detections)
            self._last_rect = None
            self._last_shape = None
            self._last_is_closed: bool = False
            self._last_face_detected: bool = False

        def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
            # to_ndarray(format="bgr24") → OpenCV-compatible BGR numpy array
            img = frame.to_ndarray(format="bgr24")
            self._f_idx += 1

            # Run dlib detection only every frame_skip frames (same as original)
            if self._f_idx % max(1, self.frame_skip) == 0:
                small = cv2.resize(img, (320, 240))
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

                result = {"face_detected": False, "timestamp": time.time()}

                if DETECTOR is not None and PREDICTOR is not None:
                    faces = DETECTOR(gray_small, 0)

                    if faces:
                        face = faces[0]
                        # Scale landmark rect back to full-resolution coordinates
                        r = dlib.rectangle(
                            face.left() * 2, face.top() * 2,
                            face.right() * 2, face.bottom() * 2,
                        )
                        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        shape = PREDICTOR(gray_full, r)
                        shape = face_utils.shape_to_np(shape)

                        left_eye = shape[36:42]
                        right_eye = shape[42:48]
                        cur_ear = (eye_ear(left_eye) + eye_ear(right_eye)) / 2.0
                        is_closed = cur_ear < self.ear_threshold

                        # Cache for annotation on frames between detections
                        self._last_rect = r
                        self._last_shape = shape
                        self._last_is_closed = is_closed
                        self._last_face_detected = True

                        result.update({
                            "face_detected": True,
                            "ear": cur_ear,
                            "is_closed": is_closed,
                        })
                    else:
                        self._last_face_detected = False
                        self._last_rect = None

                # Non-blocking enqueue — drop oldest entry if queue is full
                try:
                    self.result_queue.put_nowait(result)
                except _queue_module.Full:
                    try:
                        self.result_queue.get_nowait()
                    except _queue_module.Empty:
                        pass
                    try:
                        self.result_queue.put_nowait(result)
                    except _queue_module.Full:
                        pass

            # Draw annotations every frame using last valid detection result
            if self._last_face_detected and self._last_rect is not None:
                if self.draw_box:
                    color = (0, 0, 255) if self._last_is_closed else (16, 185, 129)
                    r = self._last_rect
                    cv2.rectangle(
                        img,
                        (r.left(), r.top()),
                        (r.right(), r.bottom()),
                        color,
                        2,
                    )
                if self.draw_landmarks and self._last_shape is not None:
                    for (x, y) in self._last_shape:
                        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

            # Return annotated BGR frame — streamlit-webrtc streams it to the browser
            return av.VideoFrame.from_ndarray(img, format="bgr24")

else:
    DrowsinessTransformer = None  # type: ignore[assignment,misc]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown(
        """
        <div class='glass-card' style='padding:1rem 1rem 0.9rem; margin-bottom:1rem;'>
            <div class='eyebrow'>System Core</div>
            <h2 style='margin:0.7rem 0 0.25rem; color:#f8fbff;'>Driver Drowsiness Detector & SOS System</h2>
            <p class='panel-subtitle'>AI-powered vigilance dashboard for driver state monitoring.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    # Persistent Master Toggle
    monitor_on = st.toggle("Start Monitoring System", value=st.session_state.run)
    
    # Handle state transition
    if monitor_on != st.session_state.run:
        st.session_state.run = monitor_on
        if monitor_on:
            st.session_state.start_time = time.time()
            st.session_state.blink_count = 0
            st.session_state.sos_triggered = False
            st.session_state.ear_hist.clear()
            st.session_state.time_hist.clear()
            st.session_state.proc_state = _make_proc_state()  # fresh detection counters
        else:
            st.session_state.proc_state = None  # release state when monitoring stops
        st.rerun()

    st.markdown("---")
    
    # Performance Mode
    perf_mode = st.select_slider("Performance Mode", options=["Accuracy", "Balanced", "Speed"], value="Balanced")
    FRAME_SKIP = {"Accuracy": 1, "Balanced": 2, "Speed": 4}[perf_mode]

    st.markdown("#### Emergency Contact")
    st.text_input(
        "Mobile Number",
        key="emergency_number_input",
        placeholder="+919876543210",
        help="Enter a Twilio-reachable mobile number in international format.",
    )
    if st.button("Save Emergency Number", use_container_width=True):
        normalized_number = _normalize_phone_number(st.session_state.emergency_number_input)
        if normalized_number:
            st.session_state.emergency_number = normalized_number
            if save_emergency_number(normalized_number):
                st.success("Number saved successfully.")
            else:
                st.warning("Number saved for this session, but local persistence failed.", icon="⚠️")
        else:
            st.warning("Enter a valid mobile number before saving.", icon="⚠️")

    if st.session_state.emergency_number:
        st.caption(f"Current saved number: {st.session_state.emergency_number}")
    else:
        st.caption("No emergency number saved yet.")
    _t_col1, _t_col2 = st.columns(2)
    with _t_col1:
        if st.button("TEST SMS", use_container_width=True):
            if send_sos(trigger_source="manual_test_button"):
                st.success("Test SMS sent.")
            else:
                st.warning("Test SMS failed.", icon="⚠️")
    with _t_col2:
        if st.button("TEST CALL", use_container_width=True):
            if make_sos_call(trigger_source="manual_test_button"):
                st.success("Test Call initiated.")
            else:
                st.warning("Test Call failed.", icon="⚠️")

    # ── SOS Status & Cancel Countdown ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🚨 SOS Alert Status")

    _sm_state = _sos_get_state()
    _sm = _get_sos_manager()

    if _sm_state == "COUNTDOWN":
        _remaining = _sos_get_countdown_remaining(time.time())
        st.warning(
            f"⚠️ **SOS COUNTDOWN ACTIVE**\n\nSOS alert in **{int(_remaining) + 1}** seconds.\n\nPress below to cancel.",
            icon="🚨",
        )
        if st.button("🚫 CANCEL SOS", use_container_width=True, type="primary"):
            _sos_cancel_countdown()
            st.success("SOS countdown cancelled.")
            st.rerun()

    elif _sm_state == "COOLDOWN":
        _cd_remaining = _sm.get_cooldown_remaining(time.time())
        st.info(
            f"ℹ️ SOS sent. Cooldown active — **{int(_cd_remaining)}s** remaining.\n\n"
            "Auto-SOS will resume after the cooldown period.",
            icon="ℹ️",
        )
        if st.session_state.sos_last_sent_time:
            st.caption(f"Last alert: {st.session_state.sos_last_sent_time}")

    elif _sm_state == "TRIGGERED":
        st.warning("🚨 SOS alert is being processed...", icon="🚨")

    else:  # IDLE
        _cfg = _sos_config_status() if SOS_MODULE_AVAILABLE else {}
        if _cfg.get("is_fully_configured", False) or (
            _read_setting("TWILIO_ACCOUNT_SID") and _read_setting("TWILIO_AUTH_TOKEN")
        ):
            st.success("🟢 SOS System Ready", icon="✅")
        else:
            st.warning(
                "⚠️ SOS not configured. Add credentials to `.env` and restart.",
                icon="⚠️",
            )

    # ── Manual SOS with 2-step Confirmation ───────────────────────────────
    st.markdown("---")
    st.markdown("#### 🆘 Manual SOS")

    if _sm_state == "COOLDOWN":
        _cd_rem = _sm.get_cooldown_remaining(time.time())
        st.info(f"SOS cooldown active. Manual SOS available in {int(_cd_rem)}s.")
    elif not st.session_state.get("sos_manual_confirm_pending", False):
        st.caption("Use only in a genuine emergency situation.")
        if st.button("🚨 SEND MANUAL SOS", use_container_width=True):
            st.session_state.sos_manual_confirm_pending = True
            st.rerun()
    else:
        st.error(
            "⚠️ **Are you sure?**\n\nThis will immediately send an emergency SMS "
            "to your saved contact.",
            icon="🚨",
        )
        _col_confirm, _col_cancel = st.columns(2)
        with _col_confirm:
            if st.button("✅ CONFIRM SOS", use_container_width=True, type="primary"):
                st.session_state.sos_manual_confirm_pending = False
                _manual_ok = _sos_trigger_manual()
                if _manual_ok:
                    st.success("🚨 Emergency SMS sent!")
                else:
                    st.error("SMS failed. Check Twilio config and recipient number.")
                st.rerun()
        with _col_cancel:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.sos_manual_confirm_pending = False
                st.rerun()

    # ── SOS Settings ──────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ SOS Settings", expanded=False):
        _auto_toggle = st.toggle(
            "Automatic SOS",
            value=st.session_state.get("sos_auto_enabled", True),
            help="Automatically trigger SOS after sustained severe drowsiness + countdown.",
        )
        if _auto_toggle != st.session_state.get("sos_auto_enabled", True):
            st.session_state.sos_auto_enabled = _auto_toggle

        _new_cd = st.slider(
            "Countdown Duration (seconds)",
            min_value=0,
            max_value=30,
            value=SosManager.DEFAULT_COUNTDOWN_SECONDS if SOS_MODULE_AVAILABLE else 0,
            step=1,
            help="How long the countdown runs before automatically sending SOS.",
        )
        if SOS_MODULE_AVAILABLE:
            _get_sos_manager().countdown_seconds = _new_cd

        st.caption(
            f"Auto SOS: {'Enabled ✅' if st.session_state.get('sos_auto_enabled', True) else 'Disabled ❌'} | "
            f"Countdown: {_new_cd}s | "
            f"Cooldown: {SosManager.DEFAULT_COOLDOWN_SECONDS // 60}min"
            if SOS_MODULE_AVAILABLE else
            f"Auto SOS: {'Enabled ✅' if st.session_state.get('sos_auto_enabled', True) else 'Disabled ❌'} | "
            f"Countdown: {_new_cd}s"
        )

    st.markdown("#### 👁️ Sensitivity")

    if st.session_state.emergency_number.startswith("+91"):
        st.info(
            "India numbers can fail even when Twilio accepts the API request. Trial accounts, carrier filtering, "
            "and India sender registration rules may block SMS delivery.",
            icon="ℹ️",
        )

    with st.expander("SOS Debug Panel", expanded=False):
        st.caption("Recommended Twilio settings keys: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER")
        if st.session_state.last_sos_result:
            st.json(st.session_state.last_sos_result)
        else:
            st.caption("No SOS attempt has been recorded yet.")

        if st.session_state.sos_debug_log:
            st.code("\n".join(st.session_state.sos_debug_log), language="text")
        else:
            st.caption("Debug logs will appear here after the first SOS attempt.")

    ear_threshold = st.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01)
    st.caption(
        f"FPS ~ {TARGET_FPS} | "
        f"Drowsy >= {DROWSY_FRAMES} frames | "
        f"Sleeping >= {SLEEP_FRAMES} frames | "
        f"SOS >= {SOS_FRAMES} frames (~{SOS_FRAMES / TARGET_FPS:.1f}s)"
    )
    if TWILIO_AVAILABLE and GEOCODER_AVAILABLE:
        st.caption("SOS messaging: ready when TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER are configured.")
    else:
        st.caption("SOS messaging: install `twilio` and `geocoder` to enable SMS/WhatsApp alerts.")
    
    st.markdown("#### 🎨 UI Options")
    draw_landmarks = st.toggle("Show Landmarks", value=False)
    draw_box = st.toggle("Show Box", value=True)
    live_graph_active = st.toggle("Live Analytics", value=True)

    st.markdown("#### 🔔 Alert Settings")
    alarm_enabled = st.toggle(
        "Enable Alarm",
        value=True,
        help="Play the alarm.mp3 file only when the driver enters the sleeping state"
    )
    if alarm_enabled and not os.path.exists(ALARM_SOUND_PATH):
        st.warning(f"Alarm file not found at {ALARM_SOUND_PATH}.", icon="⚠️")
    else:
        st.caption(f"Alarm file: {ALARM_SOUND_PATH}")
        st.caption("Audio backend: HTML5 Browser Audio")

    # One-click sound test bypasses detection logic
    if st.button("🔊 Test Alarm Sound", help="Play the configured alarm.mp3 file once"):
        if not os.path.exists(ALARM_SOUND_PATH):
            st.error(f"Alarm file not found at {ALARM_SOUND_PATH}.", icon="❌")
        else:
            print(f"[AlarmTest] Testing alarm file: {ALARM_SOUND_PATH}")
            start_alert()
            st.toast("🔊 Alarm sound started.", icon="🔊")

    if st.button("🔇 Stop Alarm Test", help="Stop the currently playing alarm"):
        print("[AlarmTest] Stop Alarm Test button pressed")
        stop_alert()
        st.toast("🔇 Alarm sound stopped.", icon="🔇")



    # Live debug counters — visible while monitoring is active
    st.markdown("#### 🛠️ Live Debug")
    debug_placeholder = st.empty()

    st.markdown("---")
    st.caption("v3.1 Stable Release | OpenCV + dlib")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN UI LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
h_col1, h_col2, h_col3 = st.columns([2.6, 1, 1], gap="medium")
with h_col1:
    st.markdown(
        """
        <div class='hero-shell'>
            <div class='hero-kicker'>Autonomous Safety Console</div>
            <h1 class='hero-title'>Driver Drowsiness Detector & SOS System</h1>
            <p class='hero-subtitle'>
                Real-time fatigue intelligence with live camera telemetry, alert escalation,
                and emergency response readiness in a premium control-room layout.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_placeholder = st.empty()
with h_col2:
    fps_placeholder = st.empty()
with h_col3:
    time_placeholder = st.empty()

col_v, col_a = st.columns([1.8, 1], gap="large")

with col_v:
    st.markdown(
        """
        <div class='camera-shell'>
            <div class='camera-header'>
                <div>
                    <div class='eyebrow'>Vision Feed</div>
                    <h3 class='panel-title'>Driver Camera</h3>
                    <p class='panel-subtitle'>Live monitoring window with enhanced focus frame.</p>
                </div>
                <div class='fps-badge'>LIVE CAMERA</div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    # ── Browser webcam via streamlit-webrtc ─────────────────────────────────
    # Replaces: cv2.VideoCapture(0) / ThreadedCamera
    # The browser requests camera permission; frames are streamed to the server
    # via WebRTC and processed by DrowsinessTransformer.recv().
    if (
        st.session_state.run
        and WEBRTC_AVAILABLE
        and DETECTOR is not None
        and PREDICTOR is not None
    ):
        ctx = webrtc_streamer(
            key="drowsiness-monitor",
            video_processor_factory=DrowsinessTransformer,
            rtc_configuration=_RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.caption(
            "📷 Click **START** above to grant camera permission and begin monitoring. "
            "Your webcam runs in the browser — no camera access on the server."
        )
    elif st.session_state.run and not WEBRTC_AVAILABLE:
        ctx = None
        st.error(
            "streamlit-webrtc is not installed. "
            "Add `streamlit-webrtc>=0.47.0` to requirements.txt and redeploy.",
            icon="❌",
        )
    else:
        ctx = None
    video_placeholder = st.empty()  # Standby UI shown when not monitoring
    st.markdown("</div>", unsafe_allow_html=True)
    alert_placeholder = st.empty()
with col_a:
    st.markdown(
        """
        <div class='glass-card'>
            <div class='panel-header'>
                <div>
                    <div class='metric-label'>Fatigue Probability</div>
                    <p class='panel-subtitle'>Composite risk score from EAR and closed-eye duration.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    fatigue_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown(
            """
            <div class='glass-card'>
                <div class='metric-label'>Blink Count</div>
                <p class='metric-note'>Detected blink events</p>
            """,
            unsafe_allow_html=True,
        )
        blink_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    with s_col2:
        st.markdown(
            """
            <div class='glass-card'>
                <div class='metric-label'>Eye Aspect Ratio</div>
                <p class='metric-note'>Current eyelid openness score</p>
            """,
            unsafe_allow_html=True,
        )
        ear_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='glass-card'>
            <div class='panel-header'>
                <div>
                    <div class='metric-label'>EAR History</div>
                    <p class='panel-subtitle'>Recent signal trend for fatigue drift analysis.</p>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )
    graph_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  WEBRTC PROCESSING LOOP  (replaces ThreadedCamera + while-loop)
#
#  Architecture:
#    DrowsinessTransformer.recv()  ← background thread (streamlit-webrtc)
#        ↓ result dict via queue
#    Main thread (each st.rerun)   → drowsiness state → SOS/alarm → UI update
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state.run and DETECTOR is not None and PREDICTOR is not None:

    # ── Ensure proc_state exists (safety net for unexpected reruns) ───────
    if st.session_state.get("proc_state") is None:
        st.session_state.proc_state = _make_proc_state()
    ps = st.session_state.proc_state

    # ── Propagate sidebar settings into the transformer (GIL-safe writes) ─
    if ctx is not None and ctx.video_processor is not None:
        ctx.video_processor.draw_box = draw_box
        ctx.video_processor.draw_landmarks = draw_landmarks
        ctx.video_processor.ear_threshold = ear_threshold
        ctx.video_processor.frame_skip = FRAME_SKIP

    # ── Drain result queue — keep only the latest result ─────────────────
    result = None
    if ctx is not None and ctx.video_processor is not None:
        latest = None
        while True:
            try:
                latest = ctx.video_processor.result_queue.get_nowait()
            except _queue_module.Empty:
                break
        result = latest

    # ── Process detection result from transformer ─────────────────────────
    if result is not None:
        perf_tracker = ps["perf"]
        perf_tracker.tick()
        ps["f_idx"] += 1

        now = result["timestamp"]
        closed_counter = ps["closed_counter"]
        closure_start_time = ps["closure_start_time"]
        last_sleep_detected_at = ps["last_sleep_detected_at"]

        if result.get("face_detected"):
            cur_ear = result["ear"]
            is_closed = result["is_closed"]

            # ── Blink filtering and time-based eye-closure detection ──────
            if is_closed:
                if closed_counter == 0:
                    closure_start_time = now
                closed_counter += 1
            else:
                if 0 < closed_counter < DROWSY_FRAMES:
                    st.session_state.blink_count += 1
                    print(
                        f"[EYE] Blink counted | closed_frames={closed_counter} "
                        f"| blink_count={st.session_state.blink_count}"
                    )
                closed_counter = 0
                closure_start_time = 0.0
                # Notify SOS state machine that driver is awake.
                # This cancels any active countdown automatically.
                _sos_notify_driver_active()
                st.session_state.sos_triggered = False

            ps["was_closed"] = is_closed
            ps["closed_counter"] = closed_counter
            ps["closure_start_time"] = closure_start_time

            # ── Drowsiness state machine (identical to original logic) ────
            # Frame-based states:
            #   blink    → below DROWSY_FRAMES, no alert
            #   drowsy   → >= DROWSY_FRAMES
            #   sleeping → >= SLEEP_FRAMES
            #   sos      → >= 6 seconds continuous closure (time-based)
            if closure_start_time > 0 and (now - closure_start_time) >= 6.0:
                last_sleep_detected_at = now
                # Notify state machine; it handles countdown + SMS + cooldown
                if st.session_state.get("sos_auto_enabled", True):
                    _sos_notify_severe_drowsiness(now)
                # Set status based on current SOS state
                _current_sos_state = _sos_get_state()
                if _current_sos_state == "COUNTDOWN":
                    st.session_state.status = "SOS Countdown"
                elif _current_sos_state in ("TRIGGERED", "COOLDOWN"):
                    st.session_state.status = "SOS Triggered"
                else:
                    st.session_state.status = "Sleeping"
                if alarm_enabled:
                    start_alert()
                else:
                    print("[SOS] Alarm suppressed because alarm_enabled is False")
                    stop_alert()

            elif closed_counter >= SLEEP_FRAMES:
                last_sleep_detected_at = now
                st.session_state.status = "Sleeping"
                print(
                    f"[ALARM] Sleeping detected | closed_frames={closed_counter} "
                    f"| threshold={SLEEP_FRAMES} | alarm_enabled={alarm_enabled}"
                )
                if alarm_enabled:
                    start_alert()
                else:
                    print("[ALARM] Alarm suppressed because alarm_enabled is False")
                    stop_alert()
            elif closed_counter >= DROWSY_FRAMES:
                st.session_state.status = "Drowsy"
                print(
                    f"[ALARM] Drowsy detected | closed_frames={closed_counter} "
                    f"| threshold={DROWSY_FRAMES}"
                )
                time_since_sleep = now - last_sleep_detected_at
                if time_since_sleep >= ALARM_RELEASE_SECONDS:
                    print("[ALARM] Release hysteresis passed in drowsy state | stopping alarm")
                    stop_alert()
                else:
                    print(
                        f"[ALARM] Holding alarm during drowsy fluctuation | "
                        f"remaining={ALARM_RELEASE_SECONDS - time_since_sleep:.2f}s"
                    )
            else:
                st.session_state.status = "Active"
                if closed_counter > 0:
                    print(
                        f"[ALARM] Eyes closed below drowsy threshold | "
                        f"closed_frames={closed_counter}"
                    )
                else:
                    print("[ALARM] Active state detected")
                time_since_sleep = now - last_sleep_detected_at
                if time_since_sleep >= ALARM_RELEASE_SECONDS:
                    print("[ALARM] Release hysteresis passed in active state | stopping alarm")
                    stop_alert()
                else:
                    print(
                        f"[ALARM] Holding alarm during active fluctuation | "
                        f"remaining={ALARM_RELEASE_SECONDS - time_since_sleep:.2f}s"
                    )

            ps["last_sleep_detected_at"] = last_sleep_detected_at
            st.session_state.ear_hist.append(cur_ear)
            st.session_state.time_hist.append(now - st.session_state.start_time)

        else:
            # No face detected
            st.session_state.status = "No Face"
            ps["was_closed"] = False
            ps["closed_counter"] = 0
            time_since_sleep = now - ps["last_sleep_detected_at"]
            if time_since_sleep >= ALARM_RELEASE_SECONDS:
                print("[ALARM] No face detected and hysteresis passed | stopping alarm")
                stop_alert()
            else:
                print(
                    f"[ALARM] No face fluctuation | keeping alarm alive for "
                    f"{ALARM_RELEASE_SECONDS - time_since_sleep:.2f}s"
                )

    # ── Live debug counter update in sidebar ─────────────────────────────
    # Safe read: ear_hist may be empty on startup frames before any face
    # has been detected, so always guard with a fallback.
    closed_counter = ps.get("closed_counter", 0)
    _ear_display = f"{st.session_state.ear_hist[-1]:.3f}" if st.session_state.ear_hist else "--"
    _closed_display = str(closed_counter) if closed_counter > 0 else "--"
    alert_state = "PLAYING" if st.session_state.get("alarm_active", False) else "OFF"
    _dbg_sos_state = _sos_get_state()
    _dbg_countdown = _sos_get_countdown_remaining(time.time())
    _dbg_countdown_str = f"{_dbg_countdown:.1f}s" if _dbg_sos_state == "COUNTDOWN" else "--"
    debug_placeholder.markdown(
        f"""
        | Counter | Value |
        |---|---|
        | `closed_frames` | **{_closed_display}** |
        | `sleep_after` | **{SLEEP_FRAMES} frames** |
        | `sos_after` | **{SOS_FRAMES} frames** |
        | `drowsy_after` | **{DROWSY_FRAMES} frames** |
        | `EAR` | **{_ear_display}** |
        | `SOS State` | **{_dbg_sos_state}** |
        | `Countdown` | **{_dbg_countdown_str}** |
        | `Alarm` | {alert_state} |
        """,
        unsafe_allow_html=False,
    )

    # ── UI update ─────────────────────────────────────────────────────────
    st_text = st.session_state.status
    s_pill = st_text.lower().replace(" ", "-")
    status_placeholder.markdown(
        f"<div class='status-pill status-{s_pill}'>{st_text}</div>",
        unsafe_allow_html=True,
    )

    perf_obj = ps["perf"]
    fps_placeholder.markdown(
        f"""
        <div class='micro-card hero-stat'>
            <div>
                <div class='metric-label'>System Performance</div>
                <div class='fps-badge'>{perf_obj.get_fps():.1f} FPS</div>
                <div class='metric-note'>Realtime processing throughput</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    elap = int(time.time() - st.session_state.start_time)
    time_placeholder.markdown(
        f"""
        <div class='micro-card hero-stat'>
            <div>
                <div class='metric-label'>Elapsed Time</div>
                <div class='metric-value metric-small'>{elap//60:02d}:{elap%60:02d}s</div>
                <div class='metric-note'>Session runtime</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ear_now = st.session_state.ear_hist[-1] if st.session_state.ear_hist else 0.0
    fatigue = min(
        100,
        max(
            0,
            (ear_threshold - ear_now) / ear_threshold * 40
            + min(60, closed_counter * 2),
        ),
    )
    f_color = "#ef4444" if fatigue > 70 else ("#fbbf24" if fatigue > 35 else "#10b981")
    fatigue_placeholder.markdown(
        f"""
        <div class='metric-value' style='color:{f_color};'>{fatigue:.1f}%</div>
        <div class='metric-note'>Risk escalates with sustained eye closure and reduced EAR.</div>
        <div class='fatigue-track'>
            <div class='fatigue-fill' style='width:{fatigue}%; background:{f_color}; color:{f_color};'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    blink_placeholder.markdown(
        f"<div class='metric-value'>{st.session_state.blink_count}</div>",
        unsafe_allow_html=True,
    )
    ear_placeholder.markdown(
        f"<div class='metric-value'>{ear_now:.3f}</div>",
        unsafe_allow_html=True,
    )

    # ── Alert / Countdown Banner ──────────────────────────────────────────
    _now_ts = time.time()

    if st_text == "SOS Countdown":
        _secs_left = _sos_get_countdown_remaining(_now_ts)
        alert_placeholder.markdown(
            f"""<div class='countdown-banner'>
                🚨 WARNING — SEVERE DROWSINESS DETECTED<br>
                <span style='font-size:1.6rem; font-weight:900;'>
                    SOS IN {int(_secs_left) + 1} SECOND{'S' if int(_secs_left) + 1 != 1 else ''}
                </span><br>
                <span style='font-size:0.82rem; opacity:0.85; letter-spacing:0.06em;'>
                    Press <b>CANCEL SOS</b> in the sidebar to abort
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
    elif st_text == "SOS Triggered":
        _sent_ts = st.session_state.get("sos_last_sent_time", "")
        _sent_label = f"Alert sent at {_sent_ts}" if _sent_ts else "Emergency alert sent"
        alert_placeholder.markdown(
            f"""<div class='sos-sent-banner'>
                🚨 SOS TRIGGERED • DRIVER UNRESPONSIVE • EMERGENCY ALERT ACTIVE<br>
                <span style='font-size:0.82rem; opacity:0.8;'>{_sent_label}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    elif st_text == "Sleeping":
        alert_placeholder.markdown(
            "<div class='alert-bar'>WAKE UP DRIVER • EYES CLOSED DETECTED</div>",
            unsafe_allow_html=True,
        )
    else:
        alert_placeholder.empty()

    # ── Analytics Graph (every 10 processing results) ────────────────────
    f_idx = ps.get("f_idx", 0)
    if live_graph_active and PLOTLY_AVAILABLE and f_idx % 10 == 0:
        if len(st.session_state.ear_hist) > 5:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(st.session_state.time_hist),
                    y=list(st.session_state.ear_hist),
                    mode="lines",
                    line=dict(color="#6ee7f9", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(110,231,249,0.12)",
                )
            )
            fig.update_layout(
                height=180,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(
                    range=[0.1, 0.4],
                    showgrid=True,
                    gridcolor="rgba(148,163,184,0.12)",
                    tickfont=dict(color="#8fa6c2"),
                    zeroline=False,
                ),
                showlegend=False,
            )
            graph_placeholder.empty()
            graph_placeholder.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"ear_analytics_{f_idx}",
            )

    # ── Rerun to keep processing browser webcam frames ───────────────────
    # The WebRTC video stream is independent and always runs at camera FPS.
    # This rerun only updates metrics/banners at ~20 FPS.
    time.sleep(0.05)
    st.rerun()

else:
    # ── Standby UI (monitoring off or models unavailable) ────────────────
    stop_alert()  # Ensure alarm stops when monitoring is disabled
    if "proc_state" in st.session_state:
        st.session_state.proc_state = None

    video_placeholder.markdown("""
        <div style='background: #0b1120; border: 2px dashed #1e293b; border-radius: 16px; height: 360px; display: flex; align-items: center; justify-content: center; flex-direction: column;'>
            <div style='font-size: 3.5rem; filter: grayscale(1);'>📸</div>
            <h3 style='color: #64748b; margin-top:15px;'>System Offline</h3>
            <p style='color: #475569; font-size: 0.9rem;'>Use the sidebar toggle to begin monitoring.</p>
        </div>
    """, unsafe_allow_html=True)
    status_placeholder.markdown("<div class='status-pill status-idle'>Monitor Ready</div>", unsafe_allow_html=True)
    fps_placeholder.empty()
    time_placeholder.empty()
    alert_placeholder.empty()

if DETECTOR is None:
    st.error("Model files not found. Upload 'shape_predictor_68_face_landmarks.dat'.", icon="❌")

# Render the HTML5 browser alarm audio if active
render_browser_alarm()

