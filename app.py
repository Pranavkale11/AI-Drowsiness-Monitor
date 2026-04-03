
# ==============================================================================
#  AI-BASED DRIVER DROWSINESS & ALERT MONITORING SYSTEM (STABLE VERSION)
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
from imutils import face_utils
from collections import deque

# ── Dependency Check ──────────────────────────────────────────────────────────
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG & SESSION STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="DrowseGuard AI | Stable v3.1",
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PREMIUM STYLING (CSS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

.stApp { background: radial-gradient(circle at 10% 20%, #111827 0%, #0f172a 100%); color: #f1f5f9; font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background-color: #0b1120; border-right: 1px solid #1e293b; }
#MainMenu, footer, header { visibility: hidden; }

.glass-card { background: rgba(30, 41, 59, 0.6); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 20px; margin-bottom: 15px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); }
.metric-label { font-size: 0.7rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; }
.metric-value { font-size: 1.8rem; font-weight: 900; color: #3b82f6; margin-top: 4px; }
.status-pill { padding: 10px 20px; border-radius: 99px; font-weight: 800; text-align: center; font-size: 0.9rem; display: inline-block; border: 1px solid transparent; }
.status-active   { background: rgba(16, 185, 129, 0.1); color: #34d399; border-color: rgba(16, 185, 129, 0.3); }
.status-drowsy   { background: rgba(245, 158, 11, 0.1); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3); }
.status-sleeping { background: rgba(239, 68, 68, 0.2); color: #f87171; border-color: rgba(239, 68, 68, 0.5); animation: pulse-red 1.5s infinite; }
.status-idle { background: rgba(148, 163, 184, 0.1); color: #cbd5e1; border-color: rgba(148, 163, 184, 0.3); }
@keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
.fps-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; background: #1e293b; padding: 4px 10px; border-radius: 4px; color: #10b981; }
.alert-bar { background: linear-gradient(90deg, #7f1d1d, #ef4444, #7f1d1d); color: white; padding: 12px; border-radius: 8px; text-align: center; font-weight: 900; letter-spacing: 2px; font-size: 1.2rem; margin-bottom: 20px; }
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
# reruns (which reset plain module-level globals and silently kill threads).

class _AlertEngine:
    """Thread-safe, non-blocking continuous sound alert engine.

    Uses threading.Event for instant stop signalling so winsound.Beep()
    never blocks the stop path (avoids the raw-bool + lock deadlock pattern).
    """
    def __init__(self):
        # Event: set   → worker should keep playing
        #        clear → worker should stop and exit
        self._play_event = threading.Event()
        self._mode_lock  = threading.Lock()   # Guards _mode string only
        self._mode       = "sleeping"          # Current tone pattern
        self._thread: threading.Thread | None = None
        print("[AlertEngine] Initialised — winsound available:", WINSOUND_AVAILABLE)

    # ------------------------------------------------------------------ #
    def _worker(self):
        """Background loop: plays loud 2000 Hz beep continuously while active."""
        print("[Alarm] Worker started")
        while self._play_event.is_set():
            if not WINSOUND_AVAILABLE:
                print("[Alarm] winsound unavailable — waiting")
                self._play_event.wait(timeout=0.5)
                continue
            try:
                winsound.Beep(2000, 400)   # 2000 Hz, 400 ms — loud & piercing
                if not self._play_event.is_set():
                    break
                time.sleep(0.1)            # 100 ms gap between bursts
            except Exception as exc:
                print(f"[Alarm] Beep error: {exc}")
                time.sleep(0.3)
        print("[Alarm] Worker stopped")

    # ------------------------------------------------------------------ #

    def start(self, mode: str):
        """Activate alert in the given mode ('drowsy' or 'sleeping').

        Safe to call on every frame — only spawns a new thread when needed.
        Mode switches are handled live inside the running worker.
        """
        with self._mode_lock:
            self._mode = mode

        if self._play_event.is_set():
            # Worker already running — mode update above is enough
            print(f"[AlertEngine] Mode updated to '{mode}' (thread already live)")
            return

        # Start a fresh worker thread
        self._play_event.set()
        print(f"[AlertEngine] Starting alert thread — mode: {mode}")

        # Wait for previous thread to finish (should be instant after clear)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        self._thread = threading.Thread(
            target=self._worker, name="AlertWorker", daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop the alert immediately. Safe to call when nothing is playing."""
        if self._play_event.is_set():
            print("[AlertEngine] Stopping alert")
            self._play_event.clear()    # Worker exits after current Beep finishes

    def is_playing(self):
        """Returns True if the alert is currently active."""
        return self._play_event.is_set()


@st.cache_resource(show_spinner=False)
def _get_alarm() -> _AlertEngine:
    """Singleton engine — persists across Streamlit reruns."""
    return _AlertEngine()

# Public API — no mode arg; single loud 2000 Hz alarm only
def start_alert():
    _get_alarm().start("sleeping")   # mode arg kept for engine compat

def stop_alert():
    _get_alarm().stop()

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return detector, predictor
    except: return None, None

DETECTOR, PREDICTOR = load_models()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR COMMANDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("<h2 style='color:#3b82f6;'>⚙️ DrowseGuard Stable</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Persistent Master Toggle
    monitor_on = st.toggle("Start Monitoring System", value=st.session_state.run)
    
    # Handle state transition
    if monitor_on != st.session_state.run:
        st.session_state.run = monitor_on
        if monitor_on:
            st.session_state.start_time = time.time()
            st.session_state.blink_count = 0
            st.session_state.ear_hist.clear()
            st.session_state.time_hist.clear()
        st.rerun()

    st.markdown("---")
    
    # Performance Mode
    perf_mode = st.select_slider("Performance Mode", options=["Accuracy", "Balanced", "Speed"], value="Balanced")
    FRAME_SKIP = {"Accuracy": 1, "Balanced": 2, "Speed": 4}[perf_mode]
    
    st.markdown("#### 👁️ Sensitivity")
    ear_threshold = st.slider("EAR Threshold", 0.15, 0.35, 0.25, 0.01)
    consec_frames = st.slider("Trigger Buffer (frames)", 3, 20, 5)  # Lower default = faster trigger
    
    st.markdown("#### 🎨 UI Options")
    draw_landmarks = st.toggle("Show Landmarks", value=False)
    draw_box = st.toggle("Show Box", value=True)
    live_graph_active = st.toggle("Live Analytics", value=True)

    st.markdown("#### 🔔 Alert Settings")
    alarm_enabled = st.toggle(
        "Enable Alarm",
        value=True,
        help="Play audio tones when drowsiness or sleeping is detected (Windows only)"
    )
    if alarm_enabled and not WINSOUND_AVAILABLE:
        st.warning("winsound not available on this platform — alarm disabled.", icon="🔇")

    # One-click sound test — bypasses all detection logic
    if st.button("🔊 Test Alarm Sound", help="Plays 3 beeps immediately to verify your audio is working"):
        if WINSOUND_AVAILABLE:
            def _test_beep():
                import winsound as _ws, time as _t
                _ws.Beep(2000, 300)
                _t.sleep(0.1)
                _ws.Beep(2000, 300)
                _t.sleep(0.1)
                _ws.Beep(2000, 300)
            threading.Thread(target=_test_beep, daemon=True).start()
            st.toast("🔊 Beep sent! Did you hear it?", icon="✅")
        else:
            st.error("winsound is not available on this system.")

    # Live debug counters — visible while monitoring is active
    st.markdown("#### 🛠️ Live Debug")
    debug_placeholder = st.empty()

    st.markdown("---")
    st.caption("v3.1 Stable Release | OpenCV + dlib")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN UI LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
h_col1, h_col2, h_col3 = st.columns([2.5, 1, 1])
with h_col1:
    st.markdown("<h1 style='margin-bottom:0;'>AI Driver Monitor</h1>", unsafe_allow_html=True)
    status_placeholder = st.empty()
with h_col2:
    fps_placeholder = st.empty()
with h_col3:
    time_placeholder = st.empty()

st.markdown("<br>", unsafe_allow_html=True)
col_v, col_a = st.columns([1.7, 1], gap="medium")

with col_v:
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
with col_a:
    st.markdown("<div class='glass-card'><div class='metric-label'>Fatigue Probability</div>", unsafe_allow_html=True)
    fatigue_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown("<div class='glass-card'><div class='metric-label'>Blinks</div>", unsafe_allow_html=True)
        blink_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    with s_col2:
        st.markdown("<div class='glass-card'><div class='metric-label'>EAR</div>", unsafe_allow_html=True)
        ear_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'><div class='metric-label'>EAR History</div>", unsafe_allow_html=True)
    graph_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STABLE EXECUTION LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state.run:
    # Use context manager for safe camera handling
    with ThreadedCamera(src=0, width=640, height=480) as cam:
        perf = PerformanceTracker()
        f_idx = 0
        sleep_ctr = 0
        drowsy_ctr = 0
        was_closed = False
        last_ui = 0
        last_graph = 0
        
        while st.session_state.run:
            ret, frame = cam.read()
            if not ret or frame is None:
                st.error("Camera connection failed.")
                st.session_state.run = False
                break
            
            perf.tick()
            f_idx += 1
            
            # Logic Processing
            if f_idx % FRAME_SKIP == 0:
                # Optimized sub-sampling
                small = cv2.resize(frame, (320, 240))
                gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = DETECTOR(gray_small, 0)
                
                if faces:
                    face = faces[0]
                    # Scale back to original
                    r = dlib.rectangle(face.left()*2, face.top()*2, face.right()*2, face.bottom()*2)
                    
                    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    shape = PREDICTOR(gray_full, r)
                    shape = face_utils.shape_to_np(shape)
                    
                    # EAR Calculation
                    left_eye = shape[36:42]
                    right_eye = shape[42:48]
                    
                    def eye_ear(pts):
                        A = np.linalg.norm(pts[1]-pts[5])
                        B = np.linalg.norm(pts[2]-pts[4])
                        C = np.linalg.norm(pts[0]-pts[3])
                        return (A+B)/(2.0*C)
                    
                    cur_ear = (eye_ear(left_eye) + eye_ear(right_eye)) / 2.0
                    
                    # Blink & State
                    is_closed = cur_ear < ear_threshold
                    if is_closed and not was_closed:
                        st.session_state.blink_count += 1
                    was_closed = is_closed
                    
                    # ── Counter logic ─────────────────────────────────────
                    # IMPORTANT: drowsy is NOT the opposite of sleeping.
                    # When eyes are closed (is_closed), ONLY sleep_ctr grows.
                    # When eyes are in the partial-close band, ONLY drowsy_ctr grows.
                    # Neither branch resets the other's counter prematurely.
                    if is_closed:
                        sleep_ctr += 1
                        # Do NOT touch drowsy_ctr here — let it hold its value
                    elif cur_ear < ear_threshold + 0.04:
                        drowsy_ctr += 1
                        # Do NOT reset sleep_ctr here — driver may re-close eyes next frame
                    else:
                        # Eyes clearly open — reset both counters and stop alert
                        sleep_ctr = 0
                        drowsy_ctr = 0
                    
                    # ── Update status & drive alarm ──────────────────────────
                    if sleep_ctr > consec_frames:
                        # Eyes closed long enough — SLEEPING
                        st.session_state.status = "Sleeping"
                        print(f"[ALARM] Sleeping — sleep_ctr={sleep_ctr}")
                        if alarm_enabled:
                            start_alert()   # Loud continuous 2000 Hz alarm
                        else:
                            stop_alert()
                    elif drowsy_ctr > consec_frames:
                        # Eyes partially closed — DROWSY (no alarm, just visual)
                        st.session_state.status = "Drowsy"
                        stop_alert()        # No sound for drowsy state
                    else:
                        # Eyes open — ACTIVE
                        st.session_state.status = "Active"
                        stop_alert()        # Silence alarm immediately
                        
                    st.session_state.ear_hist.append(cur_ear)
                    st.session_state.time_hist.append(time.time() - st.session_state.start_time)
                    
                    # Drawing
                    if draw_box:
                        c = (0,0,255) if is_closed else (16,185,129)
                        cv2.rectangle(frame, (r.left(), r.top()), (r.right(), r.bottom()), c, 2)
                    if draw_landmarks:
                        for (x,y) in shape: cv2.circle(frame, (x,y), 1, (255,255,255), -1)
                else:
                    st.session_state.status = "No Face"
                    sleep_ctr = 0
                    drowsy_ctr = 0
                    stop_alert()  # No face detected — stop alarm

            # ── Live debug counter update in sidebar ──────────────────────
            # Safe read: ear_hist may be empty on startup frames before any
            # face has been detected, so always guard with a fallback.
            _ear_display = f"{st.session_state.ear_hist[-1]:.3f}" if st.session_state.ear_hist else "--"
            alarm = _get_alarm()
            alert_state = "🔴 PLAYING" if alarm.is_playing() else "⚫ OFF"
            debug_placeholder.markdown(
                f"""
                | Counter | Value |
                |---|---|
                | `sleep_ctr` | **{sleep_ctr}** / {consec_frames} |
                | `drowsy_ctr` | **{drowsy_ctr}** / {consec_frames} |
                | `EAR` | **{_ear_display}** |
                | `Alarm` | {alert_state} |
                """,
                unsafe_allow_html=False
            )

            # UI Update Throttling (~15 FPS)
            if time.time() - last_ui > 0.06:
                # Video
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb, channels="RGB", use_container_width=True)
                
                # Logic Stats
                st_text = st.session_state.status
                s_pill = st_text.lower().replace(" ", "-")
                status_placeholder.markdown(f"<div class='status-pill status-{s_pill}'>{st_text}</div>", unsafe_allow_html=True)
                
                fps_placeholder.markdown(f"<div class='metric-label'>System Performance</div><div class='fps-badge'>{perf.get_fps():.1f} FPS</div>", unsafe_allow_html=True)
                
                elap = int(time.time() - st.session_state.start_time)
                time_placeholder.markdown(f"<div class='metric-label'>Elapsed Time</div><div class='metric-value' style='font-size:1rem;'>{elap//60:02d}:{elap%60:02d}s</div>", unsafe_allow_html=True)
                
                # Fatigue
                ear_now = st.session_state.ear_hist[-1] if st.session_state.ear_hist else 0.0
                fatigue = min(100, max(0, (ear_threshold-ear_now)/ear_threshold*40 + (sleep_ctr*5)))
                f_color = "#ef4444" if fatigue > 70 else ("#fbbf24" if fatigue > 35 else "#10b981")
                fatigue_placeholder.markdown(f"""
                    <div style='font-size:2rem; font-weight:900; color:{f_color};'>{fatigue:.1f}%</div>
                    <div style='width:100%; height:8px; background:#1e293b; border-radius:10px;'>
                        <div style='width:{fatigue}%; height:100%; background:{f_color}; border-radius:10px;'></div>
                    </div>
                """, unsafe_allow_html=True)
                
                blink_placeholder.markdown(f"<div class='metric-value'>{st.session_state.blink_count}</div>", unsafe_allow_html=True)
                ear_placeholder.markdown(f"<div class='metric-value'>{ear_now:.3f}</div>", unsafe_allow_html=True)
                
                if st_text == "Sleeping":
                    alert_placeholder.markdown("<div class='alert-bar'>⚠️ WAKE UP DRIVER! ⚠️</div>", unsafe_allow_html=True)
                else:
                    alert_placeholder.empty()
                
                # Analytics Graph Refresh (Throttled to every 10 frames)
                if live_graph_active and PLOTLY_AVAILABLE and f_idx % 10 == 0:
                    if len(st.session_state.ear_hist) > 5:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=list(st.session_state.time_hist), y=list(st.session_state.ear_hist), mode='lines', line=dict(color='#3b82f6', width=3), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'))
                        fig.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(range=[0.1, 0.4]), showlegend=False)
                        
                        # Use dynamic unique key to bypass StreamlitDuplicateElementId issues
                        # This ensures each render is treated as a unique element in the frontend.
                        graph_placeholder.empty()
                        graph_placeholder.plotly_chart(
                            fig, 
                            use_container_width=True, 
                            config={'displayModeBar': False},
                            key=f"ear_analytics_{f_idx}" 
                        )
                
                last_ui = time.time()

else:
    # Standby UI
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
    st.error("Model files not found. Upload 'shape_predictor_68_face_landmarks.dat'.")
