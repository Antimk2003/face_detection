import streamlit as st
from PIL import Image
import numpy as np
import io
import os
import urllib.request
import threading
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── EMOTION CONFIG ───────────────────────────────────────────────────────────
EMOTION_NAMES  = ["anger", "content", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOTION_EMOJIS = {
    "anger": "😠", "content": "😌", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "neutral": "😐", "sad":  "😢", "surprise": "😲"
}
EMOTION_COLORS = {
    "anger":    "#FF4757", "content":  "#2ED573", "disgust":  "#A55EEA",
    "fear":     "#1E90FF", "happy":    "#FFD32A", "neutral":  "#747D8C",
    "sad":      "#70A1FF", "surprise": "#FF6B81"
}
EMOTION_DESC = {
    "anger":    "Strong displeasure detected",
    "content":  "Calm satisfaction detected",
    "disgust":  "Aversion response detected",
    "fear":     "Anxiety or dread detected",
    "happy":    "Joy and delight detected",
    "neutral":  "Balanced expression detected",
    "sad":      "Sorrow or grief detected",
    "surprise": "Astonishment detected"
}

# ─── MODEL LOAD ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
        from ultralytics import YOLO
        HF_URL = "https://huggingface.co/akser-123/face_detection/resolve/69bd802ad6e316ae1bd3b7ac4c14d3ba93cb5177/emotion_best.pt"
        local  = "emotion_best.pt"
        if not os.path.exists(local):
            urllib.request.urlretrieve(HF_URL, local)
        model = YOLO(local)
        return model, True
    except Exception as e:
        return None, str(e)

# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict(model, pil_img):
    arr = np.array(pil_img.convert("RGB"))
    res = model.predict(source=arr, imgsz=224, verbose=False)
    probs     = res[0].probs
    top1_idx  = int(probs.top1)
    top1_conf = float(probs.top1conf)
    all_probs = probs.data.cpu().numpy().tolist()
    return top1_idx, top1_conf, all_probs

# ─── LIVE VIDEO TRANSFORMER ───────────────────────────────────────────────────
class EmotionVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.lock  = threading.Lock()
        self.result = {"emotion": "neutral", "conf": 0.0, "emoji": "😐", "color": "#747D8C", "all_probs": []}
        self._frame_count = 0

    def set_model(self, model):
        with self.lock:
            self.model = model

    def get_result(self):
        with self.lock:
            return dict(self.result)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process every 5th frame to reduce CPU load
        self._frame_count += 1
        if self._frame_count % 5 == 0 and self.model is not None:
            try:
                # BGR → RGB → PIL
                rgb = img[:, :, ::-1]
                pil = Image.fromarray(rgb)
                top1_idx, top1_conf, all_probs = predict(self.model, pil)
                emotion = EMOTION_NAMES[top1_idx]
                emoji   = EMOTION_EMOJIS[emotion]
                color   = EMOTION_COLORS[emotion]
                with self.lock:
                    self.result = {
                        "emotion":   emotion,
                        "conf":      top1_conf,
                        "emoji":     emoji,
                        "color":     color,
                        "all_probs": all_probs
                    }
            except Exception:
                pass

        # Draw overlay on frame
        with self.lock:
            r = dict(self.result)

        try:
            import cv2
            emotion   = r["emotion"]
            conf      = r["conf"]
            emoji_map = {"anger":"ANGER","content":"CONTENT","disgust":"DISGUST",
                         "fear":"FEAR","happy":"HAPPY","neutral":"NEUTRAL",
                         "sad":"SAD","surprise":"SURPRISE"}
            emoji_vis = EMOTION_EMOJIS[emotion]

            # Hex color → BGR
            hex_c = r["color"].lstrip("#")
            rgb_c = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            bgr_c = (rgb_c[2], rgb_c[1], rgb_c[0])

            h, w = img.shape[:2]

            # Top banner
            cv2.rectangle(img, (0, 0), (w, 54), (10, 10, 20), -1)
            cv2.rectangle(img, (0, 0), (w, 54), bgr_c, 2)

            label = f"{emotion.upper()}  {conf*100:.1f}%"
            cv2.putText(img, label, (14, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, bgr_c, 2, cv2.LINE_AA)

            # Confidence bar at bottom
            bar_h = 8
            cv2.rectangle(img, (0, h - bar_h), (w, h), (30, 30, 30), -1)
            bar_w = int(w * conf)
            cv2.rectangle(img, (0, h - bar_h), (bar_w, h), bgr_c, -1)

            # Corner scan-lines effect
            corner = 30
            thick  = 2
            cv2.line(img, (0, 0), (corner, 0), bgr_c, thick)
            cv2.line(img, (0, 0), (0, corner), bgr_c, thick)
            cv2.line(img, (w, 0), (w - corner, 0), bgr_c, thick)
            cv2.line(img, (w, 0), (w, corner), bgr_c, thick)
            cv2.line(img, (0, h), (corner, h), bgr_c, thick)
            cv2.line(img, (0, h), (0, h - corner), bgr_c, thick)
            cv2.line(img, (w, h), (w - corner, h), bgr_c, thick)
            cv2.line(img, (w, h), (w, h - corner), bgr_c, thick)

        except ImportError:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg:        #08090E;
    --bg2:       #0D1117;
    --glass:     rgba(255,255,255,0.04);
    --glass2:    rgba(255,255,255,0.07);
    --border:    rgba(255,255,255,0.08);
    --border2:   rgba(255,255,255,0.14);
    --accent:    #7C3AED;
    --accent2:   #A78BFA;
    --text:      #F1F5F9;
    --muted:     #64748B;
    --muted2:    #94A3B8;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
.stApp {
    background:
        radial-gradient(ellipse 100% 50% at 50% -20%, rgba(124,58,237,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 100% 80%, rgba(167,139,250,0.06) 0%, transparent 50%),
        var(--bg) !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 2px; }

@keyframes fadeUp   { from { opacity:0; transform:translateY(24px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeIn   { from { opacity:0; } to { opacity:1; } }
@keyframes pulse    { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
@keyframes glow     { 0%,100% { box-shadow:0 0 20px rgba(124,58,237,0.3); } 50% { box-shadow:0 0 40px rgba(124,58,237,0.6); } }
@keyframes shimmer  { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
@keyframes borderPulse { 0%,100% { border-color: rgba(124,58,237,0.3); } 50% { border-color: rgba(167,139,250,0.7); } }
@keyframes barFill  { from { width:0%; } to { width:var(--w); } }
@keyframes countUp  { from { opacity:0; transform:scale(0.8); } to { opacity:1; transform:scale(1); } }
@keyframes dot1 { 0%,80%,100% { transform:scale(0); } 40% { transform:scale(1); } }
@keyframes dot2 { 0%,20%,100% { transform:scale(0); } 60% { transform:scale(1); } }
@keyframes dot3 { 0%,40%,100% { transform:scale(0); } 80% { transform:scale(1); } }
@keyframes livePulse { 0%,100% { opacity:1; box-shadow:0 0 0 0 rgba(255,71,87,0.4); } 70% { opacity:0.7; box-shadow:0 0 0 8px rgba(255,71,87,0); } }

.hero {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    animation: fadeUp 0.8s ease both;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(124,58,237,0.12);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent2);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-dot {
    width: 6px; height: 6px;
    background: var(--accent2);
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.8rem, 6vw, 5rem) !important;
    font-weight: 800 !important;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #fff 0%, var(--accent2) 50%, #C084FC 100%);
    background-size: 200% auto;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    animation: shimmer 4s linear infinite;
    line-height: 1 !important;
    margin-bottom: 0.8rem !important;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--muted2);
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto 1.5rem;
    line-height: 1.7;
}
.hero-stats {
    display: inline-flex;
    gap: 2rem;
    padding: 0.8rem 2rem;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 100px;
    backdrop-filter: blur(20px);
}
.hstat { text-align: center; }
.hstat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent2);
}
.hstat-label {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border2), var(--accent), var(--border2), transparent);
    margin: 1.5rem 0;
    animation: fadeIn 1s ease 0.5s both;
}
.card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.4rem;
    backdrop-filter: blur(20px);
    margin-bottom: 1rem;
    animation: fadeUp 0.5s ease both;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover { border-color: var(--border2); }
.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent2);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
}
.result-card {
    border-radius: 24px;
    padding: 2rem;
    background: var(--glass2);
    border: 1px solid var(--border2);
    backdrop-filter: blur(30px);
    animation: fadeUp 0.5s ease both, glow 3s ease infinite;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--emotion-color, #7C3AED), transparent);
}
.result-emoji {
    font-size: 5rem;
    line-height: 1;
    margin-bottom: 0.5rem;
    animation: countUp 0.4s ease both;
    display: block;
}
.result-emotion {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
    animation: countUp 0.4s ease 0.1s both;
}
.result-desc {
    font-size: 0.85rem;
    color: var(--muted2);
    margin-bottom: 1.2rem;
    animation: fadeIn 0.4s ease 0.2s both;
}
.confidence-big {
    font-family: 'Space Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    animation: countUp 0.5s ease 0.15s both;
}
.conf-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.2rem;
}
.conf-bar-wrap {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
    margin: 0.8rem 0 0.3rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 3px;
    animation: barFill 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    animation: fadeIn 0.4s ease both;
}
.prob-row:last-child { border-bottom: none; }
.prob-emoji { font-size: 1rem; width: 1.4rem; text-align: center; flex-shrink: 0; }
.prob-name  { font-size: 0.8rem; color: var(--muted2); width: 70px; flex-shrink: 0; font-weight: 500; }
.prob-bar-bg { flex: 1; height: 5px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
.prob-bar-fg { height: 100%; border-radius: 3px; transition: width 0.8s cubic-bezier(0.16,1,0.3,1); }
.prob-pct { font-family: 'Space Mono', monospace; font-size: 0.75rem; width: 42px; text-align: right; flex-shrink: 0; color: var(--muted2); }
.prob-pct.top { color: var(--text); font-weight: 700; }
.stat-row { display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.83rem; }
.stat-row:last-child { border-bottom: none; }
.sk { color: var(--muted); }
.sv { font-family: 'Space Mono', monospace; color: var(--text); font-size: 0.8rem; }
.loading-dots { display: flex; gap: 6px; align-items: center; justify-content: center; }
.loading-dot { width: 8px; height: 8px; background: var(--accent2); border-radius: 50%; }
.loading-dot:nth-child(1) { animation: dot1 1.2s infinite ease-in-out; }
.loading-dot:nth-child(2) { animation: dot2 1.2s infinite ease-in-out; }
.loading-dot:nth-child(3) { animation: dot3 1.2s infinite ease-in-out; }

/* Live badge */
.live-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(255,71,87,0.12);
    border: 1px solid rgba(255,71,87,0.3);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #FF4757;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.live-dot {
    width: 7px; height: 7px;
    background: #FF4757;
    border-radius: 50%;
    animation: livePulse 1.2s infinite;
}

/* Live result display */
.live-result-panel {
    border-radius: 20px;
    padding: 1.5rem;
    background: var(--glass2);
    border: 1px solid var(--border2);
    backdrop-filter: blur(30px);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    min-height: 140px;
}
.live-result-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--live-color, #7C3AED), transparent);
}

/* Streamlit overrides */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(124,58,237,0.04) !important;
    border: 2px dashed rgba(124,58,237,0.3) !important;
    border-radius: 16px !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #9333EA) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important; }
div[data-testid="stImage"] img { border-radius: 16px !important; border: 1px solid var(--border) !important; }
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
label { color: var(--muted2) !important; font-size: 0.85rem !important; }
div[data-testid="stCameraInput"] > div { border-radius: 16px !important; border: 1px solid var(--border) !important; background: var(--glass) !important; }
.stAlert { border-radius: 12px !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <div class="hero-dot"></div>
        Real-Time Emotion Intelligence
    </div>
    <div class="hero-title">EmotiSense AI</div>
    <p class="hero-sub">
        Upload image, capture photo, or use <b>live video</b> for continuous emotion detection
        powered by YOLOv8 deep learning.
    </p>
    <div class="hero-stats">
        <div class="hstat"><div class="hstat-val">69%</div><div class="hstat-label">Top-1 Acc</div></div>
        <div class="hstat"><div class="hstat-val">98.6%</div><div class="hstat-label">Top-5 Acc</div></div>
        <div class="hstat"><div class="hstat-val">8</div><div class="hstat-label">Emotions</div></div>
        <div class="hstat"><div class="hstat-val">LIVE</div><div class="hstat-label">Video</div></div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [("model_ready", False), ("model_obj", None), ("mode", "Upload Image"),
             ("transformer", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── MODEL LOAD ───────────────────────────────────────────────────────────────
if not st.session_state.model_ready:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; margin-bottom:0.5rem;">
                Initialize Model
            </div>
            <div style="color:var(--muted2); font-size:0.9rem; margin-bottom:2rem; line-height:1.7;">
                YOLOv8 emotion model will be loaded from Hugging Face.<br>
                First load takes ~30 seconds.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Load EmotiSense Model", use_container_width=True):
            with st.spinner("Loading model..."):
                model, status = load_model()
            if status is True:
                st.session_state.model_ready = True
                st.session_state.model_obj   = model
                st.success("✅ Model ready! Start detecting emotions.")
                st.rerun()
            else:
                st.error(f"❌ {status}")
    st.stop()

model = st.session_state.model_obj

# ─── MODE SELECTOR ────────────────────────────────────────────────────────────
col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
with col_nav2:
    mode = st.radio(
        "mode", ["📁 Upload Image", "📷 Photo Capture", "🎥 Live Video"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    pil_image   = None
    run_predict = False

    # ── UPLOAD MODE ──
    if "Upload" in mode:
        st.markdown("""
        <div class="card">
            <div class="card-label">📁 &nbsp; Image Upload</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-bottom:0.3rem;">
                Select Face Image
            </div>
            <div style="font-size:0.82rem; color:var(--muted); margin-bottom:1rem;">
                JPG, JPEG, PNG — clear front-facing face works best
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            pil_image   = Image.open(uploaded).convert("RGB")
            run_predict = True
            st.image(pil_image, caption="Uploaded Image", use_container_width=True)
        st.markdown("""
        <div class="card" style="margin-top:0.8rem;">
            <div class="card-label">💡 &nbsp; Tips for Best Results</div>
            <div class="stat-row"><span class="sk">Face angle</span><span class="sv">Front-facing preferred</span></div>
            <div class="stat-row"><span class="sk">Lighting</span><span class="sv">Well-lit environment</span></div>
            <div class="stat-row"><span class="sk">Resolution</span><span class="sv">224×224px minimum</span></div>
            <div class="stat-row"><span class="sk">Content</span><span class="sv">One face per image</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── PHOTO CAPTURE MODE ──
    elif "Photo" in mode:
        st.markdown("""
        <div class="card">
            <div class="card-label">📷 &nbsp; Photo Capture</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-bottom:0.3rem;">
                Capture from Camera
            </div>
            <div style="font-size:0.82rem; color:var(--muted); margin-bottom:1rem;">
                Allow camera access → Click capture → Get instant result
            </div>
        </div>
        """, unsafe_allow_html=True)
        cam_img = st.camera_input("", label_visibility="collapsed")
        if cam_img:
            pil_image   = Image.open(cam_img).convert("RGB")
            run_predict = True
        st.markdown("""
        <div class="card" style="margin-top:0.8rem;">
            <div class="card-label">🎯 &nbsp; Webcam Tips</div>
            <div class="stat-row"><span class="sk">Distance</span><span class="sv">Arm's length from camera</span></div>
            <div class="stat-row"><span class="sk">Background</span><span class="sv">Plain background ideal</span></div>
            <div class="stat-row"><span class="sk">Expression</span><span class="sv">Hold expression while capturing</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── LIVE VIDEO MODE ──
    else:
        st.markdown("""
        <div class="card">
            <div class="card-label">🎥 &nbsp; Live Video Detection</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; margin-bottom:0.3rem;">
                Real-Time Camera Feed
            </div>
            <div style="font-size:0.82rem; color:var(--muted); margin-bottom:0.8rem;">
                Camera on hote hi automatic emotion detect hogi har frame pe
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Create transformer and inject model
        transformer = EmotionVideoTransformer()
        transformer.set_model(model)

        ctx = webrtc_streamer(
            key="emotion-live",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=lambda: transformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Store transformer reference in session state for result polling
        if ctx.video_transformer:
            st.session_state.transformer = ctx.video_transformer
            st.session_state.transformer.set_model(model)

        st.markdown("""
        <div class="card" style="margin-top:0.8rem;">
            <div class="card-label">⚡ &nbsp; Live Mode Info</div>
            <div class="stat-row"><span class="sk">Inference</span><span class="sv">Every 5th frame</span></div>
            <div class="stat-row"><span class="sk">Overlay</span><span class="sv">Emotion + confidence bar</span></div>
            <div class="stat-row"><span class="sk">Latency</span><span class="sv">~100–200ms</span></div>
            <div class="stat-row"><span class="sk">Results</span><span class="sv">Panel mein live update</span></div>
        </div>
        """, unsafe_allow_html=True)

# ─── RIGHT PANEL — RESULTS ───────────────────────────────────────────────────
with right:

    # ── Live Video results panel ──
    if "Live" in mode:
        st.markdown("""
        <div class="live-badge">
            <div class="live-dot"></div>
            Live Detecting
        </div>
        """, unsafe_allow_html=True)

        result_placeholder   = st.empty()
        probbar_placeholder  = st.empty()

        # Poll results if transformer is active
        tr = st.session_state.get("transformer")
        if tr is not None:
            r = tr.get_result()
            emotion  = r["emotion"]
            conf     = r["conf"]
            emoji    = r["emoji"]
            color    = r["color"]
            desc     = EMOTION_DESC[emotion]
            conf_pct = conf * 100
            all_probs = r.get("all_probs", [])

            with result_placeholder.container():
                st.markdown(f"""
                <div class="live-result-panel" style="--live-color:{color};">
                    <div style="display:flex; gap:1rem; align-items:center;">
                        <span style="font-size:4rem;">{emoji}</span>
                        <div style="flex:1;">
                            <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800;
                                 color:{color}; text-transform:uppercase; letter-spacing:-1px;">{emotion}</div>
                            <div style="font-size:0.82rem; color:var(--muted2);">{desc}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-family:'Space Mono',monospace; font-size:2.5rem;
                                 font-weight:700; color:{color}; line-height:1;">{conf_pct:.1f}%</div>
                            <div style="font-size:0.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:2px;">Confidence</div>
                        </div>
                    </div>
                    <div class="conf-bar-wrap" style="margin-top:1rem;">
                        <div class="conf-bar-fill" style="width:{conf_pct}%;
                             background:linear-gradient(90deg,{color}88,{color}); --w:{conf_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if all_probs:
                with probbar_placeholder.container():
                    st.markdown("""
                    <div class="card">
                        <div class="card-label">📊 &nbsp; All Emotion Probabilities</div>
                    """, unsafe_allow_html=True)
                    sorted_probs = sorted(enumerate(all_probs), key=lambda x: x[1], reverse=True)
                    bars_html = ""
                    for delay_i, (idx, prob) in enumerate(sorted_probs):
                        emo   = EMOTION_NAMES[idx]
                        emj   = EMOTION_EMOJIS[emo]
                        col   = EMOTION_COLORS[emo]
                        pct   = prob * 100
                        is_top = idx == list(EMOTION_NAMES).index(emotion) if emotion in EMOTION_NAMES else False
                        delay  = delay_i * 0.06
                        bars_html += f"""
                        <div class="prob-row" style="animation-delay:{delay}s;">
                            <span class="prob-emoji">{emj}</span>
                            <span class="prob-name" style="color:{'var(--text)' if is_top else 'var(--muted2)'}; font-weight:{'700' if is_top else '400'};">{emo}</span>
                            <div class="prob-bar-bg"><div class="prob-bar-fg" style="width:{pct:.1f}%; background:{col}{'cc' if not is_top else ''};"></div></div>
                            <span class="prob-pct {'top' if is_top else ''}">{pct:.1f}%</span>
                        </div>"""
                    st.markdown(bars_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="min-height:320px; display:flex; flex-direction:column;
                 align-items:center; justify-content:center; text-align:center; gap:1rem;">
                <div style="font-size:4rem; opacity:0.2;">🎥</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:var(--muted2);">
                    Camera Start Karo
                </div>
                <div style="font-size:0.82rem; color:var(--muted); max-width:220px; line-height:1.7;">
                    Left side mein START button dabao — camera on hote hi emotion detect hoga
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Auto-refresh for live updates
        import time
        time.sleep(0.5)
        if st.session_state.get("transformer"):
            st.rerun()

    # ── Static image results (Upload / Photo) ──
    else:
        if not run_predict or pil_image is None:
            st.markdown("""
            <div class="card" style="min-height:480px; display:flex; flex-direction:column;
                 align-items:center; justify-content:center; text-align:center; gap:1rem;">
                <div style="font-size:5rem; opacity:0.15;">🎭</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
                     color:var(--muted2);">Awaiting Input</div>
                <div style="font-size:0.83rem; color:var(--muted); max-width:220px; line-height:1.7;">
                    Upload an image or capture from webcam to detect emotions
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; justify-content:center; margin-top:0.5rem;">
                    <span style="padding:0.25rem 0.7rem; border-radius:50px; background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.2); font-size:0.75rem; color:var(--accent2);">😠 Anger</span>
                    <span style="padding:0.25rem 0.7rem; border-radius:50px; background:rgba(46,213,115,0.08); border:1px solid rgba(46,213,115,0.2); font-size:0.75rem; color:#2ED573;">😌 Content</span>
                    <span style="padding:0.25rem 0.7rem; border-radius:50px; background:rgba(255,214,42,0.08); border:1px solid rgba(255,214,42,0.2); font-size:0.75rem; color:#FFD32A;">😊 Happy</span>
                    <span style="padding:0.25rem 0.7rem; border-radius:50px; background:rgba(112,161,255,0.08); border:1px solid rgba(112,161,255,0.2); font-size:0.75rem; color:#70A1FF;">😢 Sad</span>
                    <span style="padding:0.25rem 0.7rem; border-radius:50px; background:rgba(30,144,255,0.08); border:1px solid rgba(30,144,255,0.2); font-size:0.75rem; color:#1E90FF;">😨 Fear</span>
                    <span style="padding:0.25rem 0.7rem; border-radius:50px; background:rgba(255,107,129,0.08); border:1px solid rgba(255,107,129,0.2); font-size:0.75rem; color:#FF6B81;">😲 Surprise</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner(""):
                st.markdown("""
                <div style="text-align:center; padding:1rem;">
                    <div class="loading-dots">
                        <div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>
                    </div>
                    <div style="font-size:0.82rem; color:var(--muted); margin-top:0.5rem;">Analyzing expression...</div>
                </div>
                """, unsafe_allow_html=True)
                top1_idx, top1_conf, all_probs = predict(model, pil_image)

            emotion  = EMOTION_NAMES[top1_idx]
            emoji    = EMOTION_EMOJIS[emotion]
            color    = EMOTION_COLORS[emotion]
            desc     = EMOTION_DESC[emotion]
            conf_pct = top1_conf * 100

            st.markdown(f"""
            <div class="result-card" style="--emotion-color:{color};">
                <div style="display:flex; gap:1.5rem; align-items:flex-start;">
                    <div style="flex:1;">
                        <span class="result-emoji">{emoji}</span>
                        <div class="result-emotion" style="color:{color};">{emotion}</div>
                        <div class="result-desc">{desc}</div>
                    </div>
                    <div style="text-align:right;">
                        <div class="confidence-big" style="color:{color};">{conf_pct:.1f}%</div>
                        <div class="conf-label">Confidence</div>
                    </div>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill" style="width:{conf_pct}%; background:linear-gradient(90deg,{color}88,{color}); --w:{conf_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            st.markdown("""<div class="card"><div class="card-label">📊 &nbsp; All Emotion Probabilities</div>""", unsafe_allow_html=True)
            sorted_probs = sorted(enumerate(all_probs), key=lambda x: x[1], reverse=True)
            bars_html = ""
            for delay_i, (idx, prob) in enumerate(sorted_probs):
                emo   = EMOTION_NAMES[idx]
                emj   = EMOTION_EMOJIS[emo]
                col   = EMOTION_COLORS[emo]
                pct   = prob * 100
                is_top = idx == top1_idx
                delay  = delay_i * 0.06
                bars_html += f"""
                <div class="prob-row" style="animation-delay:{delay}s;">
                    <span class="prob-emoji">{emj}</span>
                    <span class="prob-name" style="color:{'var(--text)' if is_top else 'var(--muted2)'}; font-weight:{'700' if is_top else '400'};">{emo}</span>
                    <div class="prob-bar-bg"><div class="prob-bar-fg" style="width:{pct:.1f}%; background:{col}{'cc' if not is_top else ''};"></div></div>
                    <span class="prob-pct {'top' if is_top else ''}">{pct:.1f}%</span>
                </div>"""
            st.markdown(bars_html + "</div>", unsafe_allow_html=True)

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            st.download_button("⬇️ Download Image", data=buf.getvalue(),
                               file_name=f"emotion_{emotion}_{int(conf_pct)}.png",
                               mime="image/png", use_container_width=True)

# ─── MODEL INFO ───────────────────────────────────────────────────────────────
st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin-bottom:1.2rem;">
    <div style="font-family:'Space Mono',monospace; font-size:0.68rem; color:var(--accent2);
         text-transform:uppercase; letter-spacing:3px; margin-bottom:0.4rem;">Model Intelligence</div>
    <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;">Architecture & Training Details</div>
</div>
""", unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3, gap="medium")
with mc1:
    st.markdown("""<div class="card"><div class="card-label">🧠 &nbsp; Architecture</div>
        <div class="stat-row"><span class="sk">Model</span><span class="sv">YOLOv8m-cls</span></div>
        <div class="stat-row"><span class="sk">Backbone</span><span class="sv">CSPDarknet53</span></div>
        <div class="stat-row"><span class="sk">Parameters</span><span class="sv">15.7M</span></div>
        <div class="stat-row"><span class="sk">Output</span><span class="sv">Softmax × 8</span></div>
    </div>""", unsafe_allow_html=True)
with mc2:
    st.markdown("""<div class="card"><div class="card-label">⚙️ &nbsp; Training Config</div>
        <div class="stat-row"><span class="sk">Epochs</span><span class="sv">34 best / 50</span></div>
        <div class="stat-row"><span class="sk">Optimizer</span><span class="sv">Adam</span></div>
        <div class="stat-row"><span class="sk">Batch Size</span><span class="sv">32</span></div>
        <div class="stat-row"><span class="sk">Image Size</span><span class="sv">224 × 224</span></div>
    </div>""", unsafe_allow_html=True)
with mc3:
    st.markdown("""<div class="card"><div class="card-label">📈 &nbsp; Performance</div>
        <div class="stat-row"><span class="sk">Top-1 Accuracy</span><span class="sv">69.3%</span></div>
        <div class="stat-row"><span class="sk">Top-5 Accuracy</span><span class="sv">98.6%</span></div>
        <div class="stat-row"><span class="sk">Best Class</span><span class="sv">Happy (82%)</span></div>
        <div class="stat-row"><span class="sk">Inference</span><span class="sv">~0.9ms/img</span></div>
    </div>""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 1rem 1.5rem; margin-top:1rem; border-top:1px solid var(--border);">
    <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
         background:linear-gradient(135deg,#fff,var(--accent2)); -webkit-background-clip:text;
         -webkit-text-fill-color:transparent; background-clip:text; margin-bottom:0.3rem;">EmotiSense AI</div>
    <div style="color:var(--muted); font-size:0.82rem; margin-bottom:0.5rem;">
        Face Emotion Detection · YOLOv8 Deep Learning · 8 Emotion Classes · Live Video
    </div>
    <div style="color:var(--muted); font-size:0.72rem;">
        ⚠️ For research & educational purposes only — not a clinical tool
    </div>
</div>
""", unsafe_allow_html=True)
