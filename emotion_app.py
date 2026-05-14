import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import io
import time
import os
import urllib.request
import cv2

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

# ─── FACE DETECTOR LOAD ───────────────────────────────────────────────────────
@st.cache_resource
def load_face_detector():
    """Load OpenCV Haar cascade face detector (built into opencv-python, no download needed)."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return None
    return detector

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

# ─── FACE CROP ────────────────────────────────────────────────────────────────
def crop_face(pil_img, detector, padding=0.30):
    """
    Detect the largest face in the image, pad it, and return a 224x224 crop.
    Returns (face_pil, bbox) or (None, None) if no face is found.
    """
    # Fix EXIF rotation — critical for webcam/phone images
    pil_img = ImageOps.exif_transpose(pil_img)

    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Enhance contrast for better detection in low-light
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        # Second pass: slightly more lenient (fewer neighbors)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    if len(faces) == 0:
        return None, None

    # Pick the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Add padding so ears, chin, forehead are included
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    img_w, img_h = pil_img.size

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    face_crop = pil_img.crop((x1, y1, x2, y2))

    # Pad to square to avoid distortion before resize
    cw, ch = face_crop.size
    if cw != ch:
        size = max(cw, ch)
        padded = Image.new("RGB", (size, size), (128, 128, 128))
        padded.paste(face_crop, ((size - cw) // 2, (size - ch) // 2))
        face_crop = padded

    face_224 = face_crop.resize((224, 224), Image.LANCZOS)
    return face_224, (x1, y1, x2, y2)

# ─── PREDICT ─────────────────────────────────────────────────────────────────
def predict(model, pil_img, detector):
    """
    Full pipeline: fix rotation → detect face → crop → classify.
    Returns (top1_idx, top1_conf, all_probs, face_crop, bbox)
    or      (None, None, None, None, None) if no face detected.
    """
    face_crop, bbox = crop_face(pil_img, detector)

    if face_crop is None:
        return None, None, None, None, None

    arr = np.array(face_crop)
    res = model.predict(source=arr, imgsz=224, verbose=False)
    probs     = res[0].probs
    top1_idx  = int(probs.top1)
    top1_conf = float(probs.top1conf)
    all_probs = probs.data.cpu().numpy().tolist()
    return top1_idx, top1_conf, all_probs, face_crop, bbox

# ─── DRAW BBOX ON ORIGINAL IMAGE ─────────────────────────────────────────────
def draw_face_box(pil_img, bbox, emotion, color):
    """Draw a colored bounding box and label on the original image."""
    img_copy = ImageOps.exif_transpose(pil_img).copy()
    draw = ImageDraw.Draw(img_copy)
    x1, y1, x2, y2 = bbox

    # Draw rounded-ish box (3px thick)
    for offset in range(3):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline=color
        )

    # Label background
    label = f" {EMOTION_EMOJIS[emotion]} {emotion.upper()} "
    label_y = max(0, y1 - 28)
    draw.rectangle([x1, label_y, x1 + len(label) * 9, y1], fill=color)

    # Try to use a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((x1 + 4, label_y + 4), label, fill="white", font=font)
    return img_copy

# ─── CSS ──────────────────────────────────────────────────────────────────────
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
@keyframes barFill  { from { width:0%; } to { width:var(--w); } }
@keyframes countUp  { from { opacity:0; transform:scale(0.8); } to { opacity:1; transform:scale(1); } }
@keyframes shimmer  { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
@keyframes borderPulse { 0%,100% { border-color: rgba(124,58,237,0.3); } 50% { border-color: rgba(167,139,250,0.7); } }
@keyframes dot1 { 0%,80%,100% { transform:scale(0); } 40% { transform:scale(1); } }
@keyframes dot2 { 0%,20%,100% { transform:scale(0); } 60% { transform:scale(1); } }
@keyframes dot3 { 0%,40%,100% { transform:scale(0); } 80% { transform:scale(1); } }

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
.hero-dot { width: 6px; height: 6px; background: var(--accent2); border-radius: 50%; animation: pulse 1.5s infinite; }
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
.hero-sub { font-size: 1.05rem; color: var(--muted2); font-weight: 300; max-width: 500px; margin: 0 auto 1.5rem; line-height: 1.7; }
.hero-stats { display: inline-flex; gap: 2rem; padding: 0.8rem 2rem; background: var(--glass); border: 1px solid var(--border); border-radius: 100px; backdrop-filter: blur(20px); }
.hstat { text-align: center; }
.hstat-val { font-family: 'Space Mono', monospace; font-size: 1rem; font-weight: 700; color: var(--accent2); }
.hstat-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

.divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border2), var(--accent), var(--border2), transparent); margin: 1.5rem 0; animation: fadeIn 1s ease 0.5s both; }

.card { background: var(--glass); border: 1px solid var(--border); border-radius: 20px; padding: 1.4rem; backdrop-filter: blur(20px); margin-bottom: 1rem; animation: fadeUp 0.5s ease both; transition: border-color 0.3s ease; }
.card:hover { border-color: var(--border2); }
.card-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--accent2); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.8rem; }

.result-card {
    border-radius: 24px; padding: 2rem;
    background: var(--glass2); border: 1px solid var(--border2);
    backdrop-filter: blur(30px);
    animation: fadeUp 0.5s ease both, glow 3s ease infinite;
    position: relative; overflow: hidden;
}
.result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--emotion-color, #7C3AED), transparent); }
.result-emoji { font-size: 5rem; line-height: 1; margin-bottom: 0.5rem; animation: countUp 0.4s ease both; display: block; }
.result-emotion { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; text-transform: uppercase; letter-spacing: -1px; margin-bottom: 0.3rem; animation: countUp 0.4s ease 0.1s both; }
.result-desc { font-size: 0.85rem; color: var(--muted2); margin-bottom: 1.2rem; animation: fadeIn 0.4s ease 0.2s both; }
.confidence-big { font-family: 'Space Mono', monospace; font-size: 3.5rem; font-weight: 700; line-height: 1; animation: countUp 0.5s ease 0.15s both; }
.conf-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; margin-top: 0.2rem; }
.conf-bar-wrap { height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; margin: 0.8rem 0 0.3rem; }
.conf-bar-fill { height: 100%; border-radius: 3px; animation: barFill 0.8s cubic-bezier(0.16,1,0.3,1) both; }

.prob-row { display: flex; align-items: center; gap: 0.7rem; padding: 0.45rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); animation: fadeIn 0.4s ease both; }
.prob-row:last-child { border-bottom: none; }
.prob-emoji { font-size: 1rem; width: 1.4rem; text-align: center; flex-shrink: 0; }
.prob-name  { font-size: 0.8rem; color: var(--muted2); width: 70px; flex-shrink: 0; font-weight: 500; }
.prob-bar-bg { flex: 1; height: 5px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
.prob-bar-fg { height: 100%; border-radius: 3px; transition: width 0.8s cubic-bezier(0.16,1,0.3,1); }
.prob-pct { font-family: 'Space Mono', monospace; font-size: 0.75rem; width: 42px; text-align: right; flex-shrink: 0; color: var(--muted2); }
.prob-pct.top { color: var(--text); font-weight: 700; }

.no-face-card {
    border-radius: 20px; padding: 2rem;
    background: rgba(255, 71, 87, 0.08);
    border: 1px solid rgba(255, 71, 87, 0.25);
    text-align: center;
    animation: fadeUp 0.5s ease both;
}

.loading-dots { display: flex; gap: 6px; align-items: center; justify-content: center; }
.loading-dot { width: 8px; height: 8px; background: var(--accent2); border-radius: 50%; }
.loading-dot:nth-child(1) { animation: dot1 1.2s infinite ease-in-out; }
.loading-dot:nth-child(2) { animation: dot2 1.2s infinite ease-in-out; }
.loading-dot:nth-child(3) { animation: dot3 1.2s infinite ease-in-out; }

.stat-row { display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.83rem; }
.stat-row:last-child { border-bottom: none; }
.sk { color: var(--muted); }
.sv { font-family: 'Space Mono', monospace; color: var(--text); font-size: 0.8rem; }

.face-crop-label { font-family: 'Space Mono', monospace; font-size: 0.6rem; color: var(--accent2); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.4rem; text-align: center; }

section[data-testid="stFileUploadDropzone"] { background: rgba(124,58,237,0.04) !important; border: 2px dashed rgba(124,58,237,0.3) !important; border-radius: 16px !important; }
.stButton > button { background: linear-gradient(135deg, var(--accent), #9333EA) !important; border: none !important; border-radius: 12px !important; color: white !important; font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important; padding: 0.6rem 1.5rem !important; transition: all 0.2s ease !important; box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 30px rgba(124,58,237,0.5) !important; }
div[data-testid="stImage"] img { border-radius: 16px !important; border: 1px solid var(--border) !important; }
.stSlider > div > div > div > div { background: var(--accent) !important; }
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
        Upload a face image or use your webcam for instant emotion detection
        powered by YOLOv8 deep learning.
    </p>
    <div class="hero-stats">
        <div class="hstat"><div class="hstat-val">69%</div><div class="hstat-label">Top-1 Acc</div></div>
        <div class="hstat"><div class="hstat-val">98.6%</div><div class="hstat-label">Top-5 Acc</div></div>
        <div class="hstat"><div class="hstat-val">8</div><div class="hstat-label">Emotions</div></div>
        <div class="hstat"><div class="hstat-val">9.4K</div><div class="hstat-label">Images</div></div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [("model_ready", False), ("model_obj", None), ("face_detector", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── MODEL LOAD SCREEN ────────────────────────────────────────────────────────
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
                YOLOv8 emotion model + face detector will be loaded.<br>
                First load takes ~30 seconds.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Load EmotiSense Model", use_container_width=True):
            with st.spinner("Loading emotion model..."):
                model, status = load_model()
            if status is True:
                with st.spinner("Loading face detector..."):
                    face_detector = load_face_detector()
                if face_detector is None:
                    st.warning("⚠️ Face detector failed to load. Predictions will run on the full image.")
                st.session_state.model_ready    = True
                st.session_state.model_obj      = model
                st.session_state.face_detector  = face_detector
                st.success("✅ Model ready! Start detecting emotions.")
                st.rerun()
            else:
                st.error(f"❌ {status}")
    st.stop()

model        = st.session_state.model_obj
face_detector = st.session_state.face_detector

# ─── MODE SELECTOR ────────────────────────────────────────────────────────────
col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
with col_nav2:
    mode = st.radio(
        "mode", ["📁 Upload Image", "📷 Live Webcam"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

# ══════════════════════════════════════════════════════
#  LEFT PANEL
# ══════════════════════════════════════════════════════
with left:
    pil_image   = None
    run_predict = False

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
            <div class="stat-row"><span class="sk">Format</span><span class="sv">JPG / PNG</span></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card">
            <div class="card-label">📷 &nbsp; Live Webcam Detection</div>
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
            <div class="stat-row"><span class="sk">Glasses</span><span class="sv">May reduce accuracy slightly</span></div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  RIGHT PANEL — RESULTS
# ══════════════════════════════════════════════════════
with right:

    if not run_predict or pil_image is None:
        st.markdown("""
        <div class="card" style="min-height:480px; display:flex; flex-direction:column;
             align-items:center; justify-content:center; text-align:center; gap:1rem;">
            <div style="font-size:5rem; opacity:0.15;">🎭</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:var(--muted2);">Awaiting Input</div>
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
        # ── Run prediction ──
        with st.spinner(""):
            st.markdown("""
            <div style="text-align:center; padding:1rem;">
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <div style="font-size:0.82rem; color:var(--muted); margin-top:0.5rem;">
                    Detecting face &amp; analyzing expression...
                </div>
            </div>
            """, unsafe_allow_html=True)

            if face_detector is not None:
                top1_idx, top1_conf, all_probs, face_crop, bbox = predict(model, pil_image, face_detector)
            else:
                # Fallback: no face detector, run on full image (original behavior)
                arr = np.array(ImageOps.exif_transpose(pil_image).convert("RGB"))
                from ultralytics import YOLO
                res = model.predict(source=arr, imgsz=224, verbose=False)
                probs     = res[0].probs
                top1_idx  = int(probs.top1)
                top1_conf = float(probs.top1conf)
                all_probs = probs.data.cpu().numpy().tolist()
                face_crop = None
                bbox      = None

        # ── No face detected ──
        if top1_idx is None:
            st.markdown("""
            <div class="no-face-card">
                <div style="font-size:3rem; margin-bottom:0.8rem;">🔍</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
                     color:#FF4757; margin-bottom:0.5rem;">No Face Detected</div>
                <div style="font-size:0.85rem; color:var(--muted2); line-height:1.7; max-width:280px; margin:0 auto;">
                    The face detector couldn't find a face in this image.<br><br>
                    Try:
                    <ul style="text-align:left; margin-top:0.5rem; padding-left:1.2rem;">
                        <li>Face the camera directly</li>
                        <li>Ensure good lighting</li>
                        <li>Move closer / fill the frame with your face</li>
                        <li>Remove obstructions (mask, sunglasses)</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            emotion  = EMOTION_NAMES[top1_idx]
            emoji    = EMOTION_EMOJIS[emotion]
            color    = EMOTION_COLORS[emotion]
            desc     = EMOTION_DESC[emotion]
            conf_pct = top1_conf * 100

            # ── Annotated original image ──
            if bbox is not None:
                annotated = draw_face_box(pil_image, bbox, emotion, color)
                st.image(annotated, caption="Detected face region", use_container_width=True)

            # ── Face crop preview ──
            if face_crop is not None:
                col_crop1, col_crop2, col_crop3 = st.columns([1, 1, 1])
                with col_crop2:
                    st.markdown('<div class="face-crop-label">🎯 Face crop (model input)</div>', unsafe_allow_html=True)
                    st.image(face_crop, use_container_width=True)

            st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

            # ── Main result card ──
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
                    <div class="conf-bar-fill"
                         style="width:{conf_pct}%;
                                background:linear-gradient(90deg,{color}88,{color});
                                --w:{conf_pct}%;">
                    </div>
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.8rem;">
                    <span style="padding:0.2rem 0.7rem; border-radius:50px; background:rgba(255,255,255,0.06); font-size:0.72rem; color:var(--muted2);">YOLOv8m-cls</span>
                    <span style="padding:0.2rem 0.7rem; border-radius:50px; background:rgba(255,255,255,0.06); font-size:0.72rem; color:var(--muted2);">224×224 input</span>
                    <span style="padding:0.2rem 0.7rem; border-radius:50px; background:rgba(255,255,255,0.06); font-size:0.72rem; color:var(--muted2);">8-class softmax</span>
                    <span style="padding:0.2rem 0.7rem; border-radius:50px; background:rgba(46,213,115,0.12); border:1px solid rgba(46,213,115,0.3); font-size:0.72rem; color:#2ED573;">✓ Face detected</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            # ── All probabilities ──
            st.markdown("""
            <div class="card">
                <div class="card-label">📊 &nbsp; All Emotion Probabilities</div>
            """, unsafe_allow_html=True)

            sorted_probs = sorted(enumerate(all_probs), key=lambda x: x[1], reverse=True)
            bars_html = ""
            for delay_i, (idx, prob) in enumerate(sorted_probs):
                emo    = EMOTION_NAMES[idx]
                emj    = EMOTION_EMOJIS[emo]
                col    = EMOTION_COLORS[emo]
                pct    = prob * 100
                is_top = idx == top1_idx
                delay  = delay_i * 0.06
                bars_html += f"""
                <div class="prob-row" style="animation-delay:{delay}s;">
                    <span class="prob-emoji">{emj}</span>
                    <span class="prob-name" style="color:{'var(--text)' if is_top else 'var(--muted2)'}; font-weight:{'700' if is_top else '400'};">{emo}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fg" style="width:{pct:.1f}%; background:{col}{'cc' if not is_top else ''};"></div>
                    </div>
                    <span class="prob-pct {'top' if is_top else ''}">{pct:.1f}%</span>
                </div>"""

            st.markdown(bars_html + "</div>", unsafe_allow_html=True)

            # ── Download buttons ──
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Original",
                    data=buf.getvalue(),
                    file_name=f"emotion_{emotion}_{int(conf_pct)}_original.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_dl2:
                if face_crop is not None:
                    buf2 = io.BytesIO()
                    face_crop.save(buf2, format="PNG")
                    st.download_button(
                        "⬇️ Download Face Crop",
                        data=buf2.getvalue(),
                        file_name=f"emotion_{emotion}_{int(conf_pct)}_face.png",
                        mime="image/png",
                        use_container_width=True
                    )

# ─── MODEL INFO ───────────────────────────────────────────────────────────────
st.markdown("<div class='divider' style='margin-top:2rem;'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:1.2rem; animation:fadeIn 0.8s ease both;">
    <div style="font-family:'Space Mono',monospace; font-size:0.68rem; color:var(--accent2);
         text-transform:uppercase; letter-spacing:3px; margin-bottom:0.4rem;">Model Intelligence</div>
    <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;">Architecture & Training Details</div>
</div>
""", unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
with mc1:
    st.markdown("""
    <div class="card">
        <div class="card-label">🧠 &nbsp; Architecture</div>
        <div class="stat-row"><span class="sk">Model</span><span class="sv">YOLOv8m-cls</span></div>
        <div class="stat-row"><span class="sk">Backbone</span><span class="sv">CSPDarknet53</span></div>
        <div class="stat-row"><span class="sk">Neck</span><span class="sv">C2f Modules</span></div>
        <div class="stat-row"><span class="sk">Head</span><span class="sv">GAP + Dropout</span></div>
        <div class="stat-row"><span class="sk">Parameters</span><span class="sv">15.7M</span></div>
        <div class="stat-row"><span class="sk">Output</span><span class="sv">Softmax × 8</span></div>
    </div>""", unsafe_allow_html=True)

with mc2:
    st.markdown("""
    <div class="card">
        <div class="card-label">⚙️ &nbsp; Training Config</div>
        <div class="stat-row"><span class="sk">Epochs</span><span class="sv">34 best / 50</span></div>
        <div class="stat-row"><span class="sk">Optimizer</span><span class="sv">Adam</span></div>
        <div class="stat-row"><span class="sk">Dropout</span><span class="sv">0.4</span></div>
        <div class="stat-row"><span class="sk">Batch Size</span><span class="sv">32</span></div>
        <div class="stat-row"><span class="sk">Image Size</span><span class="sv">224 × 224</span></div>
        <div class="stat-row"><span class="sk">GPU</span><span class="sv">Tesla T4</span></div>
    </div>""", unsafe_allow_html=True)

with mc3:
    st.markdown("""
    <div class="card">
        <div class="card-label">📈 &nbsp; Performance</div>
        <div class="stat-row"><span class="sk">Top-1 Accuracy</span><span class="sv">69.3%</span></div>
        <div class="stat-row"><span class="sk">Top-5 Accuracy</span><span class="sv">98.6%</span></div>
        <div class="stat-row"><span class="sk">Best Class</span><span class="sv">Happy (82%)</span></div>
        <div class="stat-row"><span class="sk">Dataset</span><span class="sv">9,400 images</span></div>
        <div class="stat-row"><span class="sk">Classes</span><span class="sv">8 emotions</span></div>
        <div class="stat-row"><span class="sv">~0.9ms/img</span><span class="sk">Inference</span></div>
    </div>""", unsafe_allow_html=True)

with mc4:
    detector_status = "✅ Active" if face_detector is not None else "❌ Not loaded"
    st.markdown(f"""
    <div class="card">
        <div class="card-label">🔍 &nbsp; Face Detection</div>
        <div class="stat-row"><span class="sk">Detector</span><span class="sv">Haar Cascade</span></div>
        <div class="stat-row"><span class="sk">Status</span><span class="sv">{detector_status}</span></div>
        <div class="stat-row"><span class="sk">Padding</span><span class="sv">30% around face</span></div>
        <div class="stat-row"><span class="sk">EXIF Fix</span><span class="sv">✅ Enabled</span></div>
        <div class="stat-row"><span class="sk">Square pad</span><span class="sv">✅ Enabled</span></div>
        <div class="stat-row"><span class="sk">Fallback</span><span class="sv">Full image</span></div>
    </div>""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2.5rem 1rem 1.5rem; margin-top:1rem; border-top:1px solid var(--border);">
    <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800;
         background:linear-gradient(135deg,#fff,var(--accent2));
         -webkit-background-clip:text; -webkit-text-fill-color:transparent;
         background-clip:text; margin-bottom:0.3rem;">EmotiSense AI</div>
    <div style="color:var(--muted); font-size:0.82rem; margin-bottom:0.5rem;">
        Face Emotion Detection · YOLOv8 Deep Learning · OpenCV Face Detection · 8 Emotion Classes
    </div>
    <div style="color:var(--muted); font-size:0.72rem;">
        ⚠️ For research & educational purposes only — not a clinical tool
    </div>
</div>
""", unsafe_allow_html=True)
