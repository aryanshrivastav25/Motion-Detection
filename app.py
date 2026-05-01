import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch
import numpy as np
import threading
import time
import os
import json
from flask import Flask, render_template, Response, request, jsonify
from torchvision.models.video import r3d_18, R3D_18_Weights
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
print("Loading R3D-18 model...")
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()
labels = weights.meta["categories"]
print(f"Model loaded. {len(labels)} Kinetics-400 classes available.")

# ─── Action Mapping ───────────────────────────────────────────────────────────
FALL_KEYWORDS   = ['fall', 'stumbl', 'trip', 'collaps', 'toppl', 'tumbl']
FIGHT_KEYWORDS  = ['punch', 'slap', 'wrestling', 'fight', 'boxing', 'martial',
                   'karate', 'kick', 'headbutt', 'brawl', 'taekwondo', 'judo',
                   'jiu jitsu', 'sword fight', 'shooting']
SIT_KEYWORDS    = ['sitting', 'crouch', 'squat', 'kneel', 'yoga', 'meditation']
STAND_KEYWORDS  = ['standing', 'walk', 'run', 'jump', 'danc', 'stretch', 'lift',
                   'wave', 'clap', 'gesture']

DANGER_ACTIONS  = {'FALL', 'FIGHT/VIOLENCE'}

def map_to_action(label: str) -> tuple[str, bool]:
    """Map a Kinetics-400 label to a human action + alert flag."""
    ll = label.lower()
    if any(k in ll for k in FALL_KEYWORDS):
        return f'FALL, {ll}', True
    if any(k in ll for k in FIGHT_KEYWORDS):
        return f'FIGHT / VIOLENCE, {ll}', True
    if any(k in ll for k in SIT_KEYWORDS):
        return f'SITTING, {ll}', False
    if any(k in ll for k in STAND_KEYWORDS):
        return f'STANDING / MOVING, {ll}', False
    # Fallback: show shortened raw label
    return 'Monitoring...', False

# ─── Live Camera State ────────────────────────────────────────────────────────
cam_state: dict = {
    'active':     False,
    'prediction': '—',
    'raw_label':  '',
    'alert':      False,
    'confidence': 0.0,
}
cam_lock    = threading.Lock()
frame_lock  = threading.Lock()
output_frame: np.ndarray | None = None

from queue import Queue

latest_frame = {'frame': None}   # shared between threads
latest_frame_lock = threading.Lock()

def capture_loop(cap):
    """Thread 1: just reads frames as fast as possible."""
    while cam_state['active']:
        ret, frame = cap.read()
        if not ret:
            break
        with latest_frame_lock:
            latest_frame['frame'] = frame   # always overwrite with newest

def run_inference_loop():
    global output_frame

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        with cam_lock:
            cam_state['prediction'] = 'Camera not found'
        return

    # Start capture in its own thread
    t = threading.Thread(target=capture_loop, args=(cap,), daemon=True)
    t.start()

    frames_buf = []
    SAMPLE_EVERY = 4

    while cam_state['active']:
        # Always grab the LATEST frame
        with latest_frame_lock:
            frame = latest_frame['frame']

        if frame is None:
            time.sleep(0.01)
            continue

        # ── Build overlay on latest frame ──
        with cam_lock:
            pred   = cam_state['prediction']
            alert  = cam_state['alert']
            conf   = cam_state['confidence']

        disp = frame.copy()
        # ... (keep your existing overlay drawing code here)

        with frame_lock:
            output_frame = disp.copy()

        # ── Sample for inference ──
        frames_buf.append(
            cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        )

        if len(frames_buf) == 16:
            clip = np.array(frames_buf)
            clip = torch.tensor(clip).permute(3,0,1,2).unsqueeze(0).float() / 255.0

            with torch.no_grad():
                logits = model(clip)

            probs    = torch.softmax(logits, dim=1)
            conf_val = probs.max().item()
            pred_idx = logits.argmax().item()
            raw      = labels[pred_idx]
            action, is_alert = map_to_action(raw)

            with cam_lock:
                cam_state['prediction'] = action
                cam_state['raw_label']  = raw
                cam_state['alert']      = is_alert
                cam_state['confidence'] = conf_val

            frames_buf.clear()

        time.sleep(0.01)  # tiny yield to not spin the CPU

    cap.release()

def generate_stream():
    last_frame_id = None
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.04)
                continue
            frame_id = id(output_frame)   # check if frame is actually new
            if frame_id == last_frame_id:
                time.sleep(0.04)
                continue
            last_frame_id = frame_id
            _, buf = cv2.imencode('.jpg', output_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])  # 70 vs 80 helps

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    with cam_lock:
        if not cam_state['active']:
            cam_state['active']     = True
            cam_state['prediction'] = 'Warming up…'
            cam_state['alert']      = False
            cam_state['confidence'] = 0.0
            t = threading.Thread(target=run_inference_loop, daemon=True)
            t.start()
    return jsonify(status='started')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    with cam_lock:
        cam_state['active'] = False
    return jsonify(status='stopped')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam_status')
def cam_status():
    with cam_lock:
        return jsonify(
            prediction = cam_state['prediction'],
            raw_label  = cam_state['raw_label'],
            alert      = cam_state['alert'],
            confidence = round(cam_state['confidence'], 3),
            active     = cam_state['active'],
        )

# ─── Video Upload & Processing ───────────────────────────────────────────────
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify(error='No file provided'), 400

    f        = request.files['video']
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify(error='Invalid filename'), 400

    fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(fpath)

    try:
        results, meta = process_video_file(fpath)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)

    return jsonify(results=results, meta=meta)

def process_video_file(filepath: str):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    results      = []
    frames_buf   = []
    frame_idx    = 0
    clip_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fr_small   = cv2.resize(frame, (224, 224))
        fr_rgb     = cv2.cvtColor(fr_small, cv2.COLOR_BGR2RGB)
        frames_buf.append(fr_rgb)

        if len(frames_buf) == 16:
            clip = np.array(frames_buf)
            clip = torch.tensor(clip).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

            with torch.no_grad():
                logits = model(clip)

            probs    = torch.softmax(logits, dim=1)
            conf_val = probs.max().item()
            pred_idx = logits.argmax().item()
            raw      = labels[pred_idx]
            action, is_alert = map_to_action(raw)

            ts = round((frame_idx / fps), 2)

            results.append({
                'clip':       clip_idx + 1,
                'timestamp':  ts,
                'time_fmt':   fmt_time(ts),
                'action':     action,
                'raw_label':  raw,
                'confidence': round(conf_val, 3),
                'alert':      is_alert,
            })

            clip_idx   += 1
            frames_buf  = []

    cap.release()

    alert_count = sum(1 for r in results if r['alert'])
    meta = {
        'duration':    round(duration, 2),
        'total_clips': clip_idx,
        'alerts':      alert_count,
        'fps':         round(fps, 2),
    }
    return results, meta

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
