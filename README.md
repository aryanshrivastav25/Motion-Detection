# ActionWatch — Live Action Recognition App

Real-time human action detection (sit / stand / fall / fight) using a
pretrained R3D-18 model (Kinetics-400) served through a Flask web app.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Features

| Feature | Detail |
|---|---|
| Live feed | Streams webcam via MJPEG; runs inference every ~16 sampled frames (~2 s) |
| Video upload | Upload any MP4/AVI/MOV; returns per-clip timeline |
| Alert banner | Animated red banner + event log for FALL or FIGHT detections |
| Confidence bar | Softmax probability of the top prediction |
| Event log | Timestamped history of all detections in the live view |

## Action Mapping

The model outputs one of 400 Kinetics-400 labels.  
These are mapped to 4 categories:

- **FALL** — fall, stumble, trip, collapse …
- **FIGHT / VIOLENCE** — punch, kick, wrestling, martial arts …
- **SITTING** — sitting, crouching, squatting, yoga …
- **STANDING / MOVING** — standing, walking, running, dancing …

Anything outside those four groups shows the raw Kinetics label.

## Tips

- For best accuracy use good lighting and a clear background.
- The model needs 16 frames. Detection latency depends on CPU/GPU speed.
- On a modern CPU expect ~1–2 s per inference; on GPU < 0.3 s.
- To use GPU: make sure CUDA is installed; PyTorch will detect it automatically.
