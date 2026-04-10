# 🎯 Face Recognition Attendance System

A real-time face recognition attendance system built with Python and OpenCV. Automatically detects and identifies known faces via webcam and logs attendance with timestamps to a CSV file.

---

## 📸 Demo

> Point your webcam at a registered person → face gets detected → name appears → attendance is logged automatically.

---

## 🗂️ Project Structure

```
face_attendance/
│
├── dataset/                   # Captured face images (auto-created)
│   └── PersonName/            # One folder per person
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
│
├── trainer/                   # Trained model files (auto-created)
│   ├── trainer.yml            # LBPH trained model
│   └── labels.pkl             # ID → Name mapping
│
├── 1_capture_faces.py         # Step 1: Capture face images from webcam
├── 2_train_model.py           # Step 2: Train the face recognizer
├── 3_recognize.py             # Step 3: Real-time recognition + attendance
├── attendance.csv             # Auto-generated attendance log
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| OpenCV (`opencv-contrib-python`) | Face detection & recognition |
| NumPy | Image array processing |
| Pillow | Image loading utility |
| CSV (built-in) | Attendance logging |
| Pickle (built-in) | Saving label mappings |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/face_attendance.git
cd face_attendance
```

### 2. Create and activate a virtual environment

```bash
# Create venv
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1 — Capture face images

```bash
python 1_capture_faces.py
```

- Enter the person's name when prompted
- Look at the webcam — 50 images are auto-captured
- Repeat for every person you want to register

### Step 2 — Train the model

```bash
python 2_train_model.py
```

- Reads all images from `dataset/`
- Trains an LBPH face recognizer
- Saves the model to `trainer/`

### Step 3 — Start live recognition

```bash
python 3_recognize.py
```

- Opens webcam feed
- Recognized faces → green box with name
- Unknown faces → red box labeled "Unknown"
- Attendance auto-saved to `attendance.csv`
- Press `q` to quit

---

## 📋 Attendance Log (attendance.csv)

```
Name,Date,Time
Alice,2026-04-10,10:23:45
Bob,2026-04-10,10:24:01
Charlie,2026-04-10,10:24:18
```

Each person is marked only **once per session**, no matter how many times they appear on camera.

---

## 🧠 How It Works

```
Webcam Feed
    │
    ▼
Haar Cascade Classifier  ──→  Detects face location (bounding box)
    │
    ▼
LBPH Face Recognizer     ──→  Identifies WHO the face belongs to
    │
    ▼
Confidence Score Check   ──→  Below threshold → Known | Above → Unknown
    │
    ▼
Attendance Logger        ──→  Writes Name + Date + Time to CSV
```

**Haar Cascade** — a fast algorithm that finds where faces are in a frame using pre-trained XML patterns.

**LBPH (Local Binary Pattern Histogram)** — analyzes the texture patterns of a face and compares them against trained samples to identify a person.

---

## 🔧 Configuration

You can tweak these constants at the top of each script:

| File | Variable | Default | Description |
|---|---|---|---|
| `1_capture_faces.py` | `SAMPLES_TO_CAPTURE` | `50` | Number of images to capture per person |
| `3_recognize.py` | `CONFIDENCE_THRESHOLD` | `70` | Lower = stricter matching |

> **Tip:** If you're getting too many "Unknown" results, raise `CONFIDENCE_THRESHOLD` to `80` or `85`.

---

## ➕ Adding a New Person

1. Run `python 1_capture_faces.py` and enter the new person's name
2. Re-run `python 2_train_model.py` to retrain with the new data
3. Run `python 3_recognize.py` as usual

---

## ⚠️ Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: cv2` | venv not activated | Run `venv\Scripts\activate` |
| `ERROR: Cannot open webcam` | Camera in use or wrong index | Close other apps using the camera |
| `ERROR: No data found in dataset/` | Step 1 not run | Run `1_capture_faces.py` first |
| `Trained model not found` | Step 2 not run | Run `2_train_model.py` first |
| Face shows as "Unknown" | Too few training images | Recapture with 30–50 images |

---

## 📦 Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
opencv-contrib-python
numpy
pillow
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [OpenCV](https://opencv.org/) — Open Source Computer Vision Library
- Haar Cascade XML files bundled with OpenCV
- LBPH Face Recognizer from `opencv-contrib`