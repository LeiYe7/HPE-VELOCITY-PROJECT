# HPE Velocity Project
Digital Systems Project — Barbell Back Squat Velocity Tracking using MediaPipe BlazePose.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the test videos
Videos are **not stored in this repo** due to file size. Download them from the shared folder:

> **Video folder:** *(paste your OneDrive/Google Drive link here)*

Place the downloaded videos in a folder on your machine, then set the `VIDEO_DIR` environment variable to point to it.

**Windows:**
```bat
set VIDEO_DIR=C:\Users\you\Videos\squat-tests
```

**Mac/Linux:**
```bash
export VIDEO_DIR=/home/you/Videos/squat-tests
```

Alternatively, you can pass a video file path directly (see Usage below).

---

## Usage

### Post-processing a recorded video

```bash
# Process the default front-view video (uses VIDEO_DIR)
python main.py

# Process the side-view video
python main.py --video side

# Process any video file directly (no VIDEO_DIR needed)
python main.py --video "C:/path/to/your/video.mp4"

# With pixel-per-metre calibration (for true m/s output)
# Measure barbell width in pixels, divide by 2.2 (Olympic bar = 2.2m)
python main.py --ppm 300

# Write annotated video with skeleton overlay
python main.py --annotate
```

### Real-time tracking (live camera)
```bash
python main.py --realtime

# With calibration
python main.py --realtime --ppm 500
```

**Keys during real-time mode:**
- `c` — calibrate using a known distance
- `q` — quit and print session summary

---

## Output files

| File | Description |
|------|-------------|
| `velocity_plot.png` | Velocity-time graph with phase shading and per-rep bar chart |
| `data/results_frames.csv` | Frame-level velocity data |
| `data/results_reps.csv` | Per-rep Mean Concentric Velocity (MCV) and Peak Concentric Velocity (PCV) |
| `output_with_pose_<view>.mp4` | Annotated video with skeleton overlay (if `--annotate` used) |

---

## Project structure

```
.
├── main.py              # Entry point and CLI
├── velocity_tracker.py  # Core tracking, filtering, and metrics
├── pose_extractor.py    # MediaPipe pose extraction wrapper
├── utils.py             # Shared utilities
├── requirements.txt     # Python dependencies
├── data/                # CSV output and video index
└── README.md
```

## Calibration tip

For true m/s output, measure how many pixels span the barbell in one frame (use any image viewer), then:
```
--ppm = pixel_width_of_barbell / 2.2
```
A standard Olympic barbell is 2.2 m wide.
