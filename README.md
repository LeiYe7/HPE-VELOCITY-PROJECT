# HPE Velocity Tracking System

Squat barbell velocity tracker using human pose estimation (HPE) with MediaPipe BlazePose. Designed for Velocity-Based Training (VBT), the system estimates concentric and eccentric bar speed from video by tracking hip joint displacement as a proxy for barbell position.

Supports both **post-processing** of recorded videos and **real-time** tracking via a live camera feed.

## Features

- Frame-by-frame hip-centre velocity estimation via MediaPipe BlazePose
- Butterworth low-pass filtering for noise reduction
- Automatic concentric/eccentric phase detection and rep counting
- Per-rep metrics: Mean Concentric Velocity (MCV), Peak Concentric Velocity (PCV), eccentric speed, velocity loss %
- Velocity-time graph with phase shading and per-rep bar chart
- Annotated output video with skeleton overlay and scrolling velocity graph
- Real-time live-camera mode with on-screen calibration
- Validation utilities: MAE, RMSE, Bland-Altman analysis

## Project Structure

```
HPE-VELOCITY PROJECT/
├── main.py               # Entry point — configure and run analysis here
├── pose_extractor.py     # MediaPipe BlazePose wrapper
├── velocity_tracker.py   # Velocity calculation, phase detection, visualisation
├── utils.py              # Calibration, validation metrics, Bland-Altman plots
├── requirements.txt      # Python dependencies (pinned versions)
├── data/
│   ├── results_frames.csv  # Generated: frame-level velocity data
│   ├── results_reps.csv    # Generated: per-rep MCV/PCV metrics
│   └── README.md           # Data folder documentation
└── velocity_plot.png       # Generated: velocity-time graph
```

## Prerequisites

- **Python 3.11+**
- A webcam (for real-time mode) or video files of squat exercises

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/LeiYe7/HPE-VELOCITY-PROJECT.git
   cd HPE-VELOCITY-PROJECT
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Adding Video Files

Video files are excluded from the repository due to their size. Place your squat videos in a local folder and update the `VIDEO_DIR` path at the top of `main.py`, or set the `VIDEO_DIR` environment variable:

```bash
export VIDEO_DIR="/path/to/your/videos"        # Linux / macOS
set VIDEO_DIR=C:\path\to\your\videos            # Windows
```

## Usage

### Post-processing a recorded video

1. Open `main.py` and edit the `CONFIG` block inside `main()`:

   ```python
   VIDEO      = 'normal'     # Key from the VIDEOS dict, or a full file path
   COMPLEXITY = 2            # MediaPipe model: 0=fast, 1=balanced, 2=accurate
   CONFIDENCE = 0.3          # Minimum landmark visibility to accept a frame
   MAX_MISSING = 5           # Consecutive missed frames before interpolation
   CUTOFF     = 6.0          # Butterworth filter cutoff frequency (Hz)
   VEL_THRESHOLD = 5.0       # Minimum speed (px/s) to count as movement
   NOISE_RATIO   = 0.15      # Discard phases below this fraction of fastest rep
   GRAPH_WINDOW  = 150       # Frames of history in scrolling velocity graph
   ```

2. Run the analysis:

   ```bash
   python main.py
   ```

3. Outputs:
   - `velocity_plot.png` — velocity-time graph with phase shading
   - `data/results_frames.csv` — frame-level velocity and timestamp data
   - `data/results_reps.csv` — per-rep MCV, PCV, and speed loss metrics
   - Annotated video with pose overlay and velocity graph (saved to `OUTPUT_DIR`)

### Real-time mode

Set `REALTIME = True` in the config block and run `python main.py`. Press **c** in the video window to calibrate with a known distance, and **q** to quit.

## Output Metrics

| Metric | Description |
|--------|-------------|
| MCV | Mean Concentric Velocity — average upward speed during the concentric phase |
| PCV | Peak Concentric Velocity — maximum instantaneous upward speed |
| Eccentric speed | Average downward speed during the lowering phase |
| Speed loss % | Velocity drop relative to the first rep, used for fatigue monitoring |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| NumPy | 1.26.4 | Array operations and numerical computation |
| OpenCV | 4.11.0 | Video I/O and frame processing |
| SciPy | 1.12.0 | Butterworth filtering and statistical analysis |
| MediaPipe | 0.10.9 | BlazePose human pose estimation |
| Matplotlib | 3.8.3 | Velocity plots and Bland-Altman charts |
| pandas | 2.2.3 | CSV export of results |
