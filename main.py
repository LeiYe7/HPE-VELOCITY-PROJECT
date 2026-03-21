"""
HPE Velocity Tracking System — Main Entry Point
Squat Speed Analysis using MediaPipe BlazePose

To run: python main.py
To select a video, edit the CONFIG block inside main() at the bottom of this file.
"""

import os
import sys

import pandas as pd

from velocity_tracker import (
    calculate_hip_velocity,
    plot_velocity,
    visualise_pose_with_velocity,
    SquatVelocityTracker,
)

# ---------------------------------------------------------------------------
# Video file registry
# ---------------------------------------------------------------------------

# Test videos are not included in the repository due to size, but is submitted in a separate file included on the report
# Replace this path with the folder containing your video files.
# The environment variable VIDEO_DIR takes precedence if set.
VIDEO_DIR = os.environ.get('VIDEO_DIR', r'C:\Users\tommy\OneDrive\Documents\DSP test')

# Output videos are saved here (outside the repo to avoid committing large files).
# Replace with your preferred local output folder.
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', r'C:\Users\tommy\OneDrive\Documents\DSP test\output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEOS = {
    'normal': os.path.join(VIDEO_DIR, 'Normal Tempo - Front.mp4'),
    'slow':  os.path.join(VIDEO_DIR, 'Slow Tempo - Front.mp4'),
    'fast':  os.path.join(VIDEO_DIR, 'Fast Tempo - Front.mp4'),
    'partial':  os.path.join(VIDEO_DIR, 'Partial Rep - Front.mp4'),
    'bf':  os.path.join(VIDEO_DIR, 'Bad Form - Front.mp4'),
    'foc':  os.path.join(VIDEO_DIR, 'Front Occlusion .mp4'),
    'normal-side':  os.path.join(VIDEO_DIR, 'Normal Tempo - Side .mp4'),
}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(results: dict, path: str = 'data/results.csv') -> None:
    """
    Save frame-level velocity data and per-rep metrics to CSV.

    Args:
        results: Dict returned by calculate_hip_velocity.
        path   : Output CSV file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    unit = results.get('unit', 'px/s')  # 'm/s' when calibrated, 'px/s' otherwise

    # Frame-level data
    frame_df = pd.DataFrame({
        'time_s':            results['timestamps'],
        f'velocity_{unit}':  results['velocities'],
        f'raw_velocity_{unit}': results.get('raw_velocities', results['velocities']),
    })
    frame_path = path.replace('.csv', '_frames.csv')
    frame_df.to_csv(frame_path, index=False)
    print(f"Frame data saved : {frame_path}")

    # Per-rep metrics
    reps = results.get('reps', [])
    if reps:
        baseline_mcv = reps[0].mean_velocity if reps[0].mean_velocity else 1.0
        rep_df = pd.DataFrame([
            {
                'rep':               i + 1,
                f'conc_MCV_{unit}':  round(r.mean_velocity, 1),
                f'conc_PCV_{unit}':  round(r.peak_velocity, 1),
                'conc_duration_s':   round(r.duration, 2),
                'ecc_speed_px_s':    round(r.eccentric_mean_speed, 1),
                'ecc_duration_s':    round(r.eccentric_duration, 2),
                'speed_loss_pct':    round((baseline_mcv - r.mean_velocity) / baseline_mcv * 100, 1),
            }
            for i, r in enumerate(reps)
        ])
        rep_path = path.replace('.csv', '_reps.csv')
        rep_df.to_csv(rep_path, index=False)
        print(f"Rep metrics saved: {rep_path}")


# ---------------------------------------------------------------------------
# Post-processing pipeline
# ---------------------------------------------------------------------------

def run_post_processing(
    video_key: str,
    pixels_per_meter: float | None,
    filter_cutoff: float,
    model_complexity: int,
) -> None:
    """
    Full post-processing pipeline for a recorded squat video.

    Steps:
      1. Extract hip velocity (pose estimation + filtering + phase detection)
      2. Print results summary
      3. Save velocity plot
      4. Save CSV data
      5. Write annotated video with skeleton overlay
      6. Write side-by-side video with pose + velocity graph
    """
    # Accept either a named key ('front'/'side') or a direct file path
    if os.path.isfile(video_key):
        video_path = video_key
    else:
        video_path = VIDEOS.get(video_key)
    if video_path is None:
        print(f"Unknown video key '{video_key}'. Use 'front', 'side', or a direct file path.")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Ensure you are running from the project directory.")
        sys.exit(1)

    # 1. Extract velocities
    results = calculate_hip_velocity(
        video_path,
        pixels_per_meter=pixels_per_meter,
        filter_cutoff=filter_cutoff,
        model_complexity=model_complexity,
    )

    if results is None:
        print("Processing failed. Check video, lighting, and that the full body is visible.")
        sys.exit(1)

    # 2. Velocity-time plot
    view_label = video_key.capitalize()
    plot_title = (
        f"Hip Vertical Velocity – {view_label} Squat  "
        f"[{results['unit']}  |  {results['fps']:.0f} FPS]"
    )
    plot_velocity(results, title=plot_title, save_path='velocity_plot.png')

    # 3. CSV export
    save_results_csv(results, path='data/results.csv')

    # 4. Side-by-side pose + velocity visualization
    video_label = os.path.splitext(os.path.basename(video_key))[0] if os.path.isfile(video_key) else video_key
    visual_path = os.path.join(OUTPUT_DIR, f'analyse_{video_label}.mp4')
    visualise_pose_with_velocity(video_path, results, output_path=visual_path, model_complexity=2)

    print("\nAll done.")
    print("  velocity_plot.png     – velocity-time graph")
    print("  data/results_frames.csv – frame-level data")
    if results.get('reps'):
        print("  data/results_reps.csv   – per-rep MCV / PCV")
    print(f"  {visual_path} – pose + velocity visualization")


# ---------------------------------------------------------------------------
# Real-time tracking
# ---------------------------------------------------------------------------

def run_realtime(pixels_per_meter: float, filter_cutoff: float) -> None:
    """Start the live camera tracking loop."""
    tracker = SquatVelocityTracker(
        camera_id=0,
        pixels_per_meter=pixels_per_meter,
        target_fps=60,
        filter_cutoff=filter_cutoff,
    )
    print("\nReal-time tracker ready.")
    print("  Press 'c' to calibrate with a known distance.")
    print("  Press 'q' to quit and print session summary.\n")
    tracker.run()

    summary = tracker.get_session_summary()
    if summary:
        print("\n=== Session Summary ===")
        for key, val in summary.items():
            if isinstance(val, float):
                print(f"  {key:<25}: {val:.3f}")
            else:
                print(f"  {key:<25}: {val}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # =========================================================
    # CONFIGURE HERE — edit these values, then run python main.py
    # =========================================================

    VIDEO      = 'foc'   # key from VIDEOS dict above, or a full file path
                           # e.g. 'normal', 'slow', or r'C:\path\to\video.mp4'

    REALTIME   = False     # set True to use live camera instead of a video file

    PPM        = None      # pixels per metre for m/s output (None = use px/s)
    CUTOFF     = 6.0       # Butterworth filter cutoff in Hz
    COMPLEXITY = 2         # MediaPipe model: 0=fast/less accurate, 2=slow/most accurate

    # =========================================================

    print("\n" + "=" * 60)
    print("  HPE VELOCITY TRACKING SYSTEM")
    print("  Squat Speed Analysis — MediaPipe BlazePose")
    print("=" * 60)

    if REALTIME:
        ppm = PPM if PPM else 500.0  # rough default; press 'c' in the window to calibrate
        run_realtime(pixels_per_meter=ppm, filter_cutoff=CUTOFF)
    else:
        run_post_processing(
            video_key=VIDEO,
            pixels_per_meter=PPM,
            filter_cutoff=CUTOFF,
            model_complexity=COMPLEXITY,
        )


if __name__ == '__main__':
    main()
