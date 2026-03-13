"""
HPE Velocity Tracking System — Main Entry Point
Barbell Back Squat Analysis using MediaPipe BlazePose

Usage (post-processing):
    python main.py                          # front-view video, no calibration
    python main.py --video side             # side-view video
    python main.py --ppm 320               # calibrated: 320 pixels per metre
    python main.py --realtime              # live camera feed

Calibration tip:
    A standard Olympic barbell is 2.2 m wide.  Measure how many pixels span
    the barbell in one frame (e.g. using paint or any image viewer), then:
        pixels_per_meter = <pixel_width_of_barbell> / 2.2
"""

import argparse
import os
import sys

import pandas as pd

from velocity_tracker import (
    calculate_hip_velocity,
    plot_velocity,
    visualise_pose,
    SquatVelocityTracker,
)

# ---------------------------------------------------------------------------
# Video file registry
# ---------------------------------------------------------------------------

# Set the VIDEO_DIR environment variable to point to your local video folder.
# e.g. on Windows:  set VIDEO_DIR=C:\Users\you\Videos\squat-tests
# e.g. on Mac/Linux: export VIDEO_DIR=/home/you/Videos/squat-tests
# If not set, defaults to the current directory.
VIDEO_DIR = os.environ.get('VIDEO_DIR', '.')

VIDEOS = {
    'front': os.path.join(VIDEO_DIR, 'Front-View Squat - 30fps Trim.mp4'),
    'side':  os.path.join(VIDEO_DIR, 'Side-view Squat 30fps (2) - Trim.mp4'),
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

    unit = results.get('unit', 'px/s')

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
        rep_df = pd.DataFrame([
            {
                'rep':               i + 1,
                f'MCV_{unit}':       r.mean_velocity,
                f'PCV_{unit}':       r.peak_velocity,
                'duration_s':        r.duration,
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
    save_annotated: bool,
) -> None:
    """
    Full post-processing pipeline for a recorded squat video.

    Steps:
      1. Extract hip velocity (pose estimation + filtering + phase detection)
      2. Print results summary
      3. Save velocity plot
      4. Save CSV data
      5. (Optional) Write annotated video with skeleton overlay
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
        f"Hip Vertical Velocity – {view_label}-View Squat  "
        f"[{results['unit']}  |  {results['fps']:.0f} FPS]"
    )
    plot_velocity(results, title=plot_title, save_path='velocity_plot.png')

    # 3. CSV export
    save_results_csv(results, path='data/results.csv')

    # 4. Optional annotated video
    if save_annotated:
        annotated_path = f'output_with_pose_{video_key}.mp4'
        visualise_pose(video_path, output_path=annotated_path, model_complexity=1)

    print("\nAll done.")
    print("  velocity_plot.png     – velocity-time graph")
    print("  data/results_frames.csv – frame-level data")
    if results.get('reps'):
        print("  data/results_reps.csv   – per-rep MCV / PCV")
    if save_annotated:
        print(f"  output_with_pose_{video_key}.mp4 – annotated video")


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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HPE Velocity Tracking System – barbell squat analysis"
    )
    parser.add_argument(
        '--video', default='front',
        help=(
            "Which video to process. Use 'front' or 'side' for the default test videos, "
            "or provide a full file path to any video (default: front)"
        ),
    )
    parser.add_argument(
        '--ppm', type=float, default=None, metavar='PIXELS_PER_METRE',
        help=(
            "Calibration: pixels per metre. "
            "Omit to report in px/s (relative). "
            "Example: if the barbell spans 660 px → --ppm 300  (660/2.2)"
        ),
    )
    parser.add_argument(
        '--cutoff', type=float, default=6.0,
        help="Butterworth low-pass cutoff frequency in Hz (default: 6)",
    )
    parser.add_argument(
        '--complexity', type=int, choices=[0, 1, 2], default=2,
        help="MediaPipe model complexity: 0=Lite, 1=Full, 2=Heavy (default: 2)",
    )
    parser.add_argument(
        '--annotate', action='store_true',
        help="Also write an annotated video with the skeleton overlay",
    )
    parser.add_argument(
        '--realtime', action='store_true',
        help="Run live camera tracker instead of processing a video file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  HPE VELOCITY TRACKING SYSTEM")
    print("  Barbell Back Squat Analysis — MediaPipe BlazePose")
    print("=" * 60)

    args = parse_args()

    if args.realtime:
        ppm = args.ppm if args.ppm else 500.0
        if args.ppm is None:
            print(
                "\nNote: No --ppm supplied. Defaulting to 500 px/m. "
                "Press 'c' to calibrate with a known distance.\n"
            )
        run_realtime(pixels_per_meter=ppm, filter_cutoff=args.cutoff)
    else:
        if args.ppm is None:
            print(
                "\nNote: No --ppm calibration supplied. "
                "Velocities will be reported in px/s (relative units). "
                "For true m/s, measure the barbell span in pixels and pass "
                "--ppm <pixels> / 2.2 (Olympic bar = 2.2 m).\n"
            )
        run_post_processing(
            video_key=args.video,
            pixels_per_meter=args.ppm,
            filter_cutoff=args.cutoff,
            model_complexity=args.complexity,
            save_annotated=args.annotate,
        )


if __name__ == '__main__':
    main()
