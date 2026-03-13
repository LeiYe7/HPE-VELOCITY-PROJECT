"""
Velocity Tracker Module
Camera-based barbell velocity tracking for back squat analysis.

Implements:
  - RepData          : metrics for a single rep
  - PositionTracker  : joint position history with gap interpolation
  - RepPhaseDetector : concentric / eccentric phase classification
  - VelocityMetrics  : MCV, PCV and related statistics
  - Signal filters   : Butterworth low-pass, Savitzky-Golay
  - calculate_hip_velocity : post-processing pipeline for a video file
  - plot_velocity    : velocity-time graph with phase shading and per-rep bars
  - visualise_pose   : write annotated video with skeleton overlay
  - SquatVelocityTracker : real-time live-camera tracker
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt, lfilter, savgol_filter

from pose_extractor import PoseExtractor


# ===========================================================================
# Data class
# ===========================================================================

@dataclass
class RepData:
    """Metrics captured for a single concentric phase (one rep)."""
    mean_velocity: float    # Mean Concentric Velocity  (m/s or px/s)
    peak_velocity: float    # Peak Concentric Velocity
    duration: float         # Duration of concentric phase (s)
    timestamps: np.ndarray  # Relative timestamps within the rep (s)
    velocities: np.ndarray  # Filtered vertical velocities for the rep


# ===========================================================================
# Signal processing helpers
# ===========================================================================

def butterworth_lowpass_filter(
    data: np.ndarray,
    cutoff_freq: float,
    sample_rate: float,
    order: int = 2,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter (Section 9.2).

    Uses filtfilt for zero phase distortion — suitable for post-processing.

    Args:
        data        : 1-D input signal.
        cutoff_freq : Cutoff frequency in Hz (recommended 6 Hz for squats).
        sample_rate : Sampling rate in Hz (camera FPS).
        order       : Filter order (2 recommended).

    Returns:
        Filtered signal of the same length.
    """
    nyquist = sample_rate / 2.0
    normalized_cutoff = min(cutoff_freq / nyquist, 0.99)
    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, data)


def apply_savgol_filter(
    data: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing filter (Section 9.3).

    Preserves velocity peaks better than simple averaging.

    Args:
        data          : 1-D input signal.
        window_length : Window size (odd integer).
        polyorder     : Polynomial order (must be < window_length).

    Returns:
        Filtered signal of the same length.
    """
    if window_length % 2 == 0:
        window_length += 1
    # Clamp to signal length
    max_wl = len(data) if len(data) % 2 == 1 else len(data) - 1
    window_length = min(window_length, max_wl)
    if window_length < 3 or window_length <= polyorder:
        return data.copy()
    return savgol_filter(data, window_length, polyorder)


# ===========================================================================
# Position tracker (Section 7.3)
# ===========================================================================

class PositionTracker:
    """
    Maintains a history of (x, y) joint positions with linear interpolation
    over short detection gaps.
    """

    def __init__(self, max_missing_frames: int = 5):
        """
        Args:
            max_missing_frames: Maximum consecutive missed frames to fill by
                                interpolation before giving up.
        """
        self.positions: List[np.ndarray] = []
        self.timestamps: List[float]     = []
        self.missing_count: int          = 0
        self.max_missing                 = max_missing_frames

    def update(
        self,
        position: Optional[np.ndarray],
        timestamp: float,
        confidence: float,
    ) -> bool:
        """
        Add a new detection (or handle a gap).

        Args:
            position   : (x, y) in pixels, or None if pose was not detected.
            timestamp  : Frame timestamp in seconds.
            confidence : Landmark visibility score (0–1).

        Returns:
            True if a position was stored (detected or interpolated).
        """
        if position is None or confidence < 0.3:
            self.missing_count += 1
            if self.missing_count <= self.max_missing and len(self.positions) >= 2:
                interpolated = self._interpolate_position(timestamp)
                self.positions.append(interpolated)
                self.timestamps.append(timestamp)
                return True
            return False

        self.missing_count = 0
        self.positions.append(position)
        self.timestamps.append(timestamp)
        return True

    def _interpolate_position(self, timestamp: float) -> np.ndarray:
        """Linear extrapolation from the two most recent positions."""
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 0:
            return self.positions[-1].copy()
        velocity = (self.positions[-1] - self.positions[-2]) / dt
        new_dt   = timestamp - self.timestamps[-1]
        return self.positions[-1] + velocity * new_dt


# ===========================================================================
# Phase detector (Section 8.3)
# ===========================================================================

class RepPhaseDetector:
    """
    Classifies each velocity sample as 'concentric', 'eccentric', or
    'stationary' and extracts concentric phase windows.
    """

    def __init__(self, velocity_threshold: float = 0.05):
        """
        Args:
            velocity_threshold: Speed below which movement is 'stationary'.
                                Use m/s when calibrated, px/s otherwise.
        """
        self.velocity_threshold = velocity_threshold
        self.current_phase: str       = 'stationary'
        self.phase_history: List[str] = []

    def update(self, velocity: float) -> str:
        """
        Classify a single velocity sample.

        Args:
            velocity: Vertical velocity (positive = upward = concentric).

        Returns:
            Phase string: 'concentric', 'eccentric', or 'stationary'.
        """
        if velocity > self.velocity_threshold:
            phase = 'concentric'
        elif velocity < -self.velocity_threshold:
            phase = 'eccentric'
        else:
            phase = 'stationary'

        if self.current_phase != phase:
            if self.current_phase == 'eccentric' and phase == 'concentric':
                self.phase_history.append('bottom')
            elif self.current_phase == 'concentric' and phase in ('eccentric', 'stationary'):
                self.phase_history.append('top')
            self.current_phase = phase

        return phase

    def get_concentric_phases(
        self,
        velocities: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Scan a full velocity array and return all concentric windows.

        Args:
            velocities : 1-D array of vertical velocities.
            timestamps : Corresponding timestamps (same length).

        Returns:
            List of (start_idx, end_idx) pairs (end is exclusive).
        """
        # Reset state for a fresh pass
        self.current_phase = 'stationary'
        self.phase_history = []

        phases: List[Tuple[int, int]] = []
        in_concentric = False
        start_idx: Optional[int] = None

        for i, v in enumerate(velocities):
            phase = self.update(v)
            if phase == 'concentric' and not in_concentric:
                in_concentric = True
                start_idx = i
            elif phase != 'concentric' and in_concentric:
                in_concentric = False
                phases.append((start_idx, i))  # type: ignore[arg-type]

        if in_concentric and start_idx is not None:
            phases.append((start_idx, len(velocities)))

        return phases


# ===========================================================================
# Velocity metrics (Section 8.4)
# ===========================================================================

class VelocityMetrics:
    """Standard VBT metrics computed from a concentric velocity segment."""

    @staticmethod
    def mean_velocity(velocities: np.ndarray, timestamps: np.ndarray) -> float:
        """
        Mean Concentric Velocity (MCV).

        Computed as total displacement / total time using trapezoidal
        integration (area under the velocity curve).
        """
        total_time = float(timestamps[-1] - timestamps[0])
        if total_time <= 0:
            return 0.0
        total_displacement = float(np.trapz(velocities, timestamps))
        return total_displacement / total_time

    @staticmethod
    def peak_velocity(velocities: np.ndarray) -> float:
        """Peak Concentric Velocity (PCV)."""
        return float(np.max(velocities))

    @staticmethod
    def velocity_at_percentage(velocities: np.ndarray, percentage: float) -> float:
        """
        Velocity at a specific percentage of the concentric phase.

        Args:
            percentage: 0–100 (e.g. 50 = mid-point of the lift).
        """
        idx = int(len(velocities) * percentage / 100)
        idx = min(idx, len(velocities) - 1)
        return float(velocities[idx])

    @staticmethod
    def time_to_peak(velocities: np.ndarray, timestamps: np.ndarray) -> float:
        """Time from start of concentric phase to peak velocity (s)."""
        peak_idx = int(np.argmax(velocities))
        return float(timestamps[peak_idx] - timestamps[0])


# ===========================================================================
# Internal velocity calculation
# ===========================================================================

def _calculate_vertical_velocity(
    positions: List[np.ndarray],
    timestamps: List[float],
    pixels_per_meter: Optional[float],
) -> np.ndarray:
    """
    Compute frame-to-frame vertical velocity from a position sequence.

    Image y increases downward, so upward movement → positive velocity.

    Args:
        positions       : (x, y) pixel positions.
        timestamps      : Corresponding timestamps (s).
        pixels_per_meter: Calibration factor. None → return in px/s.

    Returns:
        1-D array of vertical velocities (length = len(positions) - 1).
    """
    velocities = []
    for i in range(1, len(positions)):
        dy_px = positions[i - 1][1] - positions[i][1]  # invert y: up = positive
        dt    = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            velocities.append(0.0)
            continue
        if pixels_per_meter is not None:
            velocities.append((dy_px / pixels_per_meter) / dt)
        else:
            velocities.append(dy_px / dt)
    return np.array(velocities)


# ===========================================================================
# Post-processing pipeline
# ===========================================================================

def calculate_hip_velocity(
    video_path: str,
    pixels_per_meter: Optional[float] = None,
    filter_cutoff: float = 6.0,
    confidence_threshold: float = 0.3,
    model_complexity: int = 2,
) -> Optional[dict]:
    """
    Extract vertical hip velocity from a squat video (post-processing).

    The full processing pipeline (Section 2.3):
      1. Frame-by-frame pose extraction with PoseExtractor
      2. Position tracking with gap interpolation (PositionTracker)
      3. Vertical velocity from finite differences
      4. Zero-phase Butterworth low-pass filter (filtfilt)
      5. Concentric / eccentric phase detection
      6. Per-rep MCV / PCV calculation

    Args:
        video_path        : Path to the input video file.
        pixels_per_meter  : Calibration factor. None → velocities in px/s.
        filter_cutoff     : Butterworth cutoff frequency in Hz (default 6 Hz).
        confidence_threshold: Minimum landmark visibility to accept detection.
        model_complexity  : MediaPipe model (0=Lite, 1=Full, 2=Heavy).

    Returns:
        Results dict, or None on failure.
    """
    extractor = PoseExtractor(model_complexity=model_complexity)
    tracker   = PositionTracker(max_missing_frames=5)

    # Phase threshold: 0.05 m/s if calibrated, 5 px/s if not
    vel_threshold = 0.05 if pixels_per_meter else 5.0
    phase_detector = RepPhaseDetector(velocity_threshold=vel_threshold)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        extractor.close()
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    unit = "m/s" if pixels_per_meter else "px/s"
    print(f"\n{'='*55}")
    print(f"Processing : {video_path}")
    print(f"Resolution : {frame_w}x{frame_h}  |  FPS: {fps:.1f}  |  Frames: {total_frames}")
    print(f"Calibration: {'%.1f px/m' % pixels_per_meter if pixels_per_meter else 'None (px/s)'}")
    print(f"Filter     : Butterworth {filter_cutoff} Hz low-pass")
    print(f"{'='*55}\n")

    frame_count = 0
    print("Extracting pose landmarks…")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps

        landmarks, confidence = extractor.process_frame(frame)

        if landmarks is not None and confidence >= confidence_threshold:
            position, conf = extractor.get_hip_center(landmarks, frame_w, frame_h)
            tracker.update(position, timestamp, conf)
        else:
            tracker.update(None, timestamp, 0.0)
            if frame_count % 30 == 0 or confidence < confidence_threshold:
                print(f"  Frame {frame_count}: No / low-confidence detection")

        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames} …")

    cap.release()
    extractor.close()

    if len(tracker.positions) < 2:
        print("Error: Not enough positions detected. Check video and lighting.")
        return None

    positions  = tracker.positions
    timestamps = np.array(tracker.timestamps)

    # Vertical velocity (raw)
    raw_vel        = _calculate_vertical_velocity(positions, list(timestamps), pixels_per_meter)
    vel_timestamps = timestamps[1:]  # one shorter than positions

    # Butterworth filter (needs ≥ 10 samples for stability)
    if len(raw_vel) >= 10:
        filtered_vel = butterworth_lowpass_filter(raw_vel, filter_cutoff, fps)
    else:
        print("Warning: Too few frames — skipping filter.")
        filtered_vel = raw_vel.copy()

    # Phase detection → per-rep metrics
    concentric_phases = phase_detector.get_concentric_phases(filtered_vel, vel_timestamps)
    reps: List[RepData] = []

    for start_idx, end_idx in concentric_phases:
        if end_idx - start_idx < 5:
            continue
        v_seg = filtered_vel[start_idx:end_idx]
        t_seg = vel_timestamps[start_idx:end_idx]
        reps.append(RepData(
            mean_velocity=VelocityMetrics.mean_velocity(v_seg, t_seg),
            peak_velocity=VelocityMetrics.peak_velocity(v_seg),
            duration=float(t_seg[-1] - t_seg[0]),
            timestamps=t_seg - t_seg[0],
            velocities=v_seg,
        ))

    # Summary statistics
    avg_vel = float(np.mean(filtered_vel))
    max_vel = float(np.max(filtered_vel))
    min_vel = float(np.min(filtered_vel))
    std_vel = float(np.std(filtered_vel))

    print(f"\n{'='*55}")
    print(f"RESULTS  ({unit})")
    print(f"{'='*55}")
    print(f"Frames processed : {frame_count}")
    print(f"Average velocity : {avg_vel:.3f} {unit}")
    print(f"Max velocity     : {max_vel:.3f} {unit}")
    print(f"Min velocity     : {min_vel:.3f} {unit}")
    print(f"Std deviation    : {std_vel:.3f} {unit}")
    print(f"Reps detected    : {len(reps)}")
    for i, rep in enumerate(reps, 1):
        print(
            f"  Rep {i}: MCV={rep.mean_velocity:.3f}  "
            f"PCV={rep.peak_velocity:.3f}  dur={rep.duration:.2f}s"
        )
    print(f"{'='*55}\n")

    return {
        'velocities':        filtered_vel.tolist(),
        'raw_velocities':    raw_vel.tolist(),
        'timestamps':        vel_timestamps.tolist(),
        'frame_numbers':     list(range(2, frame_count + 1)),  # for backward compat
        'reps':              reps,
        'concentric_phases': concentric_phases,
        'avg_velocity':      avg_vel,
        'max_velocity':      max_vel,
        'min_velocity':      min_vel,
        'std_velocity':      std_vel,
        'fps':               fps,
        'total_frames':      frame_count,
        'unit':              unit,
        'pixels_per_meter':  pixels_per_meter,
    }


# ===========================================================================
# Plotting
# ===========================================================================

def plot_velocity(
    results: dict,
    title: str = 'Hip Vertical Velocity – Squat Analysis',
    save_path: str = 'velocity_plot.png',
) -> None:
    """
    Velocity-time graph with concentric phase shading and per-rep bar chart.

    Args:
        results  : Dict returned by calculate_hip_velocity.
        title    : Plot title.
        save_path: Output PNG file path.
    """
    if results is None:
        return

    velocities        = np.array(results['velocities'])
    timestamps        = np.array(results['timestamps'])
    unit              = results.get('unit', 'm/s')
    reps: List[RepData] = results.get('reps', [])
    concentric_phases = results.get('concentric_phases', [])

    fig, (ax_vel, ax_rep) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1]},
    )

    # -- Velocity trace --------------------------------------------------
    ax_vel.plot(timestamps, velocities, 'b-', linewidth=1.5, zorder=3)
    ax_vel.axhline(y=0, color='k', linewidth=0.8, alpha=0.4)
    ax_vel.axhline(
        y=results['avg_velocity'], color='r', linestyle='--', linewidth=1.5,
    )

    for start_idx, end_idx in concentric_phases:
        t0 = timestamps[start_idx]
        t1 = timestamps[min(end_idx, len(timestamps) - 1)]
        ax_vel.axvspan(t0, t1, alpha=0.15, color='green')

    legend_handles = [
        plt.Line2D([0], [0], color='blue', linewidth=1.5,
                   label='Hip velocity (filtered)'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                   label=f"Mean: {results['avg_velocity']:.3f} {unit}"),
    ]
    if concentric_phases:
        legend_handles.append(Patch(facecolor='green', alpha=0.3,
                                    label='Concentric phase'))

    ax_vel.set_xlabel('Time (s)', fontsize=12)
    ax_vel.set_ylabel(f'Velocity ({unit})', fontsize=12)
    ax_vel.set_title(title, fontsize=14, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(handles=legend_handles, fontsize=10)

    # -- Per-rep bar chart -----------------------------------------------
    if reps:
        rep_nums = list(range(1, len(reps) + 1))
        mcvs     = [r.mean_velocity for r in reps]
        pcvs     = [r.peak_velocity for r in reps]
        x        = np.arange(len(rep_nums))
        width    = 0.35

        ax_rep.bar(x - width / 2, mcvs, width, label='MCV',
                   color='steelblue', alpha=0.85)
        ax_rep.bar(x + width / 2, pcvs, width, label='PCV',
                   color='darkorange', alpha=0.85)
        ax_rep.set_xticks(x)
        ax_rep.set_xticklabels([f'Rep {n}' for n in rep_nums])
        ax_rep.set_ylabel(f'Velocity ({unit})', fontsize=10)
        ax_rep.set_title('Per-Rep Metrics (MCV & PCV)', fontsize=11)
        ax_rep.legend(fontsize=9)
        ax_rep.grid(True, alpha=0.3, axis='y')
    else:
        ax_rep.text(0.5, 0.5, 'No reps detected — adjust velocity threshold',
                    ha='center', va='center', transform=ax_rep.transAxes,
                    fontsize=11, color='grey')
        ax_rep.set_title('Per-Rep Metrics', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.show()


# ===========================================================================
# Pose visualisation (annotated video output)
# ===========================================================================

def visualise_pose(
    video_path: str,
    output_path: str = 'output_with_pose.mp4',
    model_complexity: int = 1,
) -> None:
    """
    Write a copy of the video with the MediaPipe skeleton overlaid.

    Args:
        video_path      : Input video file.
        output_path     : Output annotated video file.
        model_complexity: MediaPipe model complexity (1 is fast enough here).
    """
    extractor = PoseExtractor(model_complexity=model_complexity)
    cap = cv2.VideoCapture(video_path)

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Writing annotated video → {output_path}")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        extractor.process_frame(frame)   # updates extractor.pose_landmarks
        extractor.draw_pose(frame)       # draws skeleton in-place
        out.write(frame)

    cap.release()
    out.release()
    extractor.close()
    print(f"Done. Wrote {frame_count} frames → {output_path}")


# ===========================================================================
# Real-time tracker (live camera, Section 10)
# ===========================================================================

class SquatVelocityTracker:
    """
    Real-time barbell velocity tracker using a live camera feed (Section 13.1).

    Usage::

        tracker = SquatVelocityTracker(camera_id=0, pixels_per_meter=300)
        tracker.run()            # blocking; press 'q' to quit, 'c' to calibrate
        print(tracker.get_session_summary())
    """

    def __init__(
        self,
        camera_id: int = 0,
        pixels_per_meter: float = 500.0,
        target_fps: int = 60,
        filter_cutoff: float = 6.0,
    ):
        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FPS,          target_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Components
        self.extractor     = PoseExtractor(model_complexity=1)
        self.phase_detector = RepPhaseDetector(velocity_threshold=0.05)

        # Calibration
        self.pixels_per_meter = pixels_per_meter
        self.sample_rate      = target_fps
        self.filter_cutoff    = filter_cutoff

        # Data
        self.positions:  List[np.ndarray] = []
        self.timestamps: List[float]      = []
        self.velocities: List[float]      = []
        self.reps:       List[RepData]    = []

        self.current_phase = 'stationary'
        self.rep_start_idx: Optional[int] = None

        # Real-time filter state
        self._init_realtime_filter(filter_cutoff, target_fps)

    # ------------------------------------------------------------------
    # Real-time filter initialisation
    # ------------------------------------------------------------------

    def _init_realtime_filter(self, cutoff: float, sample_rate: int) -> None:
        nyquist           = sample_rate / 2.0
        normalized_cutoff = min(cutoff / nyquist, 0.99)
        self._b, self._a  = butter(2, normalized_cutoff, btype='low')
        self._filt_state  = np.zeros(max(len(self._a), len(self._b)) - 1)

    def _filter_sample(self, sample: float) -> float:
        """Process one sample through a stateful causal filter."""
        filtered, self._filt_state = lfilter(
            self._b, self._a, [sample], zi=self._filt_state
        )
        return float(filtered[0])

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, known_distance_meters: float = 2.2) -> None:
        """
        Interactive calibration — click two points of known distance.

        Args:
            known_distance_meters: Real-world distance between the two clicked
                                   points (default: Olympic barbell = 2.2 m).
        """
        print("=== CALIBRATION ===")
        print(f"Click two points that are {known_distance_meters} m apart.")
        print("Press any key once both points are selected.")

        points: List[Tuple[int, int]] = []

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                print(f"  Point {len(points)}: ({x}, {y})")

        ret, frame = self.cap.read()
        if not ret:
            print("Could not capture frame for calibration.")
            return

        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', on_click)

        while len(points) < 2:
            display = frame.copy()
            for p in points:
                cv2.circle(display, p, 6, (0, 255, 0), -1)
            cv2.imshow('Calibration', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow('Calibration')

        if len(points) == 2:
            px_dist = float(np.linalg.norm(
                np.array(points[1]) - np.array(points[0])
            ))
            self.pixels_per_meter = px_dist / known_distance_meters
            print(f"Calibration: {self.pixels_per_meter:.1f} px/m")
        else:
            print("Calibration cancelled.")

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _get_hip_center(
        self, landmarks, frame_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, float]:
        h, w = frame_shape[:2]
        return self.extractor.get_hip_center(landmarks, w, h)

    def _compute_velocity(self) -> Optional[float]:
        if len(self.positions) < 2:
            return None
        dy_px = self.positions[-2][1] - self.positions[-1][1]   # up = positive
        dy_m  = dy_px / self.pixels_per_meter
        dt    = self.timestamps[-1] - self.timestamps[-2]
        return dy_m / dt if dt > 0 else None

    def _filter_velocities(self, velocities: np.ndarray) -> np.ndarray:
        if len(velocities) < 10:
            return velocities
        nyquist = self.sample_rate / 2.0
        norm_cut = min(self.filter_cutoff / nyquist, 0.99)
        b, a = butter(2, norm_cut, btype='low')
        return filtfilt(b, a, velocities)

    def _process_rep_completion(self, end_idx: int) -> None:
        if self.rep_start_idx is None:
            return
        start = self.rep_start_idx
        if end_idx - start < 5:
            return

        t_arr = np.array(self.timestamps[start:end_idx])
        v_arr = np.array(self.velocities[start:end_idx])
        v_filt = self._filter_velocities(v_arr)

        rep = RepData(
            mean_velocity=VelocityMetrics.mean_velocity(v_filt, t_arr),
            peak_velocity=VelocityMetrics.peak_velocity(v_filt),
            duration=float(t_arr[-1] - t_arr[0]),
            timestamps=t_arr - t_arr[0],
            velocities=v_filt,
        )
        self.reps.append(rep)
        print(
            f"Rep {len(self.reps)}: "
            f"MCV={rep.mean_velocity:.2f} m/s  "
            f"PCV={rep.peak_velocity:.2f} m/s"
        )

    def process_frame(self, frame: np.ndarray, timestamp: float) -> dict:
        """
        Process one live frame and return current tracking state.

        Returns:
            dict with keys: position, velocity, phase, confidence.
        """
        landmarks, confidence = self.extractor.process_frame(frame)

        output = {
            'position':   None,
            'velocity':   None,
            'phase':      self.current_phase,
            'confidence': confidence,
        }

        if landmarks is not None and confidence > 0.3:
            position, conf = self._get_hip_center(landmarks, frame.shape)
            output['position']   = position
            output['confidence'] = conf
            self.positions.append(position)
            self.timestamps.append(timestamp)

            raw_velocity = self._compute_velocity()
            if raw_velocity is not None:
                filtered_velocity = self._filter_sample(raw_velocity)
                self.velocities.append(filtered_velocity)
                output['velocity'] = filtered_velocity

                new_phase = self.phase_detector.update(filtered_velocity)

                if self.current_phase == 'eccentric' and new_phase == 'concentric':
                    self.rep_start_idx = len(self.velocities) - 1
                elif self.current_phase == 'concentric' and new_phase != 'concentric':
                    self._process_rep_completion(len(self.velocities))
                    self.rep_start_idx = None

                self.current_phase = new_phase
                output['phase'] = new_phase

        return output

    # ------------------------------------------------------------------
    # Display overlay
    # ------------------------------------------------------------------

    def draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Render velocity, phase, and rep info onto *frame*."""
        h, w = frame.shape[:2]

        self.extractor.draw_pose(frame)

        velocity = result.get('velocity')
        phase    = result.get('phase', 'stationary')

        # Velocity text
        if velocity is not None:
            color    = (0, 255, 0) if velocity > 0 else (0, 0, 255)
            cv2.putText(frame, f"Vel: {velocity:.2f} m/s",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        # Phase text
        phase_colors = {
            'concentric': (0, 255, 0),
            'eccentric':  (0, 0, 255),
            'stationary': (255, 255, 0),
        }
        cv2.putText(frame, phase.upper(),
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    phase_colors.get(phase, (255, 255, 255)), 3)

        # Rep counter
        cv2.putText(frame, f"Reps: {len(self.reps)}",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Last rep metrics
        if self.reps:
            last = self.reps[-1]
            cv2.putText(frame, f"Last MCV: {last.mean_velocity:.2f} m/s",
                        (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"Last PCV: {last.peak_velocity:.2f} m/s",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Velocity bar (right edge)
        if velocity is not None:
            bar_x      = w - 80
            bar_y      = h // 2
            bar_height = int(min(abs(velocity) * 200, 400))
            bar_color  = (0, 255, 0) if velocity > 0 else (0, 0, 255)
            if velocity > 0:
                cv2.rectangle(frame,
                              (bar_x, bar_y - bar_height),
                              (bar_x + 50, bar_y),
                              bar_color, -1)
            else:
                cv2.rectangle(frame,
                              (bar_x, bar_y),
                              (bar_x + 50, bar_y + bar_height),
                              bar_color, -1)

        return frame

    # ------------------------------------------------------------------
    # Main loop

    def run(self) -> None:
        """
        Start real-time tracking loop.

        Keys:  'c' = calibrate,  'q' = quit.
        """
        print("Starting real-time velocity tracker…")
        print("Press 'c' to calibrate, 'q' to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = time.perf_counter()
            result    = self.process_frame(frame, timestamp)
            display   = self.draw_overlay(frame.copy(), result)

            cv2.imshow('Squat Velocity Tracker', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibrate()

        self.cleanup()

    def cleanup(self) -> None:
        """Release all resources."""
        self.cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()

    def get_session_summary(self) -> dict:
        """Return aggregate statistics for the session."""
        if not self.reps:
            return {}
        mcvs = [r.mean_velocity for r in self.reps]
        pcvs = [r.peak_velocity for r in self.reps]
        return {
            'total_reps': len(self.reps),
            'mean_mcv':   float(np.mean(mcvs)),
            'std_mcv':    float(np.std(mcvs)),
            'mean_pcv':   float(np.mean(pcvs)),
            'std_pcv':    float(np.std(pcvs)),
            'velocity_loss_pct': (
                (mcvs[0] - mcvs[-1]) / mcvs[0] * 100
                if len(mcvs) > 1 else 0.0
            ),
        }


# ===========================================================================
# Legacy entry-point (kept for backward compatibility)
# ===========================================================================

def main():
    video_path = "Front-View Squat - 30fps Trim.mp4"

    print("\n" + "=" * 60)
    print("FRONT/SIDE-VIEW VELOCITY TRACKER")
    print("Barbell Back Squat Analysis using MediaPipe BlazePose")
    print("=" * 60)

    results = calculate_hip_velocity(video_path)

    if results is not None:
        plot_velocity(results)
        visualise_pose(video_path)
        print("\nProcessing complete!")
    else:
        print("\nProcessing failed. Check video path and lighting.")


if __name__ == "__main__":
    main()
