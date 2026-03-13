"""
Utils Module
Calibration helpers, validation metrics, and Bland-Altman analysis
for the HPE velocity tracking system.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ===========================================================================
# Spatial calibration (Section 6.2)
# ===========================================================================

def calculate_scale(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    known_distance_meters: float,
) -> float:
    """
    Calculate pixels per metre from two image points and a known distance.

    Args:
        point1               : (x, y) pixel coordinates of the first point.
        point2               : (x, y) pixel coordinates of the second point.
        known_distance_meters: Real-world distance between the two points (m).
                               Standard Olympic barbell = 2.2 m.

    Returns:
        pixels_per_meter: Conversion factor (pixels / metre).

    Example::

        # Barbell endpoints in the calibration frame
        scale = calculate_scale((120, 540), (1800, 540), known_distance_meters=2.2)
    """
    pixel_distance = float(np.sqrt(
        (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2
    ))
    if pixel_distance == 0:
        raise ValueError("The two calibration points must not be identical.")
    pixels_per_meter = pixel_distance / known_distance_meters
    return pixels_per_meter


# ===========================================================================
# Validation metrics (Section 11.2)
# ===========================================================================

def calculate_validation_metrics(
    camera_values: List[float],
    reference_values: List[float],
) -> dict:
    """
    Compute error statistics comparing camera-based measurements to a
    reference device (e.g. GymAware, linear position transducer).

    Metrics returned:
        MAE    – Mean Absolute Error
        RMSE   – Root Mean Square Error
        Bias   – Systematic error (mean difference)
        SD     – Standard deviation of errors
        LoA_lower / LoA_upper – 95 % Bland-Altman limits of agreement
        r      – Pearson correlation coefficient
        TEE    – Typical Error of Estimate
        CV_pct – Coefficient of Variation (%)

    Args:
        camera_values   : Velocities measured by the camera system.
        reference_values: Velocities from the gold-standard device.

    Returns:
        dict of metric name → value.
    """
    cam = np.array(camera_values, dtype=float)
    ref = np.array(reference_values, dtype=float)

    if len(cam) != len(ref):
        raise ValueError("camera_values and reference_values must be the same length.")
    if len(cam) < 2:
        raise ValueError("At least two paired measurements are required.")

    errors = cam - ref

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    sd   = float(np.std(errors, ddof=1))

    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd

    r, _  = stats.pearsonr(cam, ref)
    tee   = sd / np.sqrt(2)
    cv    = (sd / float(np.mean(ref))) * 100 if np.mean(ref) != 0 else float('nan')

    return {
        'MAE':       mae,
        'RMSE':      rmse,
        'Bias':      bias,
        'SD':        sd,
        'LoA_lower': loa_lower,
        'LoA_upper': loa_upper,
        'r':         float(r),
        'TEE':       float(tee),
        'CV_pct':    float(cv),
        'n':         len(cam),
    }


def print_validation_report(metrics: dict, unit: str = 'm/s') -> None:
    """Print a formatted validation summary to stdout."""
    print(f"\n{'='*50}")
    print("VALIDATION METRICS")
    print(f"{'='*50}")
    print(f"  n                  : {metrics['n']}")
    print(f"  Mean Abs Error     : {metrics['MAE']:.4f} {unit}")
    print(f"  RMSE               : {metrics['RMSE']:.4f} {unit}")
    print(f"  Bias               : {metrics['Bias']:.4f} {unit}")
    print(f"  SD of errors       : {metrics['SD']:.4f} {unit}")
    print(f"  Limits of Agreement: [{metrics['LoA_lower']:.4f}, {metrics['LoA_upper']:.4f}] {unit}")
    print(f"  Pearson r          : {metrics['r']:.4f}")
    print(f"  Typical Error      : {metrics['TEE']:.4f} {unit}")
    print(f"  CV%                : {metrics['CV_pct']:.2f}%")
    print(f"{'='*50}\n")


# ===========================================================================
# Bland-Altman analysis (Section 11.3)
# ===========================================================================

def bland_altman_plot(
    camera_values: List[float],
    reference_values: List[float],
    title: str = 'Bland-Altman Plot',
    unit: str = 'm/s',
    save_path: Optional[str] = 'bland_altman.png',
) -> plt.Figure:
    """
    Generate a Bland-Altman plot for method comparison.

    Plots the difference (camera − reference) against the mean of both
    methods. The mean bias and ±1.96 SD limits of agreement are marked.

    Args:
        camera_values   : Camera-system velocity measurements.
        reference_values: Reference-device velocity measurements.
        title           : Plot title.
        unit            : Velocity unit label (e.g. 'm/s').
        save_path       : PNG output path (None = do not save).

    Returns:
        Matplotlib Figure object.
    """
    cam = np.array(camera_values, dtype=float)
    ref = np.array(reference_values, dtype=float)

    means       = (cam + ref) / 2.0
    differences = cam - ref

    mean_diff = float(np.mean(differences))
    std_diff  = float(np.std(differences, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(means, differences, alpha=0.6, color='steelblue', edgecolors='white', s=50)

    ax.axhline(mean_diff,  color='red',   linestyle='-',  linewidth=1.8,
               label=f'Mean bias: {mean_diff:.3f} {unit}')
    ax.axhline(loa_upper,  color='red',   linestyle='--', linewidth=1.4,
               label=f'+1.96 SD: {loa_upper:.3f} {unit}')
    ax.axhline(loa_lower,  color='red',   linestyle='--', linewidth=1.4,
               label=f'−1.96 SD: {loa_lower:.3f} {unit}')
    ax.axhline(0.0,        color='black', linestyle=':',  linewidth=0.8, alpha=0.5)

    ax.set_xlabel(f'Mean of Methods ({unit})', fontsize=12)
    ax.set_ylabel(f'Difference — Camera minus Reference ({unit})', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bland-Altman plot saved: {save_path}")

    return fig


# ===========================================================================
# Frame timing utility (Section 6.3)
# ===========================================================================

class FrameTimer:
    """Records per-frame timestamps for accurate dt calculation."""

    def __init__(self):
        self.timestamps: List[float] = []

    def capture(self) -> float:
        """Record and return the current timestamp (seconds)."""
        import time
        ts = time.perf_counter()
        self.timestamps.append(ts)
        return ts

    def get_dt(self, frame_idx: int) -> Optional[float]:
        """Time difference between frame_idx and the previous frame."""
        if frame_idx < 1 or frame_idx >= len(self.timestamps):
            return None
        return self.timestamps[frame_idx] - self.timestamps[frame_idx - 1]

    def get_actual_fps(self) -> Optional[float]:
        """Actual achieved frame rate over all recorded frames."""
        if len(self.timestamps) < 2:
            return None
        total_time = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / total_time if total_time > 0 else None
