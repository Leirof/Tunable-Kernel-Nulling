"""Bootstrap study of crosstalk-limited null depth for the PHOB N4x4-T8."""

from __future__ import annotations

from typing import Iterable

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from pathlib import Path

try:
    from ..io_utils import get_archive, save_figure, save_dataset
except ImportError:
    from src.analysis.io_utils import get_archive, save_figure, save_dataset

try:
    from .utils import (
        CurveStatistics,
        make_phob_context,
        random_energy_conserving_matrix,
        statistics,
        validate_energy_conservation,
    )
except ImportError:
    from utils import (
        CurveStatistics,
        make_phob_context,
        random_energy_conserving_matrix,
        statistics,
        validate_energy_conservation,
    )



def _null_depth(ctx) -> float:
    """Measure the perfect-camera null depth from the raw output fluxes."""
    outputs = ctx.observe(upstream_pistons=np.zeros(4) * u.nm)
    if outputs[0] <= 0:
        return np.nan
    return float(outputs[3] / outputs[0])


def simulate_crosstalk_curve(
    crosstalk_levels: Iterable[float],
    bootstrap_samples: int = 100,
    seed: int | None = 0,
    beta: float = 0.9,
    show_progress: bool = True,
) -> tuple[np.ndarray, CurveStatistics, CurveStatistics]:
    """Simulate null-depth distributions before and after calibration.

    Args:
        crosstalk_levels: Maximum off-diagonal field coefficients as fractions.
        bootstrap_samples: Number of independent matrix pairs per level.
        seed: Seed for reproducible bootstrap matrices, or ``None``.
        beta: Hooke&Jeeves step reduction factor.
        show_progress: Display nested tqdm progress bars while computing.

    Returns:
        Tuple containing the levels and statistics for the uncalibrated and
        calibrated cases, respectively.
    """
    levels = np.asarray(tuple(crosstalk_levels), dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("crosstalk_levels must be a non-empty one-dimensional sequence")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    rng = np.random.default_rng(seed)
    before = np.empty((levels.size, bootstrap_samples))
    after = np.empty_like(before)

    level_iterator = tqdm(
        enumerate(levels),
        total=levels.size,
        desc="Crosstalk levels",
        unit="level",
        disable=not show_progress,
    )
    for level_index, level in level_iterator:
        sample_iterator = tqdm(
            range(bootstrap_samples),
            desc=f"{level * 100:g}%",
            unit="matrix",
            leave=False,
            position=1,
            disable=not show_progress,
        )
        for sample_index in sample_iterator:
            cin = random_energy_conserving_matrix(level, rng)
            cout = random_energy_conserving_matrix(level, rng)
            if not validate_energy_conservation(cin) or not validate_energy_conservation(cout):
                raise RuntimeError("Generated crosstalk matrix is not energy conserving")

            context = make_phob_context(cin, cout)
            before[level_index, sample_index] = _null_depth(context)
            context.interferometer.chip.calibrate(method="Hooke&Jeeves", β=beta)
            after[level_index, sample_index] = _null_depth(context)

    return levels, statistics(before, axis=1), statistics(after, axis=1)


def plot_crosstalk_curve(
    levels: np.ndarray,
    before: CurveStatistics,
    after: CurveStatistics,
    ax=None,
    save_as=None,
):
    """Plot mean, median, percentiles, and extrema on a logarithmic null axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    else:
        fig = ax.figure

    for statistics, color, label in (
        (before, "tab:blue", "Before calibration"),
        (after, "tab:orange", "After Hooke&Jeeves calibration"),
    ):
        floor = np.finfo(float).tiny
        ax.fill_between(levels * 100, np.maximum(statistics.minimum, floor), np.maximum(statistics.maximum, floor), color=color, alpha=0.18)
        ax.plot(levels * 100, np.maximum(statistics.mean, floor), color=color, label=f"{label} mean")
        ax.plot(levels * 100, np.maximum(statistics.median, floor), color=color, linestyle="--", label=f"{label} median")
        ax.plot(levels * 100, np.maximum(statistics.percentile_05, floor), color=color, linestyle=":")
        ax.plot(levels * 100, np.maximum(statistics.percentile_95, floor), color=color, linestyle=":")

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Maximum off-diagonal crosstalk coefficient (%)")
    ax.set_ylabel("Null depth")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    if save_as:
        save_figure(fig, save_as, default_name="crosstalk_vs_null")
    return ax


if __name__ == "__main__":
    print("=== Running Crosstalk vs Null Analysis Standalone ===")
    arc = get_archive(Path(__file__).parent, name="crosstalk_vs_null")
    levels = np.geomspace(1e-6, 1e-1, 8)
    levels, before, after = simulate_crosstalk_curve(levels, bootstrap_samples=10)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    plot_crosstalk_curve(levels, before, after, ax=ax)
    save_figure(fig, arc.path / "crosstalk_vs_null")
    save_dataset(
        arc,
        "crosstalk_curve_data",
        levels=levels,
        before_mean=before.mean,
        before_median=before.median,
        after_mean=after.mean,
        after_median=after.median,
    )
    plt.show()
    print(f"Archived results to {arc.path}")

