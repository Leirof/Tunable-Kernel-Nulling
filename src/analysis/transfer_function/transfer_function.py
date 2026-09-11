"""Transfer function analysis for the 4-telescope kernel-nuller architecture.

Provides analytical (SymPy) and numerical transfer matrices for the chip layers:
nulling stage, phase shifters, recombiners, and kernel extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

try:
    from ..io_utils import get_archive, save_dataset, save_figure
except ImportError:
    from src.analysis.io_utils import get_archive, save_dataset, save_figure


def build_symbolic_transfer_matrix():
    """Build the analytical SymPy transfer matrix M such that out_fields = M * in_fields."""
    p = sp.IndexedBase('phi', real=True)
    e = sp.IndexedBase('sigma', real=True)

    p_1_4 = sp.Matrix([
        [sp.exp(sp.I * (e[1] + p[1])), 0, 0, 0],
        [0, sp.exp(sp.I * (e[2] + p[2])), 0, 0],
        [0, 0, sp.exp(sp.I * (e[3] + p[3])), 0],
        [0, 0, 0, sp.exp(sp.I * (e[4] + p[4]))]
    ])

    Nlayer = 1/sp.sqrt(2) * sp.Matrix([
        [1,  1,  0,  0],
        [1, -1,  0,  0],
        [0,  0,  1,  1],
        [0,  0,  1, -1]
    ])

    invert_2_3 = sp.Matrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    p_5_8 = sp.Matrix([
        [sp.exp(sp.I * (e[5] + p[5])), 0, 0, 0],
        [0, sp.exp(sp.I * (e[6] + p[6])), 0, 0],
        [0, 0, sp.exp(sp.I * (e[7] + p[7])), 0],
        [0, 0, 0, sp.exp(sp.I * (e[8] + p[8]))]
    ])

    splitters = 1/sp.sqrt(2) * sp.Matrix([
        [sp.sqrt(2), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ])

    p_9_14 = sp.Matrix([
        [1, 0, 0, 0, 0, 0, 0],
        [0, sp.exp(sp.I * (e[9] + p[9])), 0, 0, 0, 0, 0],
        [0, 0, sp.exp(sp.I * (e[10] + p[10])), 0, 0, 0, 0],
        [0, 0, 0, sp.exp(sp.I * (e[11] + p[11])), 0, 0, 0],
        [0, 0, 0, 0, sp.exp(sp.I * (e[12] + p[12])), 0, 0],
        [0, 0, 0, 0, 0, sp.exp(sp.I * (e[13] + p[13])), 0],
        [0, 0, 0, 0, 0, 0, sp.exp(sp.I * (e[14] + p[14]))],
    ])

    invert_23_45 = sp.Matrix([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ])

    theta = sp.pi / 4
    Slayer = 1/sp.sqrt(2) * sp.Matrix([
        [sp.sqrt(2), 0, 0, 0, 0, 0, 0],
        [0, sp.exp(sp.I * theta), sp.exp(-sp.I * theta), 0, 0, 0, 0],
        [0, sp.exp(-sp.I * theta), sp.exp(sp.I * theta), 0, 0, 0, 0],
        [0, 0, 0, sp.exp(sp.I * theta), sp.exp(-sp.I * theta), 0, 0],
        [0, 0, 0, sp.exp(-sp.I * theta), sp.exp(sp.I * theta), 0, 0],
        [0, 0, 0, 0, 0, sp.exp(sp.I * theta), sp.exp(-sp.I * theta)],
        [0, 0, 0, 0, 0, sp.exp(-sp.I * theta), sp.exp(sp.I * theta)]
    ])

    # Overall transfer matrix: 7 outputs x 4 inputs
    M = Slayer * invert_23_45 * p_9_14 * splitters * Nlayer * invert_2_3 * p_5_8 * Nlayer * p_1_4
    return M


def plot_ideal_transfer_amplitudes(figsize=(8, 6), save_as=None):
    """Compute and plot the ideal transmission coefficients from 4 inputs to 7 outputs."""
    # Ideal numerical matrix (all phases = 0)
    N1 = 1 / np.sqrt(2) * np.array([
        [1,  1,  0,  0],
        [1, -1,  0,  0],
        [0,  0,  1,  1],
        [0,  0,  1, -1]
    ], dtype=complex)
    P23 = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    N = N1 @ P23 @ N1

    splitters = 1 / np.sqrt(2) * np.array([
        [np.sqrt(2), 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]
    ], dtype=complex)

    P_split = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=complex)

    theta = np.pi / 4
    Slayer = 1 / np.sqrt(2) * np.array([
        [np.sqrt(2), 0, 0, 0, 0, 0, 0],
        [0, np.exp(1j * theta), np.exp(-1j * theta), 0, 0, 0, 0],
        [0, np.exp(-1j * theta), np.exp(1j * theta), 0, 0, 0, 0],
        [0, 0, 0, np.exp(1j * theta), np.exp(-1j * theta), 0, 0],
        [0, 0, 0, np.exp(-1j * theta), np.exp(1j * theta), 0, 0],
        [0, 0, 0, 0, 0, np.exp(1j * theta), np.exp(-1j * theta)],
        [0, 0, 0, 0, 0, np.exp(-1j * theta), np.exp(1j * theta)]
    ], dtype=complex)

    M = Slayer @ P_split @ splitters @ N
    intensities = np.abs(M) ** 2

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(intensities, cmap="viridis", aspect="auto")
    fig.colorbar(cax, ax=ax, label="Power Transmission")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Input {i+1}" for i in range(4)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Bright", "Dark 1", "Dark 2", "Dark 3", "Dark 4", "Dark 5", "Dark 6"])
    ax.set_title("Ideal 4-Telescope Kernel-Nuller Transfer Matrix")
    
    for i in range(7):
        for j in range(4):
            ax.text(j, i, f"{intensities[i, j]:.3f}", ha="center", va="center", color="white" if intensities[i, j] < 0.3 else "black")

    if save_as:
        save_figure(fig, save_as, default_name="transfer_matrix")

    return fig, intensities


def run(save_as=None):
    """Standalone runner that generates and archives transfer matrix analysis."""
    arc = get_archive(Path(__file__).parent, name="transfer_function")
    fig, intensities = plot_ideal_transfer_amplitudes(save_as=arc.path / "transfer_matrix")
    save_dataset(arc, "transfer_matrix", intensities=intensities)
    if save_as:
        save_figure(fig, save_as, default_name="transfer_matrix")
    plt.close(fig)
    print(f"Transfer function analysis archived to {arc.path}")
    return intensities


if __name__ == "__main__":
    run()
