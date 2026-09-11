"""Outils de visualisation pour positions projetées des télescopes.

Contient une interface graphique minimale pour afficher la géométrie
du réseau de télescopes projetée sur le plan du ciel.
"""
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from ipywidgets import widgets
from IPython.display import display
from io import BytesIO
from copy import deepcopy as copy
from phise import Telescope
from phise import Context

from phise.modules import utils
try:
    from src.analysis.io_utils import save_figure, save_dataset, get_archive
except ImportError:
    from io_utils import save_figure, save_dataset, get_archive

def plot_positions(ctx: Context=None, n=11, save_as=None):
    """Plot projected telescope positions over time, saving figures and datasets."""
    if ctx is None:
        ctx = Context.get_VLTI()
    else:
        ctx = copy(ctx)

    fig, ax = plt.subplots(figsize=(6, 6))
    h_range = np.linspace(ctx.h - ctx.Δh / 2, ctx.h + ctx.Δh / 2, n, endpoint=True)
    positions = []
    for i, h in enumerate(h_range):
        ctx_h = copy(ctx)
        ctx_h.h = h
        positions.append(ctx_h.p.to(u.m).value)
        for j, (x, y) in enumerate(ctx_h.p):
            ax.scatter(x, y, label=f"Telescope {j+1}" if i == len(h_range) - 1 else None, color=f"C{j}", s=1 + 14 * i / len(h_range))

    for (x, y) in ctx.p:
        ax.scatter(x, y, color="black", marker="+")

    ax.set_aspect("equal")
    ax.set_xlabel(f"x [{ctx.p.unit}]")
    ax.set_ylabel(f"y [{ctx.p.unit}]")
    ax.set_title(f"Projected telescope positions over time ({ctx.Δh.to(u.hourangle).value * u.h} long)")
    ax.legend()
    if save_as:
        save_figure(fig, "projected_telescopes", save_as, analysis_name="projected_telescopes")
        save_dataset({
            "h_range": h_range.to(u.hourangle).value,
            "positions": np.array(positions),
        }, "projected_positions_data", save_as=save_as, analysis_name="projected_telescopes")
    plt.show()
    return fig, ax

def gui(ctx: Context=None, n=10, save_as=None) -> None:
    """
    GUI to visualize the projected positions of the telescopes in the array.

    Parameters
    ----------
    ctx: Context
        Context object containing the interferometer and target information.
        If None, a default context is used.
    n: int
        Number of telescopes in the interferometer. Default is 10.
    save_as: str
        Path to save the initial plot.
    """
    if ctx is None:
        ref_ctx = Context.get_VLTI()
    else:
        ref_ctx = copy(ctx)

    l_slider = widgets.FloatSlider(value=ref_ctx.interferometer.l.to(u.deg).value, min=-90, max=90, step=0.01, description='Latitude (deg):')
    δ_slider = widgets.FloatSlider(value=ref_ctx.target.δ.to(u.deg).value, min=-90, max=90, step=0.01, description='Declination (deg):')
    reset = widgets.Button(description='Reset to default')
    export = widgets.Button(description='Export')
    plot = widgets.Image(width=500, height=500)

    def update_plot(*_):
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        plot.value = ctx.plot_projected_positions(N=n, return_image=True)

    def reset_values(*_):
        l_slider.value = ref_ctx.interferometer.l.to(u.deg).value
        δ_slider.value = ref_ctx.target.δ.to(u.deg).value
    
    def export_plot(*_):
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        plot_positions(ctx, n=n, save_as=save_as)

    reset.on_click(reset_values)
    export.on_click(export_plot)
    l_slider.observe(update_plot, 'value')
    δ_slider.observe(update_plot, 'value')
    display(widgets.VBox([l_slider, δ_slider, widgets.HBox([reset, export]), plot]))
    update_plot()
    return

def run(save_as: str = "archives"):
    """Standalone runner for projected telescope positions analysis."""
    print("Running projected_telescopes analysis...")
    ctx = Context.get_VLTI()
    plot_positions(ctx, n=15, save_as=save_as)
    print("Done projected_telescopes.")

if __name__ == "__main__":
    run()