"""Interfaces et affichage des cartes de transmission du nuller.

Contient des widgets et fonctions pour calculer/afficher les cartes de
transmission, gradients et exporter des images.
"""
from copy import deepcopy as copy
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from phise import Context
try:
    from src.analysis.io_utils import save_figure, save_dataset, get_archive
except ImportError:
    from io_utils import save_figure, save_dataset, get_archive

def plot(ctx: Context=None, N: int=100, grad: bool=False, save_as=None):
    """Plot and save transmission maps with dual figure saving and archiving."""
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.um
    else:
        ctx = copy(ctx)

    fig = plt.figure(figsize=(15, 6))
    ctx.plot_transmission_maps(N=N, return_plot=False, grad=grad)
    current_fig = plt.gcf()
    
    if save_as:
        tag = "transmission_maps_grad" if grad else "transmission_maps"
        save_figure(current_fig, tag, save_as, analysis_name="transmission_maps")
        raw_maps, proc_maps = ctx.get_transmission_maps(N=N)
        save_dataset({
            "raw_maps": raw_maps,
            "processed_maps": proc_maps,
            "N": N,
        }, f"{tag}_data", save_as=save_as, analysis_name="transmission_maps")

def gui(ctx: Context=None, N: int=100, save_as=None):
    """
    GUI to visualize the transmission maps of the VLTI.
    """
    if ctx is None:
        ref_ctx = Context.get_VLTI()
        ref_ctx.interferometer.chip.σ = np.zeros(14) * u.um
    else:
        ref_ctx = ctx

    h_slider = widgets.FloatSlider(value=0, min=(ref_ctx.h - ref_ctx.Δh / 2).value, max=(ref_ctx.h + ref_ctx.Δh / 2).value, step=0.01, description='Hour angle:')
    l_slider = widgets.FloatSlider(value=ref_ctx.interferometer.l.to(u.deg).value, min=-90, max=90, step=0.01, description='Latitude:')
    δ_slider = widgets.FloatSlider(value=ref_ctx.target.δ.to(u.deg).value, min=-90, max=90, step=0.01, description='Declination:')
    reset = widgets.Button(description='Reset values')
    run_btn = widgets.Button(description='Run')
    export = widgets.Button(description='Export')
    plot_img = widgets.Image()
    plot_gradient = widgets.Image()
    transmission = widgets.HTML()

    def update_plot(*args):
        run_btn.button_style = 'warning'
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        ctx.h = h_slider.value * u.hourangle
        (img, txt) = ctx.plot_transmission_maps(N=N, return_plot=True)
        plot_img.value = img
        transmission.value = txt
        (img, txt) = ctx.plot_transmission_maps(N=N, return_plot=True, grad=True)
        plot_gradient.value = img
        run_btn.button_style = ''

    def export_plot(*args):
        ctx = copy(ref_ctx)
        ctx.interferometer.l = l_slider.value * u.deg
        ctx.target.δ = δ_slider.value * u.deg
        ctx.h = h_slider.value * u.hourangle
        plot(ctx, N=N, save_as=save_as)

    def reset_values(*args):
        l_slider.value = ref_ctx.interferometer.l.to(u.deg).value
        δ_slider.value = ref_ctx.target.δ.to(u.deg).value
        h_slider.value = ref_ctx.h.to(u.deg).value
        run_btn.color = 'blue'
        enable_run()

    def enable_run(*args):
        run_btn.button_style = 'success'

    reset.on_click(reset_values)
    h_slider.observe(enable_run)
    l_slider.observe(enable_run)
    δ_slider.observe(enable_run)
    run_btn.on_click(update_plot)
    export.on_click(export_plot)
    display(widgets.VBox([h_slider, l_slider, δ_slider, widgets.HBox([reset, run_btn, export]), plot_img, transmission, plot_gradient]))
    update_plot()

def run(save_as: str = "archives"):
    """Standalone runner for transmission maps analysis."""
    print("Running transmission_maps analysis...")
    ctx = Context.get_VLTI()
    plot(ctx, N=50, save_as=save_as)
    print("Done transmission_maps.")

if __name__ == "__main__":
    run()