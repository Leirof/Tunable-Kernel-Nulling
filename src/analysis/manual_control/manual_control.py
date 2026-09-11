"""Interface interactive pour contrôle manuel et visualisation.

Ce module expose des interfaces Jupyter/IPython (widgets) pour tester
manuellement des configurations d'entrée, observer les sorties du
nuller et visualiser phases, intensités et cartes.
"""
import os
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
try:
    plt.rcParams['image.origin'] = 'lower'
except Exception:
    pass
import ipywidgets as widgets
import astropy.units as u
from copy import deepcopy as copy
from phise.classes import Context
from phise.classes import Companion, Target, Telescope, SuperKN, Interferometer, Camera
from phise.modules import *
try:
    from src.analysis.io_utils import save_figure, save_dataset, get_archive
except ImportError:
    from io_utils import save_figure, save_dataset, get_archive

def gui(ctx: Context=None, λ: u.Quantity=None, φ: u.Quantity=None, σ: u.Quantity=None):

    if ctx is not None:
        ref_ctx = ctx
    else:
        ref_ctx = Context.get_VLTI()

    if λ is not None:
        ref_ctx.interferometer.λ = λ
    λ = ref_ctx.interferometer.λ
    if φ is not None:
        ref_ctx.interferometer.chip.φ = φ
    if σ is not None:
        ref_ctx.interferometer.chip.σ = σ

    step = 1e-20
    IA_sliders = [widgets.FloatSlider(value=0.5, min=0, max=0.5, step=step, description=f'I{i + 1}', continuous_update=False) for i in range(4)]
    IP_sliders = [widgets.FloatSlider(value=0, min=0, max=λ.value, step=step, description=f'I{i + 1}', continuous_update=False) for i in range(4)]
    P_sliders = [widgets.FloatSlider(value=0, min=0, max=λ.value, step=step, description=f'P{i + 1}', continuous_update=False) for i in range(14)]
    for i in range(14):
        P_sliders[i].value = ref_ctx.interferometer.chip.φ[i].to(λ.unit).value

    def beam_repr(beam: complex) -> str:
        return f'<b>{np.abs(beam):.2e}</b> * exp(<b>{np.angle(beam) / np.pi:.2f}</b> pi i)'

    # HTML widgets for displaying values
    inputs = [widgets.HTML(value=f' ') for _ in range(4)]
    bright_output = widgets.HTML(value=f' ')
    dark_outputs = [widgets.HTML(value=f' ') for _ in range(6)]
    kernel_outputs = [widgets.HTML(value=f' ') for _ in range(3)]

    def update_gui(*args):
        ctx = copy(ref_ctx)
        ψ = np.array([IA_sliders[i].value * np.exp(1j * IP_sliders[i].value / λ.value * 2 * np.pi) for i in range(4)])
        for i in range(14):
            ctx.interferometer.chip.φ[i] = P_sliders[i].value * λ.unit

        # Compute raw outputs (intensities) and processed kernels
        out_fields = ctx.interferometer.chip.get_output_fields(ψ=ψ, λ=λ)
        raw_outs = np.abs(out_fields) ** 2
        k = ctx.interferometer.chip.process_outputs(raw_outs)
        b = raw_outs[0]
        d = raw_outs[1:]

        # Update textual widgets
        for (i, beam) in enumerate(ψ):
            inputs[i].value = f'<b>Input {i + 1} -</b> Amplitude: <code>{beam_repr(beam)}</code> Intensity: <code><b>{np.abs(beam) ** 2 * 100:.1f}%</b></code>'
        bright_output.value = f'<b>Bright -</b> Intensity: <code><b>{b * 100:.3e}%</b></code>'
        for (i, beam) in enumerate(d):
            dark_outputs[i].value = f'<b>Dark {i + 1} -</b> Intensity: <code><b>{beam * 100:.3e}%</b></code>'
        for (i, beam) in enumerate(k):
            # Kernel values are differences of intensities; normalize by bright intensity
            kernel_outputs[i].value = f'<b>Kernel {i + 1} -</b> Value: <code>{beam:.2e}</code>  KN depth: <code>{beam / b:.2e}</code>'

        phases.value = ctx.interferometer.chip.plot_output_phase(λ=λ, plot=False, ψ=ψ)

        # Update small images for inputs
        for i in range(len(ψ)):
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.imshow([[np.abs(ψ[i]) ** 2]], cmap='hot', vmin=0, vmax=np.sum(np.abs(ψ) ** 2))
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            photometric_cameras[i].value = buf.getvalue()

        # Update images for bright and dark outputs
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.imshow([[b]], cmap='hot', vmin=0, vmax=np.sum(d) + b)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        raw_cameras[0].value = buf.getvalue()

        # dark images
        for i in range(len(d)):
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.imshow([[d[i]]], cmap='hot', vmin=0, vmax=np.sum(d))
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            raw_cameras[i + 1].value = buf.getvalue()

        # kernel images
        for i in range(len(k)):
            fig, ax = plt.subplots(figsize=(1, 1))
            vmax = max(1e-9, float(np.max(np.abs(k))))
            ax.imshow([[k[i]]], cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            kernel_cameras[i].value = buf.getvalue()

        return (b, d)
    photometric_cameras = [widgets.Image(width=50, height=50) for _ in range(4)]
    # raw_cameras: bright + 6 darks
    raw_cameras = [widgets.Image(width=50, height=50) for _ in range(7)]
    dark_cameras = [widgets.Image(width=50, height=50) for _ in range(6)]
    kernel_cameras = [widgets.Image(width=50, height=50) for _ in range(3)]
    phases = widgets.Image()
    vbox = widgets.VBox([
        widgets.HTML('<h1>Inputs</h1>'),
        widgets.HTML('Amplitude:'),
        widgets.HBox(IA_sliders[:4]),
        widgets.HTML('Phase:'),
        widgets.HBox(IP_sliders[:4]),
        *[widgets.HBox([photometric_cameras[i], x]) for (i, x) in enumerate(inputs)],
        widgets.HTML('<h1>Phases</h1>'),
        phases,
        widgets.HTML('<h1>Raw outputs (bright + darks)</h1>'),
        widgets.HBox(P_sliders[:4]),
        widgets.HBox(P_sliders[4:8]),
        widgets.HBox([raw_cameras[0], bright_output]),
        *[widgets.HBox([raw_cameras[i+1], x]) for (i, x) in enumerate(dark_outputs)],
        widgets.HTML('<h1>Recombiner</h1>'),
        widgets.HBox(P_sliders[8:11]),
        widgets.HBox(P_sliders[11:14]),
        widgets.HTML('<h1>Kernels</h1>'),
        *[widgets.HBox([kernel_cameras[i], x]) for (i, x) in enumerate(kernel_outputs)]
    ])
    for widget in P_sliders:
        widget.observe(update_gui, 'value')
    for widget in IA_sliders:
        widget.observe(update_gui, 'value')
    for widget in IP_sliders:
        widget.observe(update_gui, 'value')
    update_gui()
    return vbox

def run(save_as: str = "archives"):
    """Standalone runner simulating manual control state and saving outputs."""
    print("Running manual_control analysis...")
    ctx = Context.get_VLTI()
    outs = ctx.observe()
    k = ctx.interferometer.chip.process_outputs(outs)
    b = outs[0]
    d = outs[1:]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Bright"] + [f"D{i+1}" for i in range(len(d))] + [f"K{i+1}" for i in range(len(k))],
           np.concatenate([[b], d, k]))
    ax.set_title("Manual Control - Baseline Outputs")
    ax.set_ylabel("Intensity")
    if save_as:
        save_figure(fig, "manual_control_state", save_as, analysis_name="manual_control")
        save_dataset({
            "bright": b,
            "darks": d,
            "kernels": k,
        }, "outputs", save_as=save_as, analysis_name="manual_control")
    print("Done manual_control.")

if __name__ == "__main__":
    run()