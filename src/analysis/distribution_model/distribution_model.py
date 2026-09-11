"""Ajustement de lois statistiques aux sorties du nuller.

Contient des routines pour générer des échantillons, ajuster des
distributions (Cauchy, Laplace, Johnson SU, ...) et tracer les
résultats.
"""
from copy import deepcopy as copy
from astropy import units as u
import numpy as np
import fitter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from phise import Context

try:
    from src.analysis.io_utils import save_figure, save_dataset, get_archive
except ImportError:
    from io_utils import save_figure, save_dataset, get_archive

def fit(ctx: Context=None, save_as: str = None, n_samples: int = 10000):
    """"fit.

Parameters
----------
ctx : Context, optional
    PHISE Context to use.
save_as : str, optional
    Directory or file path to save figures and datasets.
n_samples : int, optional
    Number of Monte Carlo samples to draw (default 10000).

Returns
-------
dict
    Fitted parameters for Cauchy, Laplace, Johnson SU and the dataset.
"""
    if ctx is None:
        ctx = Context.get_VLTI()
        ctx.interferometer.chip.σ = np.zeros(14) * u.nm
    else:
        ctx = copy(ctx)
    ctx.Δh = ctx.interferometer.camera.e.to(u.hour).value * u.hourangle
    ctx.target.companions = []
    N = n_samples
    data = np.empty((N, 3))
    print(f'⌛ Generating data ({N} samples)...')
    for i in range(N):
        if (i + 1) % max(1, N // 20) == 0 or i == N - 1:
            print(f'{(i + 1) / N * 100:.2f}%', end='\r')
        outs = ctx.observe()
        k = ctx.interferometer.chip.process_outputs(outs)
        b = outs[0]
        data[i] = k / b
    print('✅ Data generation complete.')
    data = data[:, 0]

    def cauchy(x, μ, σ):
        return 1 / (np.pi * σ * (1 + ((x - μ) / σ) ** 2))

    def laplace(x, μ, σ):
        return 1 / (2 * σ) * np.exp(-np.abs(x - μ) / σ)

    def johnsonsu(x, μ, σ, γ, δ):
        return 1 / (σ * np.sqrt(2 * np.pi)) * 1 / np.sqrt(1 + ((x - μ) / σ) ** 2) * np.exp(-0.5 * (γ + δ * np.sinh((x - μ) / σ)) ** 2)

    (hist, bin_edges) = np.histogram(data, bins=500, density=True)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    print('⌛ Fitting distributions...')
    (cauchy_pop, _) = curve_fit(cauchy, x, hist, p0=[np.mean(data), np.std(data)])
    print('✅ Cauchy fit complete.')
    (laplace_pop, _) = curve_fit(laplace, x, hist, p0=[np.mean(data), np.std(data)])
    print('✅ Laplace fit complete.')
    (johnsonsu_pop, _) = curve_fit(johnsonsu, x, hist, p0=[np.mean(data), np.std(data), 0, 1])
    print('✅ Johnson SU fit complete.')
    
    fig = plt.figure(figsize=(5, 5))
    plt.hist(data, bins=50, density=True, label='Data', log=True)
    plt.plot(x, cauchy(x, *cauchy_pop), 'r-', label='Cauchy Fit')
    plt.plot(x, laplace(x, *laplace_pop), 'g-', label='Laplace Fit')
    plt.plot(x, johnsonsu(x, *johnsonsu_pop), 'b-', label='Johnson SU Fit')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.ylim(bottom=0.1, top=100.0)
    plt.grid()
    if save_as:
        save_figure(fig, "distribution_fit", save_as, analysis_name="distribution_model")
        save_dataset({
            "cauchy_params": cauchy_pop,
            "laplace_params": laplace_pop,
            "johnsonsu_params": johnsonsu_pop,
            "data_sample": data[:1000]
        }, "fit_results", save_as=save_as, analysis_name="distribution_model")
    plt.show()
    return {
        "cauchy": cauchy_pop,
        "laplace": laplace_pop,
        "johnsonsu": johnsonsu_pop,
        "data": data,
    }

if __name__ == "__main__":
    fit(save_as="archives", n_samples=5000)