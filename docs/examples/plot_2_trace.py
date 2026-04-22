"""
Trace Plots
===========

[`trace()`](../../api.md#cornettotrace) displays each parameter as a function
of sample index — the standard convergence diagnostic.
[`trace_marginal()`](../../api.md#cornettotracemarginal) pairs a compact 1-D
KDE on the left with the trace on the right so you see both shape and
convergence at a glance.

Both use **datashader** for rasterized rendering when it is available;
the solid line shows the running median.
"""

# %%
# Shared synthetic posterior
# --------------------------

import numpy as np
from cornetto import trace, trace_marginal

rng = np.random.default_rng(0)
N = 20_000

data = {
    "mass_1": rng.normal(30, 4, N),
    "mass_2": rng.normal(25, 3, N),
    "spin":   rng.uniform(-1, 1, N),
    "dist":   rng.exponential(400, N),
}

labels = {
    "mass_1": r"$m_1\,[M_\odot]$",
    "mass_2": r"$m_2\,[M_\odot]$",
    "spin":   r"$\chi_{\mathrm{eff}}$",
    "dist":   r"$d_L\,[\mathrm{Mpc}]$",
}

# %%
# Default trace plot
# ------------------
# One row per parameter. The solid line is the running median over a window
# of ``N_samples // stride`` samples.

fig, axes = trace(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    stride=50,
)

# %%
# Multi-chain trace
# -----------------
# Each chain gets its own colour from the color. The legend appears
# automatically.

data_2 = {
    k: np.stack([data[k], rng.normal(data[k].mean() + 5, 3, N)])
    for k in ["mass_1", "mass_2"]
}

fig, axes = trace(
    data_2,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Event A", "Event B"],
)

# %%
# Trace + marginal (default)
# --------------------------
# `trace_marginal()` combines a 1-D KDE panel (left) and the trace (right)
# in a single figure.

fig, axes = trace_marginal(
    data,
    labels=labels,
    chain_labels=["GW200129"],
)

# %%
# Adjusting column width ratio
# ----------------------------
# ``width_ratios=(1, 6)`` makes the marginal narrower relative to the trace.

fig, axes = trace_marginal(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    width_ratios=(1, 6),
    fig_height_per_param=1.8,
)

# %%
# Multi-chain trace_marginal
# --------------------------

fig, axes = trace_marginal(
    data_2,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Event A", "Event B"],
)
