"""
Marginal Distributions
======================

[`marginal()`](../../api.md#cornettomarginal) shows each parameter's 1-D KDE in
a tidy grid, with optional CI bands and truth markers.  No 2-D panels — ideal
for quick parameter summaries or chains with many parameters.
"""

# %%
# Shared synthetic posterior
# --------------------------

import numpy as np
from cornetto import marginal

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
# Default marginal grid
# ---------------------
# Four panels arranged in one row, with ``value⁺ᵃ₋ᵦ`` titles and a truth
# marker on ``mass_1``.

fig, axes = marginal(
    data,
    labels=labels,
    truths={"mass_1": 30.0},
    chain_labels=["GW200129"],
    ncols=4,
)

# %%
# Multi-chain marginals
# ---------------------
# Compare two events side-by-side. The legend appears automatically when
# ``n_chains >= 2``.

data_2 = {
    k: np.stack([data[k], rng.normal(data[k].mean() + 5, 3, N)])
    for k in ["mass_1", "mass_2"]
}

fig, axes = marginal(
    data_2,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Event A", "Event B"],
    ncols=2,
)

# %%
# Filled curves, coral color
# ----------------------------
# ``fill_1d=True`` adds a light tint under each KDE curve.

fig, axes = marginal(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    ncols=4,
    fill_1d=True,
    color="coral",
)

# %%
# HDI statistic
# -------------
# Switch the CI interval to the 68 % Highest Density Interval instead of the
# default 16–84 percentile band.

fig, axes = marginal(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    ncols=4,
    stat="median_hdi",
)
