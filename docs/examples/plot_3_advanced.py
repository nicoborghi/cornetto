"""
Advanced Usage
==============

The [`Cornetto`](../../api.md#cornetto-class) class is the stateful interface:
it caches KDEs and statistics so you can render multiple figures from the same
dataset without recomputing.  This section also shows styling options — palettes,
dark theme, truth line customisation, and the
[`pairplot`](../../api.md#cornettopairplot) comparison layout.
"""

# %%
# Shared synthetic posterior
# --------------------------

import numpy as np
from cornetto import Cornetto, corner

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
# Using the Cornetto class
# ------------------------
# Instantiate once, then call ``.plot()``, ``.marginal()``, ``.trace()`` and
# ``.trace_marginal()`` as many times as needed.  KDEs are computed once and
# cached.

c = Cornetto(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    truths={"mass_1": 30.0},
    stat="median_hdi",
)

fig, axes = c.plot(color="indigo")

# %%
# Summary statistics
# ------------------
# ``c.summary()`` returns a ``SummaryTable`` — renders as an HTML table in
# Jupyter and as plain text in the terminal.

tbl = c.summary()
print(tbl)

# %%
# All named palettes
# ------------------
# Five built-in single-chain palettes:
# ``"indigo"`` (default), ``"coral"``, ``"teal"``, ``"gold"``, ``"ink"``.

for color in ["indigo", "coral", "teal", "gold", "ink"]:
    fig, axes = corner(
        {k: data[k] for k in ["mass_1", "mass_2"]},
        labels={k: labels[k] for k in ["mass_1", "mass_2"]},
        chain_labels=[color],
        color=color,
        show_titles=False,
    )

# %%
# Custom truth line styling
# -------------------------
# Pass ``kwargs_truths`` to change the look of truth crosshairs and markers.

fig, axes = corner(
    data,
    labels=labels,
    truths={"mass_1": 30.0, "spin": 0.0},
    chain_labels=["GW200129"],
    kwargs_truths={"ls": "-.", "lw": 1.5, "marker": "*", "markersize": 8},
)

# %%
# Pairplot — lower vs. upper triangle
# ------------------------------------
# ``Cornetto.pairplot(other)`` overlays two chains on a single corner grid:
# ``self`` on the lower triangle + diagonal, ``other`` on the upper triangle.

data_B = {
    "mass_1": rng.normal(35, 3, N),
    "mass_2": rng.normal(28, 2, N),
}

a = Cornetto(
    {k: data[k] for k in ["mass_1", "mass_2"]},
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Model A"],
)
b = Cornetto(
    data_B,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Model B"],
)

fig, axes = a.pairplot(b, other_color="coral")

# %%
# Annotation: tension and peak markers
# -------------------------------------
# ``show_tension=True`` prints the overlap-integral 2-D tension in each
# off-diagonal panel; ``annotate_peaks=True`` marks local PDF maxima on
# the diagonal.

fig, axes = corner(
    {k: data[k] for k in ["mass_1", "mass_2"]},
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["GW200129"],
    annotate_peaks=True,
)
