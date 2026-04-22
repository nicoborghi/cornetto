# Quickstart

## Install

```bash
pip install cornetto
```

## Your first corner plot

Cornetto takes a **`dict[str, array]`**: parameter name to 1-D sample array.

```python
# %%
import numpy as np
from cornetto import corner

rng = np.random.default_rng(0)
data = {
    "m1":  rng.normal(30, 4, 10_000),
    "m2":  rng.normal(25, 3, 10_000),
    "chi": rng.uniform(-1, 1, 10_000),
}

fig, axes = corner(data)
```

That's it. Labels default to the dict keys; contour levels default to 1σ and 2σ
(39.3% and 86.5% in the 2-D sense, see [Statistics](guide.md#statistics)).

## LaTeX labels

```python
# %%
corner(data, labels={
    "m1":  r"$m_1\,[M_\odot]$",
    "m2":  r"$m_2\,[M_\odot]$",
    "chi": r"$\chi_{\mathrm{eff}}$",
})
```

## Truth markers

```python
# %%
corner(data, truths={"m1": 30.0, "chi": 0.0})
```

A scalar applies to every panel for that parameter. Pass an array of length
`N_chains` to set a different truth for each chain.

!!! tip "Don't need dotted lines everywhere?"
    Set `kwargs_truths={"ls": "-", "marker": "o"}` for solid crosshairs plus a
    dot at the truth location, or pass `kwargs_truths={"alpha": 0.0}` to hide
    the lines entirely.

## Subset of parameters

```python
# %%
corner(data, params=["m1", "m2"])   # only plot these two
```

Cornetto only processes the arrays it actually draws, so this is safe to use
on wide chains.

## Saving the figure

`corner()` returns a standard `(fig, axes)` pair. Save it like any matplotlib
figure. PDF or SVG keeps vector quality for papers and talks.

```python
# %%
fig, axes = corner(data)
fig.savefig("posterior.pdf", bbox_inches="tight")   # vector, scales cleanly
# or: fig.savefig("posterior.svg", bbox_inches="tight")
```

## What the legend shows

Only the chain name(s). Everything else (statistic, contour levels, truth
styling) is available through [`Cornetto.info()`](api.md#cornettoinfo), so the
plot stays clean.

```python
# %%
from cornetto import Cornetto
c = Cornetto(data, chain_labels=["mock"])
fig, axes = c.plot()
c.info()
```

```text
Cornetto setup
──────────────
  params         : ['m1', 'm2', 'chi']
  n_chains       : 1
  stat           : median  (center line + shaded interval on 1-D marginals)
  contour sigmas : 1σ→39.3%, 2σ→86.5%
  …
```

## Next steps

- [User guide](guide.md) — multi-event, weights, statistics, styling.
- [Fast mode](quick-corner.md) — sub-second plots for iteration.
