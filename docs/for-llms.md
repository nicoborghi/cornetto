# cornetto — compact reference for LLMs

This page is a dense, self-contained reference intended for AI assistants
helping users with cornetto.

---

## Package identity

- **PyPI**: `pip install cornetto`
- **Version**: 0.2.0
- **License**: MIT
- **GitHub**: https://github.com/nicoborghi/cornetto
- **Backend**: [KDExpress](https://github.com/nicoborghi/KDExpress) FFT-KDE

---

## Data format

```python
data = {
    "param_name": array,   # shape (N,) single chain  OR  (N_chains, N) multi-chain
    ...
}
```

- All arrays in the same dict must share the same `N` (samples per chain).
- 1-D arrays are broadcast to all chains when mixed with 2-D arrays.
- Hard cap: `MAX_CHAINS = 10`.

---

## Main functions

```python
from cornetto import corner, quick_corner, marginal, trace, trace_marginal, Cornetto

# Full corner plot (FFT-KDE, smooth contours)
fig, axes = corner(data, **kwargs)

# Fast histogram-only corner (~7× faster on 8-param chains)
fig, axes = quick_corner(data, **kwargs)

# 1-D marginal grid
fig, axes = marginal(data, **kwargs)

# Trace plot (sample index vs. value)
fig, axes = trace(data, **kwargs)

# 1-D KDE + trace side-by-side
fig, axes = trace_marginal(data, **kwargs)

# Stateful class — caches KDEs between plot() calls
c = Cornetto(data, **analysis_kwargs)
fig, axes = c.plot(**plot_kwargs)
c.info()       # print setup
c.summary()    # SummaryTable (HTML in Jupyter, text in terminal)
c.latex()      # AASTeX LaTeX table (requires astropy)
a.pairplot(b)  # lower triangle = a, upper triangle = b
```

---

## Key parameters (corner / Cornetto)

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `data` | `dict[str, array]` | required | See data format above |
| `params` | `list[str]` | `None` | Subset to draw; others are ignored entirely |
| `labels` | `dict[str, str]` | `{}` | Display labels; LaTeX via raw strings works |
| `truths` | `dict[str, scalar\|array]` | `None` | Scalar = shared; array of length `N_chains` = per-chain |
| `chain_labels` | `list[str]` | `None` | Legend labels, one per chain |
| `weights` | array or list | `None` | Importance-sampling weights; normalised internally |
| `subsample` | `int` | `None` | Thin chain before KDE |
| `stat` | `str\|callable` | `"median"` | `"median"`, `"median_mad"`, `"median_hdi"`, `"mean"` |
| `sigmas` | `tuple[float]` | `(1, 2)` | n-sigma contour levels |
| `color` | `str\|list` | `"cornetto"` | Named: `"indigo"`, `"coral"`, `"teal"`, `"gold"`, `"ink"` |
| `dark` | `bool` | `False` | Dark theme |
| `smooth` | `bool` | `True` | KDE (`True`) or histogram (`False`) on diagonal |
| `fill_contours` | `bool` | `True` | Filled 2-D contour bands |
| `show_titles` | `bool` | `True` | `value⁺ᵃ₋ᵦ` heading (single-chain only) |
| `ax_lims` | `dict[str, (lo,hi)]` | `None` | Explicit axis ranges (parameter: `ax_lims`, NOT `range`) |
| `kwargs_truths` | `dict` | `None` | Truth line style: `color`, `lw`, `ls`, `alpha`, `marker` |

---

## Gotchas

- **`ax_lims`, not `range`** — parameter is `ax_lims` to avoid shadowing Python's built-in `range()`.
- **KDExpress `fft_kde2d` always needs weights** — cornetto passes uniform weights internally when the user doesn't supply any; do not pass `weights=None` directly to KDExpress.
- **Transpose** — `fft_kde2d` returns shape `(Mx, My)`; cornetto transposes to `(My, Mx)` for matplotlib's convention.
- **JAX CPU warning** — suppressed automatically at import; no user action needed.
- **LaTeX labels** — use raw strings (`r"$m_1$"`); `usetex=False` by default for portability.
- **Small N** — fewer than ~50 samples per parameter disables KDE for that panel and falls back to a histogram with a `RuntimeWarning`.
- **`N_chains` limit** — default `max_chains=10`; override per call.

---

## Custom stat function signature

```python
def my_stat(samples: np.ndarray, weights=None) -> dict:
    return {"center": float, "lo": float, "hi": float, "label": str}

corner(data, stat=my_stat)
```

---

## Minimal working example

```python
import numpy as np
from cornetto import corner

rng = np.random.default_rng(0)
data = {"m1": rng.normal(30, 4, 10_000), "m2": rng.normal(25, 3, 10_000)}
fig, axes = corner(data, labels={"m1": r"$m_1$", "m2": r"$m_2$"},
                   truths={"m1": 30.0}, chain_labels=["posterior"])
fig.savefig("corner.pdf", bbox_inches="tight")
```
