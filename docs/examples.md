# Examples

All examples use the same synthetic posterior as a starting point:

```python
# %%
import numpy as np
from cornetto import corner, marginal, trace, trace_marginal, Cornetto

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

data_2 = {k: np.stack([data[k], rng.normal(data[k].mean() + 5, 3, N)])
          for k in ["mass_1", "mass_2"]}
```

---

## Corner plot

The default `corner()` call - FFT-KDE contours, CI bands, truth markers.

```python
# %%
fig, axes = corner(
    data,
    labels=labels,
    truths={"mass_1": 30.0, "spin": 0.0},
    chain_labels=["GW200129"],
)
fig.savefig("corner.pdf", bbox_inches="tight")
```

### With datashader scatter overlay

Add `use_datashader=True` to show raw sample density under the KDE contours.
Requires `pip install datashader pandas`.

```python
# %%
fig, axes = corner(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    use_datashader=True,
    fill_contours=False,   # contour lines only - scatter carries the density
)
fig.savefig("corner_scatter.pdf", bbox_inches="tight")
```

### Two-chain comparison

```python
# %%
fig, axes = corner(
    data_2,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Event A", "Event B"],
    color="cornetto",
)
fig.savefig("corner_multi.pdf", bbox_inches="tight")
```

---

## Marginal distributions

`marginal()` shows the 1-D marginals in a tidy grid - no 2-D panels.
Good for many parameters or for a quick parameter summary.

```python
# %%
fig, axes = marginal(
    data,
    labels=labels,
    truths={"mass_1": 30.0},
    chain_labels=["GW200129"],
    ncols=4,
)
fig.savefig("marginal.pdf", bbox_inches="tight")
```

### Multi-chain marginals

```python
# %%
fig, axes = marginal(
    data_2,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Event A", "Event B"],
    ncols=2,
)
fig.savefig("marginal_multi.pdf", bbox_inches="tight")
```

---

## Trace plot

`trace()` shows each parameter as a time series - useful for diagnosing
sampler convergence. Datashader renders the trace raster; the solid line
is the running median.

```python
# %%
fig, axes = trace(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    stride=50,          # running median window = N_samples // stride
)
fig.savefig("trace.pdf", bbox_inches="tight")
```

### Tuning the datashader rendering

```python
# %%
fig, axes = trace(
    data,
    labels=labels,
    stride=100,
    datashader_kwargs={
        "n_px_w": 1200,    # wider raster for a large figure
        "how":    "eq_hist",
        "min_alpha": 30,
    },
)
```

### Multi-chain trace

```python
# %%
fig, axes = trace(
    data_2,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Event A", "Event B"],
)
fig.savefig("trace_multi.pdf", bbox_inches="tight")
```

---

## Combined 1-D + trace (`trace_marginal`)

`trace_marginal()` puts a compact 1-D KDE on the left and the trace on the
right - both the shape and the convergence at a glance.

```python
# %%
fig, axes = trace_marginal(
    data,
    labels=labels,
    chain_labels=["GW200129"],
)
fig.savefig("trace_marginal.pdf", bbox_inches="tight")
```

### Adjusting the column width ratio

```python
# %%
fig, axes = trace_marginal(
    data,
    labels=labels,
    width_ratios=(1, 6),    # narrower marginal, wider trace
    fig_height_per_param=1.8,
)
```

### Multi-chain

```python
# %%
fig, axes = trace_marginal(
    data_2,
    labels={k: labels[k] for k in ["mass_1", "mass_2"]},
    chain_labels=["Event A", "Event B"],
)
fig.savefig("trace_marginal_multi.pdf", bbox_inches="tight")
```

---

## Using the `Cornetto` class

When you want more control - or to call multiple plot types on the same
dataset without recomputing KDEs - use the `Cornetto` class directly.

```python
# %%
c = Cornetto(
    data,
    labels=labels,
    chain_labels=["GW200129"],
    truths={"mass_1": 30.0},
    stat="median_hdi",
)

fig_corner,   ax_c = c.plot(color="teal", dark=True)
fig_marginal, ax_m = c.marginal(ncols=4)
fig_trace,    ax_t = c.trace()
fig_tm,       ax_tm = c.trace_marginal()

c.info()        # print the exact setup
tbl = c.summary()
print(tbl)
```
