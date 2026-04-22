# Fast mode `quick_corner`

When you're tweaking priors, comparing chains, or sanity-checking a new sampler run, slow renders get in the way. `quick_corner` uses the same API as `corner` but skips KDE - **histograms only**.

## API parity

```python
# %%
from cornetto import quick_corner

fig, axes = quick_corner(
    data,
    chain_labels=["my chain"],
    truths={"m1": 30.0},
    labels={"m1": r"$m_1$"},
)
```

All the same semantics as `corner`:

- dict-of-arrays input, 1-D or 2-D
- `params`, `labels`, `truths`, `chain_labels`, `weights`, `ax_lims`
- `stat`, `kwargs_stats` - full custom statistic, same built-ins
  (`"median"`, `"median_mad"`, `"median_hdi"`, `"mean"`) or your own callable
- `color`, `dark`
- `sigmas` for contour levels
- returns `(fig, axes)` with the same outer dimensions as `corner()`

## What it does differently

| Feature               | `corner` | `quick_corner` |
| ---                   | ---      | ---            |
| 1-D marginals         | FFT-KDE  | Histogram (step line) |
| 2-D marginals         | FFT-KDE + contours | `histogram2d` + contours |
| Default `subsample`   | none     | 20 000 |

Beyond that, panels are drawn lighter: no fills, titles, peak markers, or additional annotations.



## Why it's fast

1. **No KDE.** FFT-KDE on an 8-parameter corner means 28 two-dimensional KDE
   calls - the dominant cost in `corner()`. `quick_corner` replaces them with
   histograms.
2. **Minimal drawing.** Coarse histogram grid, one contour line per sigma,
   no peak detection, no tension annotation, no titles.

See the [Performance](performance.md) page for numbers.

