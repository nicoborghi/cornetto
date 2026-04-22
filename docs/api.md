# API reference

## `corner(data, **kwargs)`

One-shot convenience wrapper equivalent to `Cornetto(data, ...).plot(...)`.
Returns `(fig, axes)`.

**Analysis kwargs** (`Cornetto.__init__`):

| Name            | Type               | Default    | Notes                                                            |
| ---             | ---                | ---        | ---                                                              |
| `data`          | `dict[str, array]` | (required) | 1-D `(N,)` or 2-D `(N_chains, N)` arrays, per parameter.        |
| `params`        | `list[str]`        | `None`     | Subset to plot; unused arrays are never processed.               |
| `labels`        | `dict[str, str]`   | `{}`       | Display labels (LaTeX supported via raw strings).                |
| `truths`        | `dict[str, scalar \| array]` | `None` | Scalar shared; array length `N_chains` for per-chain truths. |
| `chain_labels`  | `list[str]`        | `None`     | Legend labels, one per chain.                                    |
| `weights`       | array or list      | `None`     | Importance-sampling weights, one array per chain (or shared).    |
| `subsample`     | `int`              | `None`     | Thin each chain to this many samples.                            |
| `ax_lims`       | `dict[str, (lo, hi)]` | `None` | Explicit axis ranges.                                            |
| `max_chains`    | `int`              | `10`       | Hard cap on simultaneous chains.                                 |
| `stat`          | `str \| callable`  | `"median"` | See [Statistics](guide.md#statistics).                           |
| `sigmas`        | `tuple[float]`     | `(1, 2)`   | n-sigma to 2-D level conversion.                                 |
| `kwargs_truths` | `dict`             | `None`     | Truth line style (`color`, `lw`, `ls`, `alpha`, `marker`, `markersize`). |
| `kwargs_stats`  | `dict[str, dict]`  | `None`     | Per-stat overrides (`lw`, `ls`, `alpha`).                        |

**Plot kwargs** (`Cornetto.plot`):

| Name                | Default      | Notes                                                |
| ---                 | ---          | ---                                                  |
| `color`           | `"cornetto"` | Named palette or list of hex codes.                  |
| `dark`              | `False`      | Dark theme.                                          |
| `smooth`            | `True`       | KDE (`True`) or histogram (`False`) on the diagonal. |
| `bandwidth`         | `None`       | Explicit KDE bandwidth; overrides Silverman.         |
| `bandwidth_scale`   | `0.8`        | Multiplier on Silverman's rule.                      |
| `n_grid`            | `128`        | KDE evaluation grid per axis.                        |
| `fill_1d`           | `True`       | Light tint under the 1-D curve.                      |
| `fill_contours`     | `True`       | Filled 2-D contour bands.                            |
| `contour_lw`        | `1.2`        | 2-D contour line width.                              |
| `show_tension`      | `False`      | Overlap-integral annotation in 2-D panels.           |
| `annotate_peaks`    | `False`      | Mark local maxima in 1-D marginals.                  |
| `show_titles`       | `True`       | `value+err/-err` heading (single-chain only).        |
| `figsize`           | `None`       | Override figure size.                                |
| `fig_size_per_dim`  | `2.3`        | Inches per parameter when `figsize` is unset.        |
| `label_pad`         | `0.08`       | `hspace` / `wspace` between panels.                  |
| `title`             | `None`       | Figure suptitle.                                     |
| `fig`               | `None`       | Draw into an existing figure.                        |
| `usetex`            | `False`      | Use LaTeX for text rendering (requires a TeX install). |
| `use_datashader`    | `False`      | Rasterized scatter of raw samples in 2-D panels. Requires `datashader pandas`. |

---

## `quick_corner(data, **kwargs)`

Fast path - histograms only, same dict API.

| Name                | Default        | Notes                                       |
| ---                 | ---            | ---                                         |
| `data`              | (required)     | As above.                                   |
| `params`            | `None`         | Subset.                                     |
| `labels`            | `{}`           |                                             |
| `chain_labels`      | `None`         |                                             |
| `truths`            | `None`         |                                             |
| `weights`           | `None`         |                                             |
| `ax_lims`           | `None`         |                                             |
| `subsample`         | `20_000`       | Default thin for responsiveness.            |
| `bins`              | `30`           | Histogram bins per axis.                    |
| `sigmas`            | `(1, 2)`       |                                             |
| `stat`              | `"median"`     | Built-in name or callable; same as `corner`. |
| `kwargs_stats`      | `None`         | Per-stat `{lw, ls, alpha}` overrides.       |
| `color`           | `"cornetto"`   |                                             |
| `dark`              | `False`        |                                             |
| `figsize`           | `None`         |                                             |
| `fig_size_per_dim`  | `2.0`          | Matches `corner()` for consistent output.   |
| `title`             | `None`         |                                             |
| `fig`               | `None`         |                                             |
| `usetex`            | `False`        | Use LaTeX for text rendering.               |

Returns `(fig, axes)`.

---

## `marginal(data, **kwargs)`

1-D marginal distributions for all parameters in a grid layout.

| Name                | Default        | Notes                                                   |
| ---                 | ---            | ---                                                     |
| `data`              | (required)     | Same dict API as `corner`.                              |
| `params`            | `None`         | Subset.                                                 |
| `labels`            | `{}`           |                                                         |
| `truths`            | `None`         |                                                         |
| `chain_labels`      | `None`         |                                                         |
| `weights`           | `None`         |                                                         |
| `stat`              | `"median"`     |                                                         |
| `ncols`             | `5`            | Panels per row; last row padded with hidden axes.       |
| `color`           | `"cornetto"`   |                                                         |
| `dark`              | `False`        |                                                         |
| `show_titles`       | `True`         | `value+err/-err` heading (single-chain only).           |
| `show_legend`       | `None`         | Auto-show when `n_chains >= 2`.                         |
| `fill_1d`           | `False`        |                                                         |
| `annotate_peaks`    | `False`        |                                                         |
| `figsize`           | `None`         |                                                         |
| `fig_size_per_dim`  | `2.5`          |                                                         |
| `title`             | `None`         |                                                         |
| `usetex`            | `False`        |                                                         |

Returns `(fig, axes)` where `axes` has shape `(nrows, ncols)`.

---

## `trace(data, **kwargs)`

Trace plot - one row per parameter, x = sample index, y = value.
Uses datashader for rasterized rendering (falls back to matplotlib if unavailable).

| Name                     | Default      | Notes                                                      |
| ---                      | ---          | ---                                                        |
| `data`                   | (required)   | Same dict API as `corner`.                                 |
| `params`                 | `None`       | Subset.                                                    |
| `labels`                 | `{}`         |                                                            |
| `chain_labels`           | `None`       |                                                            |
| `stride`                 | `50`         | Running-median window = `max(1, N_samples // stride)`.    |
| `use_datashader`         | `True`       | Rasterize via datashader; needs `pip install datashader pandas`. |
| `datashader_kwargs`      | `None`       | Dict with `n_px_w`, `n_px_h`, `how`, `min_alpha`, `max_alpha`. |
| `color`                | `"cornetto"` |                                                            |
| `dark`                   | `False`      |                                                            |
| `show_legend`            | `None`       | Auto-show when `n_chains >= 2`.                            |
| `figsize`                | `None`       |                                                            |
| `fig_height_per_param`   | `1.5`        | Row height in inches.                                      |
| `title`                  | `None`       |                                                            |
| `usetex`                 | `False`      |                                                            |

Returns `(fig, axes)` where `axes` has shape `(N_params,)`.

---

## `trace_marginal(data, **kwargs)`

Combined 1-D KDE (left) + trace (right) - one row per parameter.
The 1-D panel shows KDE curves without CI bands or titles.

| Name                     | Default      | Notes                                              |
| ---                      | ---          | ---                                                |
| `data`                   | (required)   | Same dict API as `corner`.                         |
| `params`                 | `None`       | Subset.                                            |
| `labels`                 | `{}`         |                                                    |
| `truths`                 | `None`       |                                                    |
| `chain_labels`           | `None`       |                                                    |
| `stat`                   | `"median"`   |                                                    |
| `stride`                 | `50`         | Running-median window.                             |
| `use_datashader`         | `True`       | Rasterize traces via datashader.                   |
| `datashader_kwargs`      | `None`       | As for `trace`.                                    |
| `width_ratios`           | `(1.0, 4.0)` | Relative widths of the marginal and trace columns. |
| `color`                | `"cornetto"` |                                                    |
| `dark`                   | `False`      |                                                    |
| `show_legend`            | `None`       | Auto-show when `n_chains >= 2`.                    |
| `figsize`                | `None`       |                                                    |
| `fig_height_per_param`   | `1.5`        | Row height in inches.                              |
| `title`                  | `None`       |                                                    |
| `usetex`                 | `False`      |                                                    |

Returns `(fig, axes)` where `axes` has shape `(N_params, 2)`.
`axes[:, 0]` - 1-D marginal panels; `axes[:, 1]` - trace panels.

---

## `class Cornetto` {#cornetto-class}

Stateful interface - useful when you want to render multiple figures from the
same dataset without recomputing KDEs, or to inspect/export statistics.

### `Cornetto(data, ...)`

Same arguments as `corner`'s analysis kwargs.

### `Cornetto.plot(...)`

Same as `corner`'s plot kwargs. KDEs and statistics are computed lazily and
cached; subsequent calls with the same `n_grid` / `bandwidth` parameters reuse
the cache.

### `Cornetto.marginal(...)` {#cornettomarginal}

Same kwargs as `marginal()` (visual arguments only). Returns `(fig, axes)` with
shape `(nrows, ncols)`.

### `Cornetto.trace(...)` {#cornettotrace}

Same kwargs as `trace()`. Returns `(fig, axes)` with shape `(N_params,)`.

### `Cornetto.trace_marginal(...)` {#cornettotracemarginal}

Same kwargs as `trace_marginal()`. Returns `(fig, axes)` with shape `(N_params, 2)`.

### `Cornetto.pairplot(other, ...)` {#cornettopairplot}

Draw `self` in the lower triangle + diagonal and `other` in the upper triangle.
`other` must be a `Cornetto` whose params are a subset of `self`'s.

### `Cornetto.info()` {#cornettoinfo}

Print the setup (stat, contour sigma mapping, truth style, palette, theme, KDE
grid / bandwidth) used by the last `plot()` call. Intended as a readable
replacement for legend clutter.

### `Cornetto.summary()`

Return a `SummaryTable` with `median`, `p16`, `p84`, `mean`, `std` per
(param, chain). Renders as HTML in Jupyter, plain text in the terminal.

### `Cornetto.latex(caption="", label="tbl:posterior", fmt=".4g", style="aastex")`

Return a LaTeX table string via `astropy.io.ascii`. Requires `astropy`.

---

## Built-in statistics

| Name                 | Module       | `label`           |
| ---                  | ---          | ---               |
| `stat_median`        | `cornetto`   | `"median"`        |
| `stat_median_mad`    | `cornetto`   | `"median±MAD"`    |
| `stat_median_hdi`    | `cornetto`   | `"median+HDI68%"` |
| `stat_mean`          | `cornetto`   | `"mean±std"`      |

Registry: `cornetto.STAT_REGISTRY` - `{str: callable}`.

Each function has the signature `fn(samples, weights=None) -> dict` and must
return `{"center": float, "lo": float, "hi": float, "label": str}`.

---

## Utilities

### `sigmas_to_levels(sigmas)`

Convert n-sigma tuple to 2-D probability-mass levels via
`P(n) = 1 - exp(-n^2/2)`.

### `hdi(samples, prob=0.68)`

Highest Density Interval - shortest interval containing `prob` of the mass.

### `compute_stats(samples, weights=None)`

Returns `{mean, median, std, p16, p84}`.

### Module constants

- `cornetto.MAX_CHAINS`     - hard cap (10).
- `cornetto.CORNETTO_PALETTE` - multi-chain jewel-tone palette (10 colors).
- `cornetto.PALETTES`       - named single-chain palettes.
- `cornetto.DEFAULT_LEVELS` - `sigmas_to_levels((1, 2))`.
