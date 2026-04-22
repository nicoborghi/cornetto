"""
cornetto.core
~~~~~~~~~~~~~
The Cornetto class and panel-level drawing helpers.
"""
from __future__ import annotations
from typing import Any
from io import StringIO

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import KDExpress as kdx

from .kde import kde1d, kde2d
from .stats import (
    density_to_levels, overlap_integral, find_pdf_peaks,
    compute_stats, build_summary_table, SummaryTable,
    sigmas_to_levels, DEFAULT_LEVELS,
    STAT_REGISTRY, resolve_stat,
)
from .styles import (
    MAX_CHAINS, DEFAULT_PALETTE,
    get_theme, make_contour_colors, ColorManager,
)

# ── Defaults ──────────────────────────────────────────────────────────────────

_TRUTH_DEFAULTS: dict = dict(
    color="#e63946", lw=1., ls="-", alpha=0.8, zorder=5,
    marker="D", markersize=5,
)

# Per-stat visual defaults: lw/ls apply to the central-value line;
# alpha applies to the interval fill under the curve.
_STAT_DEFAULTS: dict[str, dict] = {
    "median":     dict(lw=1.2, ls="--", alpha=0.28),
    "median_mad": dict(lw=1.2, ls="--", alpha=0.22),
    "median_hdi": dict(lw=1.2, ls="--", alpha=0.28),
    "mean":       dict(lw=1.0, ls="-.", alpha=0.18),
}
_STAT_DEFAULTS_CALLABLE = dict(lw=1.0, ls="--", alpha=0.22)  # fallback for user fns


def _merge(defaults: dict, override: dict | None) -> dict:
    out = dict(defaults)
    if override:
        out.update(override)
    return out


def _line_kw(kw: dict) -> dict:
    """Strip marker keys — axvline doesn't accept them."""
    return {k: v for k, v in kw.items() if k not in ("marker", "markersize")}


def _delta_label(label: str) -> str:
    """Prefix ``label`` with Δ — LaTeX-aware for math-mode strings."""
    if "$" in label:
        i = label.index("$")
        return label[:i] + r"$\Delta " + label[i + 1:]
    return f"Δ{label}"


def _apply_delta_shift(
    chains: list[dict],
    truths: dict | None,
    n_chains: int,
) -> None:
    """Subtract per-chain truths from every chain in place.

    Each chain index subtracts its own truth value (scalar ``truths[p]``
    broadcasts to all chains; length-``n_chains`` arrays assign per chain).
    Raises if ``truths`` is empty or has no usable entries.
    """
    if not truths:
        raise ValueError("delta_mode=True requires `truths` to be provided.")
    n_applied = 0
    for p, tv in truths.items():
        if tv is None:
            continue
        arr = np.atleast_1d(tv).astype(float)
        for ch_idx, ch in enumerate(chains):
            if p not in ch:
                continue
            t = float(arr[ch_idx]) if arr.size > 1 else float(arr[0])
            ch[p] = ch[p] - t
        n_applied += 1
    if n_applied == 0:
        raise ValueError(
            "delta_mode=True but `truths` has no usable parameter entries.")


def _apply_tick_rotation(axes, rotation: float) -> None:
    """Rotate visible x- and y-tick labels on every axes in ``axes``."""
    ha = "right" if rotation > 0 else ("left" if rotation < 0 else "center")
    va = "top"   if rotation > 0 else ("bottom" if rotation < 0 else "center")
    for ax in np.asarray(axes).ravel():
        if ax is None:
            continue
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(rotation)
            lbl.set_ha(ha)
            lbl.set_rotation_mode("anchor")
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(rotation)
            lbl.set_va(va)
            lbl.set_rotation_mode("anchor")


# ── Data helpers ──────────────────────────────────────────────────────────────

def _parse_data(
    data: dict[str, Any],
    max_chains: int,
    keep_params: list[str] | None = None,
) -> tuple[list[str], list[dict[str, np.ndarray]]]:
    """Parse ``data`` into a flat list of per-chain dicts.

    Each parameter value may be:
    - 1-D array ``(N,)``         — shared across all chains
    - 2-D array ``(K, N)``       — K chains; trailing chains beyond K are absent
    - list ``[arr, arr, None, …]``— explicit per-chain assignment; ``None`` marks
                                    an absent chain at that position

    ``n_chains`` is the maximum chain count found across all parameters.
    Absent chains simply have no key for that parameter in their dict.
    """
    if not isinstance(data, dict) or not data:
        raise TypeError("`data` must be a non-empty dict.")

    params = [p for p in data.keys() if keep_params is None or p in set(keep_params)]
    if not params:
        raise ValueError("No requested params found in data.")

    # Pass 1 — determine global n_chains
    n_chains = 1
    for p in params:
        raw = data[p]
        if isinstance(raw, list):
            n_chains = max(n_chains, len(raw))
        else:
            arr = np.asarray(raw, dtype=float)
            if arr.ndim == 2:
                n_chains = max(n_chains, arr.shape[0])
            elif arr.ndim != 1:
                raise ValueError(
                    f"'{p}': arrays must be 1-D or 2-D; got shape {arr.shape}.")

    if n_chains > max_chains:
        raise ValueError(f"{n_chains} chains > max_chains={max_chains}.")

    # Pass 2 — populate chain dicts
    chains: list[dict[str, np.ndarray]] = [{} for _ in range(n_chains)]
    for p in params:
        raw = data[p]
        if isinstance(raw, list):
            for i, entry in enumerate(raw):
                if entry is not None:
                    chains[i][p] = np.asarray(entry, dtype=float)
        else:
            arr = np.asarray(raw, dtype=float)
            if arr.ndim == 1:
                for ch in chains:
                    ch[p] = arr
            else:  # 2-D
                for i in range(arr.shape[0]):
                    chains[i][p] = arr[i]
                # chains beyond arr.shape[0] remain absent for this param
    return params, chains


def _compute_ranges(
    chains: list[dict],
    params: list[str],
    user_limits: dict | None,
    pad: float = 0.05,
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for p in params:
        vals = np.concatenate([ch[p] for ch in chains if p in ch])
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            ranges[p] = (0.0, 1.0)
            continue
        lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.5))
        span = hi - lo or abs(lo) * 0.1 or 1.0
        ranges[p] = (lo - pad * span, hi + pad * span)
    if user_limits:
        for p, r in user_limits.items():
            if p in ranges:
                ranges[p] = tuple(r)  # type: ignore[assignment]
    return ranges


def _norm_weights(
    weights: Any,
    n_chains: int,
) -> list[np.ndarray | None]:
    if weights is None:
        return [None] * n_chains
    if isinstance(weights, np.ndarray) and weights.ndim == 1:
        return [weights / weights.sum()] * n_chains
    out: list[np.ndarray | None] = []
    for w in list(weights)[:n_chains]:
        out.append(None if w is None else (lambda a: a / a.sum())(np.asarray(w, float)))
    out += [None] * max(0, n_chains - len(out))
    return out


def _truth_scalars(
    truths: dict | None,
    param: str,
    n_chains: int,
) -> list[float | None]:
    if truths is None or param not in truths or truths[param] is None:
        return [None] * n_chains
    tv = np.atleast_1d(truths[param])
    vals = [float(v) for v in tv[:n_chains]]
    vals += [None] * max(0, n_chains - len(vals))
    return vals


def _scaled_bw1d(data: np.ndarray, scale: float) -> float:
    clean = data[np.isfinite(data)]
    return float(kdx.silverman_bw1d(clean)) * scale


def _scaled_bw2d(
    data_x: np.ndarray,
    data_y: np.ndarray,
    scale: float,
) -> tuple[float, float]:
    mask = np.isfinite(data_x) & np.isfinite(data_y)
    d2 = np.column_stack([data_x[mask], data_y[mask]])
    bw = np.asarray(kdx.silverman_bw2d(d2), dtype=float)
    return (float(bw[0]) * scale, float(bw[1]) * scale)


# ── Panel renderers ───────────────────────────────────────────────────────────

def _stat_kw_for(stat, stat_kws: dict) -> dict:
    """Look up visual kwargs for a stat (string key or callable label fallback)."""
    key = stat if isinstance(stat, str) else getattr(stat, "__name__", None)
    return stat_kws.get(key, stat_kws.get("_callable", _STAT_DEFAULTS_CALLABLE))


# ── Optional datashader helpers ───────────────────────────────────────────────

def _require_datashader():
    """Lazy import of datashader + pandas. Raises ImportError with install hint."""
    try:
        import datashader as ds
        import datashader.transfer_functions as tf
        import pandas as pd
        return ds, tf, pd
    except ImportError as exc:
        raise ImportError(
            "datashader and pandas are required for use_datashader=True.\n"
            "Install with:  pip install datashader pandas"
        ) from exc


def _running_median(
    samples: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_centers, medians) at ~stride evenly spaced points."""
    N = len(samples)
    if N == 0:
        return np.array([]), np.array([])
    window = max(3, N // stride)
    step   = max(1, N // min(max(stride, 1), N))
    cx     = np.arange(step // 2, N, step, dtype=np.intp)
    half   = window // 2
    med    = np.array([
        np.median(samples[max(0, i - half): min(N, i + half + 1)])
        for i in cx
    ])
    return cx.astype(float), med


def _ds_raster(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_px_w: int = 600,
    n_px_h: int = 120,
    how: str = "log",
    min_alpha: int = 20,
    max_alpha: int = 200,
) -> None:
    """Render (x, y) as a datashader raster on ``ax``."""
    ds, tf, pd = _require_datashader()
    from matplotlib.colors import to_rgba as _to_rgba

    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    ri, gi, bi = (int(v * 255) for v in _to_rgba(color)[:3])
    # datashader cmap: list of hex RGB strings; alpha is controlled separately
    color_hex = f"#{ri:02x}{gi:02x}{bi:02x}"
    cmap = [color_hex, color_hex]  # single hue; density → alpha via min_alpha/alpha

    df = pd.DataFrame({"x": x[mask].astype(np.float64),
                        "y": y[mask].astype(np.float64)})
    xlo, xhi = float(x_range[0]), float(x_range[1])
    ylo, yhi = float(y_range[0]), float(y_range[1])
    if xhi <= xlo: xhi = xlo + 1.0
    if yhi <= ylo: yhi = ylo + 1.0

    cvs = ds.Canvas(plot_width=n_px_w, plot_height=n_px_h,
                    x_range=(xlo, xhi), y_range=(ylo, yhi))
    agg = cvs.points(df, "x", "y", ds.count())
    img = tf.shade(agg, cmap=cmap, how=how, alpha=max_alpha, min_alpha=min_alpha)
    rgba = np.array(img.to_pil())
    ax.imshow(rgba, extent=[xlo, xhi, ylo, yhi],
              aspect="auto", origin="upper", zorder=1, interpolation="nearest")


def _draw_diagonal(
    ax: plt.Axes,
    pdfs: list[np.ndarray],        # one pdf per chain (already on x_grid)
    x_grid: np.ndarray,
    chains_samples: list[np.ndarray],  # raw samples per chain (for stat fns)
    chains_weights: list,              # weights per chain
    colors: list[str],
    stat,                          # str or callable
    stat_kws: dict[str, dict],
    tv_list: list[float | None],
    kw_truths: dict,
    fill_1d: bool,
    annotate_peaks: bool,
    show_titles: bool,
    txt_color: str,
    n_chains: int,
    param: str = "",
    show_stat: bool = True,
    show_median: bool = True,
) -> None:
    stat_fn       = resolve_stat(stat)
    stat_kw       = _stat_kw_for(stat, stat_kws)
    first_results: list[dict] = []   # stat result per chain, for title

    for ch_idx, (pdf, samples, weights, color) in enumerate(
            zip(pdfs, chains_samples, chains_weights, colors)):
        if pdf is None or pdf.max() == 0:
            first_results.append({})
            continue

        if fill_1d:
            ax.fill_between(x_grid, pdf, color=color, alpha=0.07, lw=0)
        ax.plot(x_grid, pdf, color=color, lw=1.8, alpha=0.92)

        if show_stat:
            result = stat_fn(samples, weights)
            lo, hi = result["lo"], result["hi"]

            # Interval fill under curve
            if np.isfinite(lo) and np.isfinite(hi):
                mask = (x_grid >= lo) & (x_grid <= hi)
                ax.fill_between(x_grid, pdf, where=mask, color=color,
                                alpha=stat_kw.get("alpha", 0.25), lw=0)

            # Central-value vertical line — hidden when truths are present
            if show_median and np.isfinite(result["center"]):
                y_top = float(np.interp(result["center"], x_grid, pdf))
                ax.plot([result["center"], result["center"]], [0.0, y_top],
                        color=color,
                        lw=stat_kw.get("lw", 1.0), ls=stat_kw.get("ls", "--"),
                        alpha=min(stat_kw.get("alpha", 0.25) * 3.5, 1.0))
        else:
            result = {}

        first_results.append(result)

        if annotate_peaks:
            for px in find_pdf_peaks(x_grid, pdf):
                py = float(pdf[np.argmin(np.abs(x_grid - px))])
                ax.annotate("", xy=(px, py * 0.98), xytext=(px, 0),
                            arrowprops=dict(arrowstyle="-", color=color,
                                           lw=0.6, alpha=0.5),
                            annotation_clip=True)

    # Truth lines
    marker   = kw_truths.get("marker")
    ms       = kw_truths.get("markersize", 6)
    lkw      = _line_kw(kw_truths)
    seen: set[float] = set()
    all_vals = [v for v in tv_list if v is not None]
    for ch_idx, tv in enumerate(tv_list):
        if tv is None:
            continue
        c = lkw.get("color", "#e63946") if len(set(all_vals)) == 1 or n_chains == 1 \
            else colors[ch_idx]
        if tv not in seen:
            ax.axvline(tv, **{**lkw, "color": c})
            seen.add(tv)

    ax.set_ylim(bottom=0)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)  # 1-D marginals read cleaner without gridlines

    # Title: single chain — "{param} = $center^{+hi}_{-lo}$"
    if show_titles and show_stat and n_chains == 1 and first_results and first_results[0]:
        r = first_results[0]
        center, lo, hi = r["center"], r["lo"], r["hi"]
        if all(np.isfinite(v) for v in (center, lo, hi)):
            lo_err = center - lo
            hi_err = hi - center
            smaller = max(min(lo_err, hi_err), 1e-9)
            dp  = max(0, -int(np.floor(np.log10(smaller))) + 1)
            fmt = f".{dp}f"
            prefix = f"{param} = " if param else ""
            ax.set_title(
                f"{prefix}${center:{fmt}}^{{+{hi_err:{fmt}}}}_{{-{lo_err:{fmt}}}}$",
                fontsize=8.5, pad=5, color=txt_color,
            )


def _draw_joint(
    ax: plt.Axes,
    densities: list[np.ndarray],    # one (My,Mx) array per chain
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    colors: list[str],
    levels_2d: tuple[float, ...],
    fill_contours: bool,
    contour_lw: float,
    tv_x: list[float | None],
    tv_y: list[float | None],
    kw_truths: dict,
    show_tension: bool,
    dark: bool,
    txt_color: str,
    n_chains: int,
) -> None:
    n_lev = len(levels_2d)
    sorted_lv = sorted(levels_2d)
    pdfs_x: list[np.ndarray] = []

    for density, color in zip(densities, colors):
        if density is None or density.max() == 0:
            continue
        thresholds = density_to_levels(density, sorted_lv)
        fill_c, line_c = make_contour_colors(color, n_lev, dark=dark)

        if fill_contours and thresholds:
            fill_lvs = thresholds + [float(density.max()) * 1.001 + 1e-30]
            ax.contourf(x_grid, y_grid, density, levels=fill_lvs,
                        colors=fill_c[::-1], extend="neither")
        if thresholds:
            ax.contour(x_grid, y_grid, density, levels=thresholds,
                       colors=line_c[::-1], linewidths=contour_lw,
                       linestyles="solid")

        pdfs_x.append(density.sum(axis=0) * (y_grid[1] - y_grid[0]))

    if show_tension and len(pdfs_x) >= 2:
        ov = overlap_integral(pdfs_x[0], pdfs_x[1], x_grid[1] - x_grid[0])
        ax.text(0.97, 0.97, f"OL={ov:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=txt_color, alpha=0.7,
                bbox=dict(facecolor=mpl.rcParams["axes.facecolor"],
                          edgecolor="none", alpha=0.5, pad=1.5))

    # Truth crosshairs + optional marker
    marker = kw_truths.get("marker")
    ms     = kw_truths.get("markersize", 6)
    lkw    = _line_kw(kw_truths)
    seen: set = set()
    all_x = [v for v in tv_x if v is not None]
    all_y = [v for v in tv_y if v is not None]
    for ch_idx in range(n_chains):
        tx, ty = tv_x[ch_idx], tv_y[ch_idx]
        c = lkw.get("color", "#e63946") \
            if (n_chains == 1 or len(set(all_x)) <= 1) else colors[ch_idx]
        kw = {**lkw, "color": c}
        if tx is not None and tx not in {s[0] for s in seen}:
            ax.axvline(tx, **kw)
        if ty is not None and ty not in {s[1] for s in seen if len(s) > 1}:
            ax.axhline(ty, **kw)
        if marker and tx is not None and ty is not None:
            ax.plot(tx, ty, marker=marker, ms=ms, color=c,
                    zorder=kw_truths.get("zorder", 5) + 1)
        if tx is not None and ty is not None:
            seen.add((tx, ty))

    ax.grid(False)


def _draw_legend(
    ax: plt.Axes,
    chain_labels: list[str | None],
    colors: list[str],
    txt_color: str,
    extra_label: str | None = None,
    extra_color: str | None = None,
) -> None:
    """Minimal legend: only chain labels. Setup details live in .info()."""
    ax.set_visible(True)
    ax.axis("off")
    handles: list = []

    if any(lbl is not None for lbl in chain_labels):
        for color, lbl in zip(colors, chain_labels):
            if lbl:
                handles.append(Line2D([], [], color=color, lw=2.2, label=lbl))

    if extra_label and extra_color:
        handles.append(Line2D([], [], color=extra_color, lw=2.2,
                              ls="--", label=extra_label))

    if handles:
        ax.legend(handles=handles, loc="center", fontsize=9,
                  frameon=False, labelcolor=txt_color)


# ── Cornetto class ────────────────────────────────────────────────────────────

class Cornetto:
    """
    Fast, beautiful corner plots powered by KDExpress FFT-KDE.

    Parameters
    ----------
    data : dict[str, array or list]
        ``{param: samples}`` where each value is:

        - ``(N_post,)`` array — single chain, shared across all chains
        - ``(K, N_post)`` array — K chains; chains beyond K are absent for
          this parameter
        - ``list`` of arrays / ``None`` — explicit per-chain assignment;
          ``None`` marks an absent chain at that position

        ``n_chains`` is the maximum chain count across all parameters.
        Absent chains are silently skipped in KDE, stats, and plot panels.
    params : list[str], optional
        Subset of parameters to keep. Unused arrays are never materialised.
    labels : dict[str, str], optional
        Display labels (LaTeX supported).
    truths : dict[str, scalar or array], optional
        True/injected values. Scalar shared across chains; array length
        ``N_chains`` applies one truth per chain.
    chain_labels : list[str], optional
        Names used in the legend, one per chain.
    weights : array or list of arrays, optional
        Importance-sampling weights.
    subsample : int, optional
        Randomly thin each chain to this many samples before plotting.
    ax_lims : dict[str, (lo, hi)], optional
        Explicit axis ranges per parameter.
    max_chains : int
        Hard cap on simultaneous chains (default 10).
    stat : str or callable
        Which 1-D statistic to overlay. Built-in strings: ``"median"``,
        ``"median_mad"``, ``"median_hdi"``, ``"mean"``. Pass a callable
        ``fn(samples, weights) → dict`` with keys ``center``, ``lo``,
        ``hi``, ``label`` for a custom statistic.
    sigmas : tuple of float
        Contour levels in n-sigma (2-D convention: ``1 - exp(-n^2/2)``).
        Default: ``(1, 2)`` → 39.3% and 86.5%.
    kwargs_truths : dict, optional
        Line kwargs for truth overlays. ``marker`` / ``markersize`` draw
        a marker at the truth location.
    kwargs_stats : dict of dict, optional
        Per-stat visual kwargs, e.g.
        ``{"median": {"ls": "--", "lw": 1.2}, "hdi": {"alpha": 0.3}}``.
        Keys that don't apply to a stat type are silently ignored.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        params:        list[str] | None        = None,
        labels:        dict[str, str] | None   = None,
        truths:        dict[str, Any] | None   = None,
        chain_labels:  list[str] | None        = None,
        weights:       Any                     = None,
        subsample:     int | None              = None,
        ax_lims:       dict | None             = None,
        max_chains:    int                     = MAX_CHAINS,
        stat:          str | callable          = "median",
        sigmas:        tuple[float, ...]       = (1, 2),
        kwargs_truths: dict | None             = None,
        kwargs_stats:  dict[str, dict] | None  = None,
        delta_mode:    bool                    = False,
    ) -> None:
        resolve_stat(stat)  # raises ValueError for unknown strings

        if params is not None:
            missing = [p for p in params if p not in data]
            if missing:
                raise ValueError(f"params not found in data: {missing}")
        self._param_list, chains = _parse_data(data, max_chains,
                                               keep_params=params)

        n_chains = len(chains)

        if subsample:
            new_chains = []
            rng = np.random.default_rng()
            for ch in chains:
                if not ch:
                    new_chains.append({})
                    continue
                min_len = min(len(v) for v in ch.values())
                n = min(subsample, min_len)
                idx = rng.choice(min_len, size=n, replace=False)
                new_chains.append({p: v[idx] for p, v in ch.items()})
            chains = new_chains

        self._delta_mode = bool(delta_mode)
        if self._delta_mode:
            _apply_delta_shift(chains, truths, n_chains)
            # truth overlays disabled — all collapse to 0 anyway
            truths = None

        self._chains        = chains
        self._n_chains      = n_chains
        self._weights       = _norm_weights(weights, n_chains)
        self._ranges    = _compute_ranges(chains, self._param_list, ax_lims)
        labels = dict(labels or {})
        if self._delta_mode:
            labels = {p: _delta_label(labels.get(p, p))
                      for p in self._param_list}
        self._labels    = labels
        self._truths    = truths
        self._levels_2d = sigmas_to_levels(sigmas)
        self._stat      = stat
        self._kw_truths = _merge(_TRUTH_DEFAULTS, kwargs_truths)

        kso = kwargs_stats or {}
        self._stat_kws: dict[str, dict] = {
            **{k: _merge(v, kso.get(k)) for k, v in _STAT_DEFAULTS.items()},
            "_callable": _merge(_STAT_DEFAULTS_CALLABLE, kso.get("_callable")),
        }

        ch_lbls = list(chain_labels or [None] * n_chains)
        ch_lbls += [None] * max(0, n_chains - len(ch_lbls))
        self._chain_labels = ch_lbls[:n_chains]

        self._cache_key:  tuple | None               = None
        self._kde1d:      dict[str, list]            = {}  # param -> [pdf_ch0, ...]
        self._kde2d:      dict[tuple, list]          = {}  # (px,py) -> [dens_ch0,...]
        self._x_grids:    dict[str, np.ndarray]      = {}

        self._stats_cache: dict[str, list[dict]] | None = None

    # ── KDE computation (cached) ───────────────────────────────────────────────

    def _ensure_kdes(
        self,
        n_grid: int,
        bandwidth: float | None,
        bandwidth_scale: float,
        smooth: bool,
    ) -> None:
        key = (n_grid, bandwidth, bandwidth_scale, smooth)
        if self._cache_key == key:
            return

        self._cache_key = key
        self._kde1d.clear()
        self._kde2d.clear()
        self._x_grids.clear()

        # Build grids
        for p in self._param_list:
            self._x_grids[p] = np.linspace(*self._ranges[p], n_grid)

        # 1-D KDE
        for p in self._param_list:
            xg = self._x_grids[p]
            pdfs: list[np.ndarray] = []
            for ch, w in zip(self._chains, self._weights):
                samples = ch.get(p)
                if samples is None:
                    pdfs.append(None)
                    continue
                if smooth:
                    bw = _scaled_bw1d(samples, bandwidth_scale) \
                         if bandwidth is None else bandwidth
                    pdfs.append(kde1d(samples, xg, weights=w, bandwidth=bw))
                else:
                    counts, _ = np.histogram(
                        samples[np.isfinite(samples)],
                        bins=n_grid, range=(xg[0], xg[-1]),
                        weights=w, density=True,
                    )
                    pdfs.append(counts)
            self._kde1d[p] = pdfs

        # 2-D KDE — lower triangle pairs only; upper derived by transpose
        N = len(self._param_list)
        for row in range(N):
            for col in range(row):
                px = self._param_list[col]   # x-axis param
                py = self._param_list[row]   # y-axis param
                xg = self._x_grids[px]
                yg = self._x_grids[py]
                densities: list[np.ndarray] = []
                for ch, w in zip(self._chains, self._weights):
                    sx, sy = ch.get(px), ch.get(py)
                    if sx is None or sy is None:
                        densities.append(None)
                        continue
                    if bandwidth is None:
                        bw = _scaled_bw2d(sx, sy, bandwidth_scale)
                    else:
                        bw = (bandwidth * bandwidth_scale,) * 2
                    densities.append(kde2d(sx, sy, xg, yg,
                                          weights=w, bandwidth=bw))
                self._kde2d[(px, py)] = densities

    # ── Stats computation (cached) ─────────────────────────────────────────────

    def _ensure_stats(self) -> None:
        if self._stats_cache is not None:
            return
        cache: dict[str, list[dict]] = {}
        for p in self._param_list:
            cache[p] = [
                compute_stats(ch[p], weights=w) if p in ch else {}
                for ch, w in zip(self._chains, self._weights)
            ]
        self._stats_cache = cache

    # ── Colour helpers ─────────────────────────────────────────────────────────

    def _resolve_colors(self, color: str | list[str]) -> list[str]:
        mgr = ColorManager(dark=getattr(self, "_last_dark", False))
        return mgr.resolve_chain_colors(color, self._n_chains)

    # ── Public: plot ───────────────────────────────────────────────────────────

    def plot(
        self,
        *,
        color:          str | list[str]            = DEFAULT_PALETTE,
        dark:             bool                       = False,
        smooth:           bool                       = True,
        bandwidth:        float | None               = None,
        bandwidth_scale:  float                      = 0.8,
        n_grid:           int                        = 128,
        fill_1d:          bool                       = False,
        fill_contours:    bool                       = True,
        contour_lw:       float                      = 1.2,
        show_tension:     bool                       = False,
        annotate_peaks:   bool                       = False,
        show_titles:      bool                       = True,
        figsize:          tuple[float, float] | None = None,
        fig_size_per_dim: float                      = 2.0,
        label_pad:        float                      = 0.08,
        title:            str | None                 = None,
        fig:              plt.Figure | None          = None,
        usetex:           bool                       = False,
        use_datashader:   bool                       = False,
        tick_rotation:    float                      = 0,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Render the corner plot.

        Parameters
        ----------
        color : str or list[str]
            Colour color name or list of hex colors (one per chain).
        dark : bool
            Dark-background theme.
        smooth : bool
            KDE (True) or histogram (False) on the diagonal.
        bandwidth : float, optional
            Explicit KDE bandwidth; overrides Silverman's rule.
        bandwidth_scale : float
            Multiplier applied to the auto-estimated bandwidth (default 0.8).
        n_grid : int
            KDE evaluation grid resolution per axis (default 128).
        fill_1d : bool
            Light fill under the full 1-D KDE curve.
        fill_contours : bool
            Filled 2-D contour bands.
        contour_lw : float
            2-D contour line width.
        show_tension : bool
            Overlap-integral annotation in 2-D panels (meaningful for 2 chains).
        annotate_peaks : bool
            Mark local maxima in 1-D marginals.
        show_titles : bool
            ``value +err/-err`` title on diagonal (single-chain only).
        figsize : (w, h), optional
            Override figure size.
        fig_size_per_dim : float
            Inches per parameter when figsize is not specified (default 2.0).
        label_pad : float
            hspace / wspace between panels.
        title : str, optional
            Figure suptitle.
        fig : Figure, optional
            Draw into an existing Figure.
        usetex : bool
            Use LaTeX for text rendering (requires a working TeX installation).
        use_datashader : bool
            Overlay a rasterized scatter of raw samples under the KDE contours
            in 2-D panels. Requires ``datashader`` and ``pandas``
            (``pip install datashader pandas``). Default False.
        tick_rotation : float
            Rotate bottom-row xtick labels by this angle (degrees). Helpful
            on large corners where labels would otherwise overlap and be
            silently hidden. Default 0 (no rotation).

        Returns
        -------
        fig : matplotlib.Figure
        axes : (N, N) ndarray of Axes
        """
        self._ensure_kdes(n_grid, bandwidth, bandwidth_scale, smooth)

        colors  = self._resolve_colors(color)
        theme   = get_theme(dark)
        # Pin rcParams that matplotlib otherwise honours during Axes.__init__ —
        # grids, minor ticks, top/right spines. Each one adds measurable setup
        # cost per axis; none are part of the cornetto look.
        rc_over = {
            **theme,
            "text.usetex": usetex, "text.latex.preamble": "",
            "axes.grid": False,
            "xtick.minor.visible": False, "ytick.minor.visible": False,
            "axes.spines.top": False, "axes.spines.right": False,
        }
        txt_clr = theme["text.color"]
        N       = len(self._param_list)

        # Remember settings for .info() — reflects the actual drawn plot
        self._last_plot_kwargs = dict(
            color=color, dark=dark, smooth=smooth, bandwidth=bandwidth,
            bandwidth_scale=bandwidth_scale, n_grid=n_grid, fill_1d=fill_1d,
            fill_contours=fill_contours, contour_lw=contour_lw,
            show_tension=show_tension, annotate_peaks=annotate_peaks,
            show_titles=show_titles, fig_size_per_dim=fig_size_per_dim,
        )

        # Isolate from the user's rcParams: reset to matplotlib defaults
        # first, then apply our theme on top. This makes cornetto plots
        # look identical regardless of prior matplotlib configuration
        # (astropy, seaborn, user rc edits in the notebook, etc.).
        with plt.style.context("default"), mpl.rc_context(rc_over):
            if figsize is None:
                figsize = (fig_size_per_dim * N, fig_size_per_dim * N)
            axes: np.ndarray = np.full((N, N), None, dtype=object)
            if fig is None:
                fig = plt.figure(figsize=figsize)
                # Only create visible cells (diagonal + lower triangle + legend
                # slot). Skipping N(N-1)/2 unused axes removes the bulk of
                # matplotlib's per-axis setup cost; sharing x per column cuts
                # another ~30% on axis init.
                gs = fig.add_gridspec(N, N)
                col_sharex: list = [None] * N
                for col in range(N):
                    col_sharex[col] = fig.add_subplot(gs[col, col])
                    axes[col, col] = col_sharex[col]
                for row in range(N):
                    for col in range(row):
                        axes[row, col] = fig.add_subplot(
                            gs[row, col], sharex=col_sharex[col])
                if N > 1:
                    axes[0, N - 1] = fig.add_subplot(gs[0, N - 1])
            else:
                # Caller supplied an existing figure with all N*N axes laid out.
                axes_raw = np.asarray(fig.axes).reshape(N, N)
                for i in range(N):
                    for j in range(N):
                        axes[i, j] = axes_raw[i, j]

            fig.set_facecolor(theme["figure.facecolor"])
            for ax in axes.ravel():
                if ax is None:
                    continue
                ax.set_facecolor(theme["axes.facecolor"])
                ax.tick_params(colors=txt_clr, labelsize=7.5)
                for sp in ax.spines.values():
                    sp.set_edgecolor(theme["axes.edgecolor"])

            for row in range(N):
                for col in range(row + 1):
                    ax    = axes[row, col]
                    p_row = self._param_list[row]
                    p_col = self._param_list[col]

                    if row == col:
                        _draw_diagonal(
                            ax=ax,
                            pdfs=self._kde1d[p_row],
                            x_grid=self._x_grids[p_row],
                            chains_samples=[ch.get(p_row) for ch in self._chains],
                            chains_weights=self._weights,
                            colors=colors,
                            stat=self._stat,
                            stat_kws=self._stat_kws,
                            tv_list=_truth_scalars(self._truths, p_row,
                                                   self._n_chains),
                            kw_truths=self._kw_truths,
                            fill_1d=fill_1d,
                            annotate_peaks=annotate_peaks,
                            show_titles=show_titles,
                            txt_color=txt_clr,
                            n_chains=self._n_chains,
                            param=self._labels.get(p_row, p_row),
                            show_median=not bool(self._truths),
                        )
                        ax.set_xlim(*self._ranges[p_row])
                    else:
                        px = p_col  # x-axis (col)
                        py = p_row  # y-axis (row)
                        if use_datashader:
                            for ch, c in zip(self._chains, colors):
                                sx, sy = ch.get(px), ch.get(py)
                                if sx is None or sy is None:
                                    continue
                                _ds_raster(
                                    ax=ax,
                                    x=sx, y=sy, color=c,
                                    x_range=self._ranges[px],
                                    y_range=self._ranges[py],
                                    n_px_w=400, n_px_h=400,
                                )
                        _draw_joint(
                            ax=ax,
                            densities=self._kde2d[(px, py)],
                            x_grid=self._x_grids[px],
                            y_grid=self._x_grids[py],
                            colors=colors,
                            levels_2d=self._levels_2d,
                            fill_contours=fill_contours,
                            contour_lw=contour_lw,
                            tv_x=_truth_scalars(self._truths, px, self._n_chains),
                            tv_y=_truth_scalars(self._truths, py, self._n_chains),
                            kw_truths=self._kw_truths,
                            show_tension=show_tension,
                            dark=dark,
                            txt_color=txt_clr,
                            n_chains=self._n_chains,
                        )
                        ax.set_xlim(*self._ranges[px])
                        ax.set_ylim(*self._ranges[py])

                    ax.xaxis.set_major_locator(
                        mticker.MaxNLocator(4, prune="lower"))
                    ax.yaxis.set_major_locator(
                        mticker.MaxNLocator(4, prune="lower"))

                    if row == N - 1:
                        ax.set_xlabel(
                            self._labels.get(p_col, p_col),
                            labelpad=4, fontsize=9.5)
                    else:
                        ax.tick_params(labelbottom=False)

                    if col == 0 and row > 0:
                        ax.set_ylabel(
                            self._labels.get(p_row, p_row),
                            labelpad=4, fontsize=9.5)
                    elif row != col:
                        ax.tick_params(labelleft=False)

            if N > 1:
                _draw_legend(
                    ax=axes[0, N - 1],
                    chain_labels=self._chain_labels,
                    colors=colors,
                    txt_color=txt_clr,
                )

            if title:
                fig.suptitle(title, fontsize=11, fontweight="600", y=1.01)
            fig.subplots_adjust(hspace=label_pad, wspace=label_pad)
            try:
                fig.align_labels()
            except Exception:
                pass

        if tick_rotation:
            _apply_tick_rotation(axes, tick_rotation)

        self._last_fig   = fig
        self._last_axes  = axes
        self._last_dark  = dark
        self._last_theme = theme
        return fig, axes

    # ── Public: pairplot ───────────────────────────────────────────────────────

    def pairplot(
        self,
        other: "Cornetto",
        *,
        other_label:      str | None             = None,
        other_color:    str | list[str]        = "coral",
        bandwidth_scale:  float                  = 0.8,
        n_grid:           int                    = 128,
        bandwidth:        float | None           = None,
        fill_contours:    bool                   = True,
        contour_lw:       float                  = 1.2,
        show_tension:     bool                   = False,
        blank_diagonal:   bool                   = True,
        diag_self:        bool                   = True,
        diag_other:       bool                   = True,
        **plot_kwargs,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Overlay two Cornetto chains in a single figure.

        Layout with ``blank_diagonal=True`` (default) — an (N+1)×(N+1) grid::

            N,  1b, 2b, 2b
            1a, N,  1b, 2b
            2a, 1a, N,  1b
            2a, 2a, 1a, N

        where ``1a``/``2a`` are self's 1-D/2-D marginals, ``1b``/``2b`` are
        other's 1-D/2-D marginals, and ``N`` marks the blank separator diagonal
        that displays parameter name labels.  The bottom-right ``N`` cell holds
        the legend.

        Layout with ``blank_diagonal=False`` — a plain N×N grid::

            1a+1b, 2b,    2b
            2a,    1a+1b, 2b
            2a,    2a,    1a+1b

        ``diag_self`` and ``diag_other`` (both default True) control whether
        each chain's 1-D marginal is shown on the diagonal.

        ``other`` must be a Cornetto object whose params are a subset of
        ``self``'s params.

        Parameters
        ----------
        other : Cornetto
            Second dataset.  Its params must be a subset of ``self``'s.
        other_label : str, optional
            Legend label for ``other``.
        other_color : str or list[str]
            Colour color for ``other``.
        blank_diagonal : bool, default True
            If True use an (N+1)×(N+1) grid with a blank separator diagonal.
            If False use an N×N grid with both chains' 1-Ds on the diagonal.
        diag_self : bool, default True
            Show self's 1-D marginals on the diagonal (blank_diagonal=False).
        diag_other : bool, default True
            Show other's 1-D marginals on the diagonal (blank_diagonal=False).
        **plot_kwargs
            Recognised keys: ``color``, ``dark``, ``smooth``, ``fill_1d``,
            ``show_titles``, ``annotate_peaks``, ``label_pad``,
            ``fig_size_per_dim``, ``figsize``, ``title``.

        Returns
        -------
        fig : Figure
        axes : ndarray of shape (N+1, N+1) or (N, N)
        """
        missing = [p for p in other._param_list if p not in self._param_list]
        if missing:
            raise ValueError(
                f"`other` has params not in self: {missing}. "
                "`other` must have a subset of self's params.")

        N = len(self._param_list)

        dark        = plot_kwargs.get("dark", False)
        smooth      = plot_kwargs.get("smooth", True)
        color     = plot_kwargs.get("color", DEFAULT_PALETTE)
        fill_1d     = plot_kwargs.get("fill_1d", False)
        show_titles = plot_kwargs.get("show_titles", False)
        ann_peaks   = plot_kwargs.get("annotate_peaks", False)
        label_pad   = plot_kwargs.get("label_pad", 0.05)
        title       = plot_kwargs.get("title", "")
        fsz_per_dim = plot_kwargs.get("fig_size_per_dim", 2.0)
        tick_rotation = plot_kwargs.get("tick_rotation", 0)

        theme   = get_theme(dark)
        txt_clr = theme["text.color"]
        rc_over = {**theme, "text.usetex": False, "text.latex.preamble": ""}

        self._ensure_kdes(n_grid, bandwidth, bandwidth_scale, smooth)
        other._ensure_kdes(n_grid, bandwidth, bandwidth_scale, smooth)

        colors       = self._resolve_colors(color)
        other_colors = ColorManager(dark=dark).resolve_chain_colors(
            other_color, other._n_chains)

        # Merged ranges: ensure both chains are visible on the same axis scale
        merged_ranges: dict[str, tuple] = {}
        for p in self._param_list:
            lo, hi = self._ranges[p]
            if p in other._param_list:
                olo, ohi = other._ranges[p]
                lo, hi = min(lo, olo), max(hi, ohi)
            merged_ranges[p] = (lo, hi)

        def _plbl(p: str) -> str:
            return self._labels.get(p, p)

        def _style_ax(ax: plt.Axes) -> None:
            ax.set_facecolor(theme["axes.facecolor"])
            ax.tick_params(colors=txt_clr, labelsize=7.5)
            for sp in ax.spines.values():
                sp.set_edgecolor(theme["axes.edgecolor"])

        # ── Raw draw helpers — no label/tick management ───────────────────────

        def _self_1d_raw(ax: plt.Axes, p: str,
                         *, show_title: bool = True) -> None:
            _draw_diagonal(
                ax=ax, pdfs=self._kde1d[p], x_grid=self._x_grids[p],
                chains_samples=[ch[p] for ch in self._chains],
                chains_weights=self._weights, colors=colors,
                stat=self._stat, stat_kws=self._stat_kws,
                tv_list=_truth_scalars(self._truths, p, self._n_chains),
                kw_truths=self._kw_truths, fill_1d=fill_1d,
                annotate_peaks=ann_peaks,
                show_titles=show_title and show_titles,
                txt_color=txt_clr, n_chains=self._n_chains, param=_plbl(p),
                show_median=not bool(self._truths),
            )
            ax.set_xlim(*merged_ranges[p])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))

        def _other_1d_raw(ax: plt.Axes, p: str,
                          *, show_title: bool = False) -> bool:
            if p not in other._param_list:
                return False
            _draw_diagonal(
                ax=ax, pdfs=other._kde1d[p],
                x_grid=other._x_grids.get(
                    p, np.linspace(*merged_ranges[p], n_grid)),
                chains_samples=[ch[p] for ch in other._chains],
                chains_weights=other._weights, colors=other_colors,
                stat=other._stat, stat_kws=other._stat_kws,
                tv_list=_truth_scalars(other._truths, p, other._n_chains),
                kw_truths=other._kw_truths, fill_1d=fill_1d,
                annotate_peaks=ann_peaks,
                show_titles=show_title and show_titles,
                txt_color=txt_clr, n_chains=other._n_chains, param=_plbl(p),
                show_median=not bool(other._truths),
            )
            ax.set_xlim(*merged_ranges[p])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
            return True

        def _self_2d(ax: plt.Axes, p_x: str, p_y: str,
                     *, x_label: bool = False, y_label: bool = False) -> None:
            dens = self._kde2d.get((p_x, p_y), [])
            if not dens:
                return
            _draw_joint(
                ax=ax, densities=dens,
                x_grid=self._x_grids[p_x], y_grid=self._x_grids[p_y],
                colors=colors, levels_2d=self._levels_2d,
                fill_contours=fill_contours, contour_lw=contour_lw,
                tv_x=_truth_scalars(self._truths, p_x, self._n_chains),
                tv_y=_truth_scalars(self._truths, p_y, self._n_chains),
                kw_truths=self._kw_truths, show_tension=show_tension,
                dark=dark, txt_color=txt_clr, n_chains=self._n_chains,
            )
            ax.set_xlim(*merged_ranges[p_x])
            ax.set_ylim(*merged_ranges[p_y])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
            if x_label:
                ax.set_xlabel(_plbl(p_x), labelpad=4, fontsize=9.5)
            else:
                ax.tick_params(labelbottom=False)
            if y_label:
                ax.set_ylabel(_plbl(p_y), labelpad=4, fontsize=9.5)
            else:
                ax.tick_params(labelleft=False)

        def _other_2d(ax: plt.Axes, p_x: str, p_y: str) -> bool:
            """Draw other's 2-D KDE with p_x on x-axis, p_y on y-axis."""
            if p_x not in other._param_list or p_y not in other._param_list:
                return False
            oi = other._param_list.index(p_x)
            oj = other._param_list.index(p_y)
            if oi > oj:
                key  = (other._param_list[oj], other._param_list[oi])
                dens = [d.T for d in other._kde2d.get(key, [])]
            else:
                key  = (other._param_list[oi], other._param_list[oj])
                dens = list(other._kde2d.get(key, []))
            if not dens:
                return False
            xg = other._x_grids.get(p_x, np.linspace(*merged_ranges[p_x], n_grid))
            yg = other._x_grids.get(p_y, np.linspace(*merged_ranges[p_y], n_grid))
            _draw_joint(
                ax=ax, densities=dens, x_grid=xg, y_grid=yg,
                colors=other_colors, levels_2d=other._levels_2d,
                fill_contours=fill_contours, contour_lw=contour_lw,
                tv_x=_truth_scalars(other._truths, p_x, other._n_chains),
                tv_y=_truth_scalars(other._truths, p_y, other._n_chains),
                kw_truths=other._kw_truths, show_tension=show_tension,
                dark=dark, txt_color=txt_clr, n_chains=other._n_chains,
            )
            ax.set_xlim(*merged_ranges[p_x])
            ax.set_ylim(*merged_ranges[p_y])
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
            return True

        # ══════════════════════════════════════════════════════════════════════
        # Path A: blank_diagonal=True → (N+1) × (N+1) grid
        #
        # Grid indices 0..N.  Cell (row, col):
        #   col == row      → blank (param label if row < N; legend if row == N)
        #   col == row-1    → self  1D for param[col]      (sub-diagonal)
        #   col <  row-1    → self  2D: x=param[col], y=param[row-1]
        #   col == row+1    → other 1D for param[row]      (super-diagonal)
        #   col >  row+1    → other 2D: x=param[col-1], y=param[row]
        # ══════════════════════════════════════════════════════════════════════
        if blank_diagonal:
            grid_sz = N + 1
            figsize = plot_kwargs.get(
                "figsize", (fsz_per_dim * grid_sz, fsz_per_dim * grid_sz))

            with plt.style.context("default"), mpl.rc_context(rc_over):
                fig, axes_raw = plt.subplots(grid_sz, grid_sz,
                                             figsize=figsize, squeeze=False)
                axes: np.ndarray = np.asarray(axes_raw)

                fig.set_facecolor(theme["figure.facecolor"])
                for ax in axes.ravel():
                    _style_ax(ax)

                for row in range(grid_sz):
                    for col in range(grid_sz):
                        ax = axes[row, col]

                        if col == row:
                            # ── blank diagonal (param label or legend) ───────
                            for sp in ax.spines.values():
                                sp.set_visible(False)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_facecolor(theme["figure.facecolor"])
                            if row < N:
                                ax.text(
                                    0.5, 0.5,
                                    _plbl(self._param_list[row]),
                                    transform=ax.transAxes,
                                    ha="center", va="center",
                                    fontsize=10, color=txt_clr,
                                    fontweight="600",
                                )

                        elif col == row - 1:
                            # ── self 1D (sub-diagonal) ───────────────────────
                            p = self._param_list[col]  # = param[row-1]
                            _self_1d_raw(ax, p)
                            if row == grid_sz - 1:
                                ax.set_xlabel(_plbl(p), labelpad=4, fontsize=9.5)
                            else:
                                ax.tick_params(labelbottom=False)

                        elif col < row - 1:
                            # ── self 2D (lower-left triangle) ────────────────
                            p_x = self._param_list[col]
                            p_y = self._param_list[row - 1]
                            _self_2d(ax, p_x, p_y,
                                     x_label=(row == grid_sz - 1),
                                     y_label=(col == 0))

                        elif col == row + 1:
                            # ── other 1D (super-diagonal) ────────────────────
                            p = self._param_list[row]
                            if not _other_1d_raw(ax, p):
                                ax.set_visible(False)
                            else:
                                ax.xaxis.tick_top()
                                ax.xaxis.set_label_position("top")
                                if row == 0:
                                    ax.set_xlabel(_plbl(p), labelpad=4,
                                                  fontsize=9.5)
                                else:
                                    ax.tick_params(labelbottom=False)

                        else:
                            # ── other 2D (upper-right triangle) ──────────────
                            p_x = self._param_list[col - 1]
                            p_y = self._param_list[row]
                            if not _other_2d(ax, p_x, p_y):
                                ax.set_visible(False)
                            else:
                                ax.xaxis.tick_top()
                                ax.xaxis.set_label_position("top")
                                ax.yaxis.tick_right()
                                ax.yaxis.set_label_position("right")
                                if row == 0:
                                    ax.set_xlabel(_plbl(p_x), labelpad=4,
                                                  fontsize=9.5)
                                else:
                                    ax.tick_params(labelbottom=False)
                                if col == grid_sz - 1:
                                    ax.set_ylabel(_plbl(p_y), labelpad=4,
                                                  fontsize=9.5)
                                else:
                                    ax.tick_params(labelleft=False)

                # Overlay legend on the first self 1D panel (sub-diagonal)
                _leg_handles = []
                for _c, _lbl in zip(colors, self._chain_labels):
                    if _lbl:
                        _leg_handles.append(Line2D([], [], color=_c, lw=2.2, label=_lbl))
                _leg_handles.append(Line2D([], [], color=other_colors[0], lw=2.2,
                                           ls="--", label=other_label or "other"))
                if _leg_handles:
                    axes[1, 0].legend(handles=_leg_handles, loc="upper right",
                                      fontsize=9, frameon=False, labelcolor=txt_clr)

                if title:
                    fig.suptitle(title, fontsize=11, fontweight="600", y=1.01)
                fig.subplots_adjust(hspace=label_pad, wspace=label_pad)
                try:
                    fig.align_labels()
                except Exception:
                    pass

            self._last_fig   = fig
            self._last_axes  = axes
            self._last_dark  = dark
            self._last_theme = theme
            return fig, axes

        # ══════════════════════════════════════════════════════════════════════
        # Path B: blank_diagonal=False → N × N grid
        #
        # Cell (row, col):
        #   col == row  → self 1D (if diag_self) + other 1D (if diag_other)
        #   col <  row  → self  2D: x=param[col], y=param[row]
        #   col >  row  → other 2D: x=param[col], y=param[row]
        # ══════════════════════════════════════════════════════════════════════
        figsize = plot_kwargs.get("figsize", (fsz_per_dim * N, fsz_per_dim * N))

        with plt.style.context("default"), mpl.rc_context(rc_over):
            fig, axes_raw = plt.subplots(N, N, figsize=figsize, squeeze=False)
            axes: np.ndarray = np.asarray(axes_raw)

            fig.set_facecolor(theme["figure.facecolor"])
            for ax in axes.ravel():
                _style_ax(ax)

            for row in range(N):
                for col in range(N):
                    ax = axes[row, col]

                    if col == row:
                        # ── diagonal: overlay self and/or other 1D ───────────
                        p = self._param_list[row]
                        if diag_self:
                            _self_1d_raw(ax, p, show_title=False)
                        if diag_other:
                            _other_1d_raw(ax, p, show_title=False)
                        if not diag_self and not diag_other:
                            for sp in ax.spines.values():
                                sp.set_visible(False)
                            ax.set_xticks([])
                            ax.set_yticks([])
                        else:
                            ax.set_xlim(*merged_ranges[p])
                            ax.xaxis.set_major_locator(
                                mticker.MaxNLocator(4, prune="lower"))
                        if row == N - 1:
                            ax.set_xlabel(_plbl(p), labelpad=4, fontsize=9.5)
                        else:
                            ax.tick_params(labelbottom=False)

                    elif col < row:
                        # ── self 2D (lower triangle) ─────────────────────────
                        p_x = self._param_list[col]
                        p_y = self._param_list[row]
                        _self_2d(ax, p_x, p_y,
                                 x_label=(row == N - 1),
                                 y_label=(col == 0))

                    else:
                        # ── other 2D (upper triangle) ────────────────────────
                        p_x = self._param_list[col]
                        p_y = self._param_list[row]
                        if not _other_2d(ax, p_x, p_y):
                            ax.set_visible(False)
                        else:
                            ax.xaxis.tick_top()
                            ax.xaxis.set_label_position("top")
                            ax.yaxis.tick_right()
                            ax.yaxis.set_label_position("right")
                            if row == 0:
                                ax.set_xlabel(_plbl(p_x), labelpad=4,
                                              fontsize=9.5)
                            else:
                                ax.tick_params(labelbottom=False)
                            if col == N - 1:
                                ax.set_ylabel(_plbl(p_y), labelpad=4,
                                              fontsize=9.5)
                            else:
                                ax.tick_params(labelleft=False)

            # Overlay legend on the first diagonal 1D panel
            _leg_handles = []
            for _c, _lbl in zip(colors, self._chain_labels):
                if _lbl:
                    _leg_handles.append(Line2D([], [], color=_c, lw=2.2, label=_lbl))
            _leg_handles.append(Line2D([], [], color=other_colors[0], lw=2.2,
                                       ls="--", label=other_label or "other"))
            if _leg_handles:
                axes[0, 0].legend(handles=_leg_handles, loc="upper right",
                                  fontsize=9, frameon=False, labelcolor=txt_clr)

            if title:
                fig.suptitle(title, fontsize=11, fontweight="600", y=1.01)
            fig.subplots_adjust(hspace=label_pad, wspace=label_pad)
            try:
                fig.align_labels()
            except Exception:
                pass

        if tick_rotation:
            _apply_tick_rotation(axes, tick_rotation)

        self._last_fig   = fig
        self._last_axes  = axes
        self._last_dark  = dark
        self._last_theme = theme
        return fig, axes

    # ── Public: marginal ──────────────────────────────────────────────────────

    def marginal(
        self,
        *,
        ncols:            int                        = 5,
        color:          str | list[str]            = DEFAULT_PALETTE,
        dark:             bool                       = False,
        smooth:           bool                       = True,
        bandwidth:        float | None               = None,
        bandwidth_scale:  float                      = 0.8,
        n_grid:           int                        = 128,
        fill_1d:          bool                       = False,
        annotate_peaks:   bool                       = False,
        show_titles:      bool                       = True,
        show_legend:      bool | None                = None,
        figsize:          tuple[float, float] | None = None,
        fig_size_per_dim: float                      = 2.2,
        title:            str | None                 = None,
        usetex:           bool                       = False,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        1-D marginal distributions for all parameters in a grid layout.

        Parameters
        ----------
        ncols : int
            Panels per row (default 5). Extra cells in the last row are hidden.
        color, dark, smooth, bandwidth, bandwidth_scale, n_grid
            Same as :meth:`plot`.
        fill_1d : bool
            Fill under the full PDF curve.
        annotate_peaks : bool
            Mark local maxima in 1-D marginals.
        show_titles : bool
            Show ``value +err/-err`` title (single-chain only).
        show_legend : bool or None
            Show legend on the first panel. None (default) shows it when
            ``n_chains >= 2``.
        figsize : (w, h), optional
            Override figure size.
        fig_size_per_dim : float
            Size per panel in inches when ``figsize`` is unset.
        title : str, optional
            Figure suptitle.
        usetex : bool
            Use LaTeX text rendering.

        Returns
        -------
        fig : Figure
        axes : ndarray of shape ``(nrows, ncols)``
        """
        N        = len(self._param_list)
        n_cols   = min(N, ncols)
        n_rows   = (N + n_cols - 1) // n_cols
        show_leg = show_legend if show_legend is not None else self._n_chains >= 2

        self._ensure_kdes(n_grid, bandwidth, bandwidth_scale, smooth)
        colors  = self._resolve_colors(color)
        theme   = get_theme(dark)
        rc_over = {**theme, "text.usetex": usetex, "text.latex.preamble": ""}
        txt_clr = theme["text.color"]

        with plt.style.context("default"), mpl.rc_context(rc_over):
            if figsize is None:
                figsize = (fig_size_per_dim * n_cols,
                           fig_size_per_dim * n_rows)
            fig, axes_raw = plt.subplots(n_rows, n_cols, figsize=figsize,
                                         squeeze=False)
            axes = np.asarray(axes_raw)

            fig.set_facecolor(theme["figure.facecolor"])
            for ax in axes.ravel():
                ax.set_facecolor(theme["axes.facecolor"])
                ax.tick_params(colors=txt_clr, labelsize=7.5)
                for sp in ax.spines.values():
                    sp.set_edgecolor(theme["axes.edgecolor"])

            for idx, param in enumerate(self._param_list):
                row, col = divmod(idx, n_cols)
                ax = axes[row, col]
                _draw_diagonal(
                    ax=ax,
                    pdfs=self._kde1d[param],
                    x_grid=self._x_grids[param],
                    chains_samples=[ch[param] for ch in self._chains],
                    chains_weights=self._weights,
                    colors=colors,
                    stat=self._stat,
                    stat_kws=self._stat_kws,
                    tv_list=_truth_scalars(self._truths, param, self._n_chains),
                    kw_truths=self._kw_truths,
                    fill_1d=fill_1d,
                    annotate_peaks=annotate_peaks,
                    show_titles=show_titles,
                    txt_color=txt_clr,
                    n_chains=self._n_chains,
                    param=self._labels.get(param, param),
                    show_median=not bool(self._truths),
                )
                ax.set_xlim(*self._ranges[param])
                ax.set_xlabel(self._labels.get(param, param),
                              labelpad=4, fontsize=9.5)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))

            # Hide unused axes in last row
            for idx in range(N, n_rows * n_cols):
                r, c = divmod(idx, n_cols)
                axes[r, c].set_visible(False)

            # Figure-level legend above the grid (keeps curves clear)
            if show_leg and any(lbl for lbl in self._chain_labels):
                handles = [
                    Line2D([], [], color=c, lw=2.2, label=lbl)
                    for c, lbl in zip(colors, self._chain_labels) if lbl
                ]
                if handles:
                    fig.legend(handles=handles,
                               loc="upper center",
                               bbox_to_anchor=(0.5, 1.0),
                               ncol=min(len(handles), 6),
                               fontsize=9, frameon=False,
                               labelcolor=txt_clr)

            if title:
                fig.suptitle(title, fontsize=11, fontweight="600", y=1.01)
            fig.tight_layout()

        return fig, axes

    # ── Public: trace ──────────────────────────────────────────────────────────

    def trace(
        self,
        *,
        color:           str | list[str]            = DEFAULT_PALETTE,
        dark:              bool                       = False,
        stride:            int                        = 50,
        use_datashader:    bool                       = True,
        datashader_kwargs: dict | None                = None,
        show_legend:       bool | None                = None,
        figsize:           tuple[float, float] | None = None,
        fig_height_per_param: float                   = 1.5,
        title:             str | None                 = None,
        usetex:            bool                       = False,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Trace plot — one row per parameter, x = sample index, y = value.

        Each chain is rendered as a rasterized trace (via datashader when
        available, otherwise matplotlib with ``rasterized=True``) with a
        running median overlay.

        Parameters
        ----------
        color, dark
            Colours and theme.
        stride : int
            Running-median window = ``max(1, N_samples // stride)``.
            Also controls the number of median points plotted.
        use_datashader : bool
            Use datashader for the trace raster (default True). Falls back
            to matplotlib on ImportError.
        datashader_kwargs : dict, optional
            Forwarded to :func:`_ds_raster`. Accepted keys:
            ``n_px_w``, ``n_px_h``, ``how``, ``min_alpha``, ``max_alpha``.
        show_legend : bool or None
            Show legend on the first row. None (default) shows it when
            ``n_chains >= 2``.
        figsize : (w, h), optional
            Override figure size.
        fig_height_per_param : float
            Row height in inches when ``figsize`` is unset.
        title : str, optional
            Figure suptitle.
        usetex : bool
            Use LaTeX text rendering.

        Returns
        -------
        fig : Figure
        axes : ndarray of shape ``(N_params,)``
        """
        N        = len(self._param_list)
        colors   = self._resolve_colors(color)
        theme    = get_theme(dark)
        rc_over  = {**theme, "text.usetex": usetex, "text.latex.preamble": "",
                    "axes.grid": False}
        txt_clr  = theme["text.color"]
        show_leg = show_legend if show_legend is not None else self._n_chains >= 2
        ds_kw    = {k: v for k, v in (datashader_kwargs or {}).items()
                    if k in ("n_px_w", "n_px_h", "how", "min_alpha", "max_alpha")}

        with plt.style.context("default"), mpl.rc_context(rc_over):
            if figsize is None:
                figsize = (10.0, fig_height_per_param * N)
            fig, axes_raw = plt.subplots(N, 1, figsize=figsize, squeeze=False)
            axes = np.asarray(axes_raw).ravel()

            fig.set_facecolor(theme["figure.facecolor"])
            for ax in axes:
                ax.set_facecolor(theme["axes.facecolor"])
                ax.tick_params(colors=txt_clr, labelsize=7.5)
                for sp in ax.spines.values():
                    sp.set_edgecolor(theme["axes.edgecolor"])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            max_n = max(len(ch[self._param_list[0]]) for ch in self._chains)

            for row, param in enumerate(self._param_list):
                ax      = axes[row]
                y_range = self._ranges[param]

                for ch, color in zip(self._chains, colors):
                    samples = ch[param]
                    N_samp  = len(samples)
                    x_arr   = np.arange(N_samp, dtype=np.float64)

                    drawn = False
                    if use_datashader:
                        try:
                            _ds_raster(
                                ax=ax, x=x_arr, y=samples, color=color,
                                x_range=(0.0, float(N_samp)),
                                y_range=y_range,
                                n_px_w=ds_kw.get("n_px_w", 800),
                                n_px_h=ds_kw.get("n_px_h", 120),
                                how=ds_kw.get("how", "log"),
                                min_alpha=ds_kw.get("min_alpha", 20),
                                max_alpha=ds_kw.get("max_alpha", 200),
                            )
                            drawn = True
                        except ImportError:
                            pass
                    if not drawn:
                        ax.plot(x_arr, samples, color=color,
                                alpha=0.35, lw=0.4, rasterized=True)

                    cx, med = _running_median(samples, stride)
                    if len(cx):
                        ax.plot(cx, med, color=color, lw=1.6,
                                alpha=0.95, zorder=10)

                ax.set_xlim(0.0, float(max_n))
                ax.set_ylim(*y_range)
                ax.set_ylabel(self._labels.get(param, param),
                              labelpad=4, fontsize=9.5, color=txt_clr)
                ax.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
                if row < N - 1:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("Sample index", labelpad=4, fontsize=9.5)

            if show_leg and any(lbl for lbl in self._chain_labels):
                handles = [Line2D([], [], color=c, lw=2.2, label=lbl)
                           for c, lbl in zip(colors, self._chain_labels) if lbl]
                if handles:
                    axes[0].legend(handles=handles, loc="upper right",
                                   fontsize=9, frameon=False, labelcolor=txt_clr)

            if title:
                fig.suptitle(title, fontsize=11, fontweight="600", y=1.01)
            fig.tight_layout()

        return fig, axes

    # ── Public: trace_marginal ─────────────────────────────────────────────────

    def trace_marginal(
        self,
        *,
        color:           str | list[str]            = DEFAULT_PALETTE,
        dark:              bool                       = False,
        smooth:            bool                       = True,
        bandwidth:         float | None               = None,
        bandwidth_scale:   float                      = 0.8,
        n_grid:            int                        = 128,
        stride:            int                        = 50,
        use_datashader:    bool                       = True,
        datashader_kwargs: dict | None                = None,
        show_legend:       bool | None                = None,
        width_ratios:      tuple[float, float]        = (1.0, 4.0),
        figsize:           tuple[float, float] | None = None,
        fig_height_per_param: float                   = 1.5,
        title:             str | None                 = None,
        usetex:            bool                       = False,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Combined 1-D marginal + trace plot — one row per parameter.

        The left column shows the 1-D KDE (no CI bands, no titles) and the
        right column shows the sample trace with a running median overlay.

        Parameters
        ----------
        color, dark, smooth, bandwidth, bandwidth_scale, n_grid
            Same as :meth:`plot`.
        stride : int
            Running-median window size relative to chain length.
        use_datashader : bool
            Use datashader for traces. Falls back to matplotlib on ImportError.
        datashader_kwargs : dict, optional
            Forwarded to :func:`_ds_raster`.
        show_legend : bool or None
            Show legend on the first 1-D panel (left, row 0).
        width_ratios : (float, float)
            Relative widths of the marginal and trace columns.
        figsize : (w, h), optional
            Override figure size.
        fig_height_per_param : float
            Row height in inches when ``figsize`` is unset.
        title : str, optional
            Figure suptitle.
        usetex : bool
            Use LaTeX text rendering.

        Returns
        -------
        fig : Figure
        axes : ndarray of shape ``(N_params, 2)``
            ``axes[:, 0]`` — 1-D marginal panels (left column)
            ``axes[:, 1]`` — trace panels (right column)
        """
        N        = len(self._param_list)
        colors   = self._resolve_colors(color)
        theme    = get_theme(dark)
        rc_over  = {**theme, "text.usetex": usetex, "text.latex.preamble": "",
                    "axes.grid": False}
        txt_clr  = theme["text.color"]
        show_leg = show_legend if show_legend is not None else self._n_chains >= 2
        ds_kw    = {k: v for k, v in (datashader_kwargs or {}).items()
                    if k in ("n_px_w", "n_px_h", "how", "min_alpha", "max_alpha")}

        self._ensure_kdes(n_grid, bandwidth, bandwidth_scale, smooth)

        with plt.style.context("default"), mpl.rc_context(rc_over):
            if figsize is None:
                total_w = fig_height_per_param * (width_ratios[0] + width_ratios[1])
                figsize = (total_w, fig_height_per_param * N)
            fig, axes_raw = plt.subplots(
                N, 2, figsize=figsize, squeeze=False,
                gridspec_kw={"width_ratios": list(width_ratios)},
            )
            axes = np.asarray(axes_raw)   # shape (N, 2)

            fig.set_facecolor(theme["figure.facecolor"])
            for ax in axes.ravel():
                ax.set_facecolor(theme["axes.facecolor"])
                ax.tick_params(colors=txt_clr, labelsize=7.5)
                for sp in ax.spines.values():
                    sp.set_edgecolor(theme["axes.edgecolor"])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            max_n = max(len(ch[self._param_list[0]]) for ch in self._chains)

            for row, param in enumerate(self._param_list):
                ax_m = axes[row, 0]   # marginal panel
                ax_t = axes[row, 1]   # trace panel
                y_range = self._ranges[param]
                lbl     = self._labels.get(param, param)

                # ── Left: 1-D KDE without CI bands ───────────────────────────
                _draw_diagonal(
                    ax=ax_m,
                    pdfs=self._kde1d[param],
                    x_grid=self._x_grids[param],
                    chains_samples=[ch[param] for ch in self._chains],
                    chains_weights=self._weights,
                    colors=colors,
                    stat=self._stat,
                    stat_kws=self._stat_kws,
                    tv_list=_truth_scalars(self._truths, param, self._n_chains),
                    kw_truths=self._kw_truths,
                    fill_1d=False,
                    annotate_peaks=False,
                    show_titles=False,
                    show_stat=False,
                    txt_color=txt_clr,
                    n_chains=self._n_chains,
                    param=lbl,
                )
                ax_m.set_xlim(*y_range)
                ax_m.xaxis.set_major_locator(mticker.MaxNLocator(3, prune="lower"))
                ax_m.set_ylabel(lbl, labelpad=4, fontsize=9.5, color=txt_clr)
                if row < N - 1:
                    ax_m.tick_params(labelbottom=False)

                # ── Right: trace ──────────────────────────────────────────────
                for ch, color in zip(self._chains, colors):
                    samples = ch[param]
                    N_samp  = len(samples)
                    x_arr   = np.arange(N_samp, dtype=np.float64)

                    drawn = False
                    if use_datashader:
                        try:
                            _ds_raster(
                                ax=ax_t, x=x_arr, y=samples, color=color,
                                x_range=(0.0, float(N_samp)),
                                y_range=y_range,
                                n_px_w=ds_kw.get("n_px_w", 800),
                                n_px_h=ds_kw.get("n_px_h", 120),
                                how=ds_kw.get("how", "log"),
                                min_alpha=ds_kw.get("min_alpha", 20),
                                max_alpha=ds_kw.get("max_alpha", 200),
                            )
                            drawn = True
                        except ImportError:
                            pass
                    if not drawn:
                        ax_t.plot(x_arr, samples, color=color,
                                  alpha=0.35, lw=0.4, rasterized=True)

                    cx, med = _running_median(samples, stride)
                    if len(cx):
                        ax_t.plot(cx, med, color=color, lw=1.6,
                                  alpha=0.95, zorder=10)

                ax_t.set_xlim(0.0, float(max_n))
                ax_t.set_ylim(*y_range)
                ax_t.tick_params(labelleft=False)
                ax_t.yaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
                if row < N - 1:
                    ax_t.tick_params(labelbottom=False)
                else:
                    ax_t.set_xlabel("Sample index", labelpad=4, fontsize=9.5)

            if show_leg and any(lbl for lbl in self._chain_labels):
                handles = [Line2D([], [], color=c, lw=2.2, label=lbl)
                           for c, lbl in zip(colors, self._chain_labels) if lbl]
                if handles:
                    axes[0, 0].legend(handles=handles, loc="upper right",
                                      fontsize=9, frameon=False, labelcolor=txt_clr)

            if title:
                fig.suptitle(title, fontsize=11, fontweight="600", y=1.01)
            fig.tight_layout()

        return fig, axes

    # ── Public: summary ────────────────────────────────────────────────────────

    def summary(self) -> SummaryTable:
        """
        Return a ``SummaryTable`` with median, p16, p84, mean, std
        for every (parameter, chain) combination.

        Renders as an HTML table in Jupyter and as plain text in the terminal.
        """
        self._ensure_stats()
        return build_summary_table(
            self._chains, self._param_list,
            self._weights, self._chain_labels,
        )

    # ── Public: latex ──────────────────────────────────────────────────────────

    def latex(
        self,
        caption: str  = "",
        label:   str  = "tbl:posterior",
        fmt:     str  = ".4g",
        style:   str  = "aastex",
    ) -> str:
        """
        Return a LaTeX table string via ``astropy.io.ascii``.

        Parameters
        ----------
        caption : str
            Table caption.
        label : str
            LaTeX table label (``\\label{<label>}``).
        fmt : str
            Python format string for numeric values.
        style : str
            astropy ascii format: ``"aastex"`` (default), ``"latex"``,
            or ``"deluxetable"``.

        Returns
        -------
        str
            LaTeX source for the table.
        """
        try:
            from astropy.table import Table
            from astropy.io import ascii as asc
        except ImportError:
            raise ImportError(
                "astropy is required for latex() output.  "
                "Install with: pip install astropy"
            )

        self._ensure_stats()
        rows = []
        for p in self._param_list:
            for ch_idx, st in enumerate(self._stats_cache[p]):
                lbl = self._chain_labels[ch_idx] or f"chain_{ch_idx}"
                rows.append(dict(
                    Parameter=self._labels.get(p, p),
                    Chain=lbl,
                    Median=f"{st['median']:{fmt}}",
                    p16=f"{st['p16']:{fmt}}",
                    p84=f"{st['p84']:{fmt}}",
                    Mean=f"{st['mean']:{fmt}}",
                    Std=f"{st['std']:{fmt}}",
                ))

        t = Table(rows=rows, names=list(rows[0].keys()))
        buf = StringIO()
        asc.write(t, buf, format=style,
                  names=list(rows[0].keys()))
        src = buf.getvalue()

        # Inject caption and label
        if caption:
            src = src.replace(r"\begin{table}", f"\\begin{{table}}\n\\caption{{{caption}}}")
        if label:
            src = src.replace(r"\begin{table}", f"\\begin{{table}}\n\\label{{{label}}}")

        return src

    # ── Public: info ──────────────────────────────────────────────────────────

    def info(self) -> None:
        """
        Print the setup used by the last ``plot()`` call.

        Shows the statistic, contour sigmas, truth styling, KDE settings,
        color and theme — everything that used to be duplicated in the
        legend. Call after ``.plot()`` / ``corner(...)`` for full detail,
        or before plotting for the static configuration.
        """
        stat_label = self._stat if isinstance(self._stat, str) \
            else getattr(self._stat, "__name__", "custom")
        sig_strs = []
        for lv in self._levels_2d:
            n = (-2.0 * np.log(1.0 - lv)) ** 0.5
            sig_strs.append(f"{n:.0f}σ→{lv*100:.1f}%")
        lines = [
            "Cornetto setup",
            "--------------",
            f"  params         : {self._param_list}",
            f"  n_chains       : {self._n_chains}",
            f"  chain_labels   : {self._chain_labels}",
            f"  stat           : {stat_label}  "
            f"(center line + shaded interval on 1-D marginals)",
            f"  contour sigmas : {', '.join(sig_strs)}",
        ]
        if self._truths:
            tk = self._kw_truths
            lines.append(
                f"  truths         : color={tk.get('color')}, "
                f"ls={tk.get('ls')!r}, lw={tk.get('lw')}"
                f"{', marker='+repr(tk['marker']) if tk.get('marker') else ''}"
            )
        else:
            lines.append("  truths         : none")
        kw = getattr(self, "_last_plot_kwargs", None)
        if kw is not None:
            lines += [
                "  ── last plot() ──",
                f"  color        : {kw['color']}",
                f"  theme          : {'dark' if kw['dark'] else 'light'}",
                f"  smooth         : {kw['smooth']}  "
                f"(KDE if True, histogram if False)",
                f"  n_grid         : {kw['n_grid']}",
                f"  bandwidth      : "
                f"{'Silverman × '+str(kw['bandwidth_scale']) if kw['bandwidth'] is None else kw['bandwidth']}",
                f"  fill_1d        : {kw['fill_1d']}",
                f"  fill_contours  : {kw['fill_contours']}",
            ]
        else:
            lines.append("  ── plot() not yet called ──")
        print("\n".join(lines))

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Cornetto("
            f"params={self._param_list}, "
            f"n_chains={self._n_chains}, "
            f"stat={self._stat!r}, "
            f"sigmas_levels={tuple(f'{v:.3f}' for v in self._levels_2d)}"
            f")"
        )


# ── Convenience wrappers ──────────────────────────────────────────────────────
#
# Each wrapper is a two-liner: split kwargs into Cornetto.__init__ args and
# method args, then call. Analysis kwargs (params, truths, stat, ...) route
# to the constructor; visual kwargs (color, dark, smooth, ...) route to the
# method. See ``Cornetto`` and its methods for full parameter documentation.

_INIT_KEYS: frozenset[str] = frozenset({
    "params", "labels", "truths", "chain_labels", "weights",
    "subsample", "ax_lims", "max_chains", "stat", "sigmas",
    "kwargs_truths", "kwargs_stats", "delta_mode",
})


def _split_init(kwargs: dict) -> tuple[dict, dict]:
    init = {k: kwargs.pop(k) for k in list(kwargs) if k in _INIT_KEYS}
    return init, kwargs


def corner(data: dict[str, Any], **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """One-shot ``Cornetto(data, ...).plot(...)`` — see :class:`Cornetto`."""
    init, plot_kw = _split_init(kwargs)
    return Cornetto(data, **init).plot(**plot_kw)


# ── quick_corner: fastest possible path ───────────────────────────────────────
#
# Profiling the default `corner()` on a 10-param × 50k-sample dataset shows
# the dominant cost (~85%) is the FFT-KDE evaluations for the N*(N-1)/2
# lower-triangle 2-D panels on a 128×128 grid, plus bandwidth estimation.
# Secondary costs are per-event stat evaluation, fill_between, contourf/contour
# with multiple levels, titles, peak detection, and axis cosmetics.
#
# `quick_corner` skips every one of those paths: raw histograms on a coarse
# grid, one contour line per sigma, no fills, no stats, no titles, no KDE.
# It produces a recognisable corner plot in a small fraction of the time and
# is O(N_params²) in histogram ops only — no KDE cost at all.

def quick_corner(
    data: dict[str, Any],
    *,
    params:        list[str] | None    = None,
    labels:        dict[str, str] | None = None,
    chain_labels:  list[str] | None    = None,
    truths:        dict[str, Any] | None = None,
    weights:       Any                  = None,
    ax_lims:       dict | None          = None,
    subsample:     int | None           = 20_000,
    bins:          int                  = 30,
    sigmas:        tuple[float, ...]    = (1, 2),
    stat:          str | callable       = "median",
    kwargs_stats:  dict[str, dict] | None = None,
    kwargs_truths: dict | None             = None,
    color:       str | list[str]      = DEFAULT_PALETTE,
    dark:          bool                 = False,
    figsize:       tuple[float, float] | None = None,
    fig_size_per_dim: float             = 1.8,
    title:         str | None           = None,
    fig:           plt.Figure | None    = None,
    usetex:        bool                 = False,
    tick_rotation: float                = 0,
    delta_mode:    bool                 = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Fastest corner plot in cornetto — histograms-only, any dimension.

    Use this when iterating: it skips KDE, fills, titles, peak detection and
    tension annotations. Switch to ``corner()`` / ``Cornetto`` when you want
    the publication-ready version.

    Parameters
    ----------
    data, params, labels, chain_labels, truths, weights, ax_lims, subsample
        Same semantics as ``corner()``.
    bins : int
        Histogram bins per axis (default 40). Lower is faster.
    sigmas : tuple of float
        One contour line per sigma on 2-D panels.
    stat : str or callable
        Central-value + interval statistic drawn on each diagonal panel.
        Built-ins: ``"median"`` (default), ``"median_mad"``, ``"median_hdi"``,
        ``"mean"``. Or pass a callable ``fn(samples, weights) -> dict`` with
        keys ``center``, ``lo``, ``hi``, ``label``.
    kwargs_stats : dict of dict, optional
        Per-stat visual overrides ``{stat_name: {"lw", "ls", "alpha"}}``.
    color, dark
        Colours and theme.
    figsize, fig_size_per_dim, title, fig
        Figure sizing / override.
    usetex : bool
        Use LaTeX for text rendering (requires a working TeX installation).

    Returns
    -------
    fig, axes
    """
    resolve_stat(stat)  # validate string keys up front
    stat_fn = resolve_stat(stat)
    kso = kwargs_stats or {}
    stat_kws: dict[str, dict] = {
        **{k: _merge(v, kso.get(k)) for k, v in _STAT_DEFAULTS.items()},
        "_callable": _merge(_STAT_DEFAULTS_CALLABLE, kso.get("_callable")),
    }
    stat_kw = _stat_kw_for(stat, stat_kws)
    kw_truths = _merge(_TRUTH_DEFAULTS, kwargs_truths)
    fill_alpha = stat_kw.get("alpha", 0.25)
    line_lw    = stat_kw.get("lw", 1.0)
    line_ls    = stat_kw.get("ls", "--")
    line_alpha = min(fill_alpha * 3.5, 1.0)
    param_list, chains = _parse_data(data, MAX_CHAINS, keep_params=params)
    n_chains = len(chains)

    if delta_mode:
        _apply_delta_shift(chains, truths, n_chains)
        truths = None
        labels = {p: _delta_label((labels or {}).get(p, p))
                  for p in param_list}

    if subsample:
        rng = np.random.default_rng()
        new_chains = []
        for ch in chains:
            min_len = min(len(v) for v in ch.values())
            n = min(subsample, min_len)
            idx = rng.choice(min_len, size=n, replace=False) \
                if n < min_len else slice(None)
            new_chains.append({p: v[idx] for p, v in ch.items()})
        chains = new_chains

    weights_list = _norm_weights(weights, n_chains)
    ranges       = _compute_ranges(chains, param_list, ax_lims)
    levels_2d    = sigmas_to_levels(sigmas)
    theme        = get_theme(dark)
    txt_clr      = theme["text.color"]
    labels       = labels or {}

    colors = ColorManager(dark=dark).resolve_chain_colors(color, n_chains)

    ch_labels = list(chain_labels or [None] * n_chains)
    ch_labels += [None] * max(0, n_chains - len(ch_labels))
    ch_labels = ch_labels[:n_chains]

    N = len(param_list)
    # Pin rcParams to values that keep mpl in its fastest path:
    # grids, minor ticks and top/right spines all add non-trivial axis init
    # cost, and we don't need them in quick mode.
    rc_over = {
        **theme,
        "text.usetex": usetex, "text.latex.preamble": "",
        "axes.grid": False, "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
    }
    with plt.style.context("default"), mpl.rc_context(rc_over):
        if figsize is None:
            figsize = (fig_size_per_dim * N, fig_size_per_dim * N)
        if fig is None:
            fig = plt.figure(figsize=figsize)
        fig.set_facecolor(theme["figure.facecolor"])

        # Only create visible cells (diagonal + lower triangle + legend slot).
        # Skipping N(N-1)/2 unused axes removes most of matplotlib's overhead
        # in per-axis setup (Transforms, Axis.__init__, tick formatters).
        # Share x per column so mpl only reconfigures tick machinery once
        # per column — another ~30% cut on axis setup in our profiles.
        gs = fig.add_gridspec(N, N, hspace=0.08, wspace=0.08)
        axes: np.ndarray = np.full((N, N), None, dtype=object)
        col_sharex: list = [None] * N
        for col in range(N):
            col_sharex[col] = fig.add_subplot(gs[col, col])  # diagonal first
            axes[col, col] = col_sharex[col]
        for row in range(N):
            for col in range(row):
                axes[row, col] = fig.add_subplot(gs[row, col],
                                                 sharex=col_sharex[col])
        if N > 1:
            axes[0, N - 1] = fig.add_subplot(gs[0, N - 1])

        ax_face = theme["axes.facecolor"]
        ax_edge = theme["axes.edgecolor"]
        for ax in axes.ravel():
            if ax is None:
                continue
            ax.set_facecolor(ax_face)
            for sp in ax.spines.values():
                sp.set_edgecolor(ax_edge)

        # Pre-compute 1-D histograms + stat endpoints per (param, chain).
        # Done once in numpy; mpl just draws the result.
        hist1d:  dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        ci1d:    dict[str, list[tuple[float, float, float]]]    = {}
        for p in param_list:
            lo, hi = ranges[p]
            edges  = np.linspace(lo, hi, bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hist1d[p] = []
            ci1d[p]   = []
            for ch, w in zip(chains, weights_list):
                vals = ch[p]
                finite = np.isfinite(vals)
                v = vals[finite]
                ww = None if w is None else w[finite]
                h, _ = np.histogram(v, bins=edges, weights=ww, density=True)
                hist1d[p].append((centers, h))
                if v.size < 2:
                    ci1d[p].append((np.nan, np.nan, np.nan))
                    continue
                st = stat_fn(v, ww)
                ci1d[p].append((float(st["lo"]), float(st["center"]),
                                float(st["hi"])))

        sorted_lv = sorted(levels_2d)

        for row in range(N):
            for col in range(row + 1):
                ax    = axes[row, col]
                p_row = param_list[row]
                p_col = param_list[col]

                if row == col:
                    for (x, h), color, (lo, med, hi) in zip(
                            hist1d[p_row], colors, ci1d[p_row]):
                        ax.plot(x, h, color=color, lw=1.5, alpha=0.95,
                                drawstyle="steps-mid")
                        if np.isfinite(lo) and np.isfinite(hi):
                            mask = (x >= lo) & (x <= hi)
                            ax.fill_between(x, h, where=mask, step="mid",
                                            color=color, alpha=fill_alpha,
                                            lw=0)
                        if np.isfinite(med):
                            y_top = float(np.interp(med, x, h))
                            ax.plot([med, med], [0.0, y_top], color=color,
                                    lw=line_lw, ls=line_ls, alpha=line_alpha)
                    tv_list = _truth_scalars(truths, p_row, n_chains)
                    lkw = _line_kw(kw_truths)
                    all_vals = [v for v in tv_list if v is not None]
                    seen_1d: set[float] = set()
                    for ch_idx, tv in enumerate(tv_list):
                        if tv is None or tv in seen_1d:
                            continue
                        c = lkw.get("color", "#e63946") \
                            if (len(set(all_vals)) == 1 or n_chains == 1) \
                            else colors[ch_idx]
                        ax.axvline(tv, **{**lkw, "color": c})
                        seen_1d.add(tv)
                    ax.set_xlim(*ranges[p_row])
                    ax.set_ylim(bottom=0)
                    ax.yaxis.set_visible(False)
                else:
                    px, py = p_col, p_row
                    x_edges = np.linspace(*ranges[px], bins + 1)
                    y_edges = np.linspace(*ranges[py], bins + 1)
                    x_ctr   = 0.5 * (x_edges[:-1] + x_edges[1:])
                    y_ctr   = 0.5 * (y_edges[:-1] + y_edges[1:])
                    for ch, w, color in zip(chains, weights_list, colors):
                        xv, yv = ch[px], ch[py]
                        mask   = np.isfinite(xv) & np.isfinite(yv)
                        H, _, _ = np.histogram2d(
                            xv[mask], yv[mask],
                            bins=[x_edges, y_edges],
                            weights=None if w is None else w[mask],
                        )
                        density = H.T  # (Ny, Nx)
                        if density.max() <= 0:
                            continue
                        thr = density_to_levels(density, sorted_lv)
                        # Make strictly increasing (required by contour)
                        thr = sorted(set(t for t in thr if t > 0))
                        if len(thr) >= 1:
                            ax.contour(x_ctr, y_ctr, density, levels=thr,
                                       colors=[color], linewidths=1.1,
                                       alpha=0.9)
                    tvx = _truth_scalars(truths, px, n_chains)
                    tvy = _truth_scalars(truths, py, n_chains)
                    marker = kw_truths.get("marker")
                    ms     = kw_truths.get("markersize", 6)
                    lkw    = _line_kw(kw_truths)
                    all_x = [v for v in tvx if v is not None]
                    seen_2d: set = set()
                    for ch_idx in range(n_chains):
                        tx, ty = tvx[ch_idx], tvy[ch_idx]
                        c = lkw.get("color", "#e63946") \
                            if (n_chains == 1 or len(set(all_x)) <= 1) \
                            else colors[ch_idx]
                        kw = {**lkw, "color": c}
                        if tx is not None and tx not in {s[0] for s in seen_2d}:
                            ax.axvline(tx, **kw)
                        if ty is not None and ty not in {s[1] for s in seen_2d if len(s) > 1}:
                            ax.axhline(ty, **kw)
                        if marker and tx is not None and ty is not None:
                            ax.plot(tx, ty, marker=marker, ms=ms, color=c,
                                    zorder=kw_truths.get("zorder", 5) + 1)
                        if tx is not None and ty is not None:
                            seen_2d.add((tx, ty))
                    ax.set_xlim(*ranges[px])
                    ax.set_ylim(*ranges[py])

                ax.xaxis.set_major_locator(
                    mticker.MaxNLocator(3, prune="lower"))
                ax.yaxis.set_major_locator(
                    mticker.MaxNLocator(3, prune="lower"))

                if row == N - 1:
                    ax.set_xlabel(labels.get(p_col, p_col),
                                  labelpad=3, fontsize=9)
                else:
                    ax.tick_params(labelbottom=False)

                if col == 0 and row > 0:
                    ax.set_ylabel(labels.get(p_row, p_row),
                                  labelpad=3, fontsize=9)
                elif row != col:
                    ax.tick_params(labelleft=False)

        if N > 1 and axes[0, N - 1] is not None:
            _draw_legend(
                ax=axes[0, N - 1],
                chain_labels=ch_labels,
                colors=colors,
                txt_color=txt_clr,
            )

        if title:
            fig.suptitle(title, fontsize=10, fontweight="600", y=1.01)

    if tick_rotation:
        _apply_tick_rotation(axes, tick_rotation)

    return fig, axes


def marginal(data: dict[str, Any], **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """One-shot ``Cornetto(data, ...).marginal(...)`` — see :class:`Cornetto`."""
    init, method_kw = _split_init(kwargs)
    return Cornetto(data, **init).marginal(**method_kw)


def trace(data: dict[str, Any], **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """One-shot ``Cornetto(data, ...).trace(...)`` — see :class:`Cornetto`."""
    init, method_kw = _split_init(kwargs)
    return Cornetto(data, **init).trace(**method_kw)


def trace_marginal(data: dict[str, Any], **kwargs) -> tuple[plt.Figure, np.ndarray]:
    """One-shot ``Cornetto(data, ...).trace_marginal(...)`` — see :class:`Cornetto`."""
    init, method_kw = _split_init(kwargs)
    return Cornetto(data, **init).trace_marginal(**method_kw)
