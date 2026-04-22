"""cornetto.stats — statistical helpers and stat descriptors."""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks as _find_peaks


# ── Sigma ↔ 2D probability-mass conversion ────────────────────────────────────

def sigmas_to_levels(sigmas: tuple[float, ...]) -> tuple[float, ...]:
    """
    Convert n-sigma values to the correct 2D probability-mass levels.

    For a 2D Gaussian the probability enclosed within the n-sigma ellipse is:
        P(n) = 1 - exp(-n² / 2)

    This differs from the 1D convention (1σ ↔ 68.27%).  In 2D:
        1σ → 39.35%,  2σ → 86.47%,  3σ → 98.89%

    See https://corner.readthedocs.io/en/stable/pages/sigmas/
    """
    return tuple(float(1.0 - np.exp(-0.5 * n ** 2)) for n in sigmas)


# Default contour levels: 1σ and 2σ in the 2D sense
DEFAULT_LEVELS = sigmas_to_levels((1, 2))   # (0.3935, 0.8647)


# ── Credible-region level conversion ──────────────────────────────────────────

def density_to_levels(density: np.ndarray, levels: tuple[float, ...]) -> list[float]:
    """
    Convert probability-mass levels (e.g. 0.68, 0.95) to density thresholds.

    Returns thresholds sorted *ascending* (outer → inner), ready for contourf.
    """
    flat = density.ravel()
    sorted_desc = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_desc)
    total = cumsum[-1]
    if total == 0:
        return [0.0] * len(levels)
    cumsum /= total

    thresholds: list[float] = []
    for lv in sorted(levels):
        idx = int(np.searchsorted(cumsum, lv))
        idx = min(idx, len(sorted_desc) - 1)
        thresholds.append(float(sorted_desc[idx]))
    return sorted(thresholds)


# ── HDI helper ────────────────────────────────────────────────────────────────

def hdi(samples: np.ndarray, prob: float = 0.68) -> tuple[float, float]:
    """
    Highest Density Interval — the *shortest* interval containing `prob`
    of the posterior mass.
    """
    s = np.sort(samples[np.isfinite(samples)])
    n = len(s)
    if n < 2:
        return float(s[0]), float(s[0])
    n_in = max(1, int(np.ceil(prob * n)))
    if n_in >= n:
        return float(s[0]), float(s[-1])
    widths = s[n_in:] - s[:n - n_in]
    idx = int(np.argmin(widths))
    return float(s[idx]), float(s[idx + n_in])


# ── Stat descriptors ──────────────────────────────────────────────────────────
# Each returns {"center": float, "lo": float, "hi": float, "label": str}
# center → vertical line position
# lo, hi → interval shaded under the 1-D KDE curve

def stat_median(samples: np.ndarray, weights=None) -> dict:
    """Median with 16th / 84th percentile interval."""
    s = samples[np.isfinite(samples)]
    if len(s) < 2:
        return dict(center=np.nan, lo=np.nan, hi=np.nan, label="median")
    center = float(np.median(s))
    lo     = float(np.quantile(s, 0.16))
    hi     = float(np.quantile(s, 0.84))
    return dict(center=center, lo=lo, hi=hi, label="median")


def stat_median_mad(samples: np.ndarray, weights=None) -> dict:
    """Median ± MAD (robust, symmetric)."""
    s = samples[np.isfinite(samples)]
    if len(s) < 2:
        return dict(center=np.nan, lo=np.nan, hi=np.nan, label="median±MAD")
    center = float(np.median(s))
    mad    = float(np.median(np.abs(s - center)))
    return dict(center=center, lo=center - mad, hi=center + mad, label="median±MAD")


def stat_median_hdi(samples: np.ndarray, weights=None, prob: float = 0.68) -> dict:
    """Median with HDI interval.  Bind ``prob`` via functools.partial."""
    s = samples[np.isfinite(samples)]
    label = f"median+HDI{int(prob * 100)}%"
    if len(s) < 2:
        return dict(center=np.nan, lo=np.nan, hi=np.nan, label=label)
    center = float(np.median(s))
    lo, hi = hdi(s, prob=prob)
    return dict(center=center, lo=lo, hi=hi, label=label)


def stat_mean(samples: np.ndarray, weights=None) -> dict:
    """Mean ± standard deviation."""
    s = samples[np.isfinite(samples)]
    if len(s) < 2:
        return dict(center=np.nan, lo=np.nan, hi=np.nan, label="mean±std")
    center = float(np.mean(s))
    std    = float(np.std(s))
    return dict(center=center, lo=center - std, hi=center + std, label="mean±std")


STAT_REGISTRY: dict[str, callable] = {
    "median":     stat_median,
    "median_mad": stat_median_mad,
    "median_hdi": stat_median_hdi,
    "mean":       stat_mean,
}


def resolve_stat(s) -> callable:
    """Return a callable stat function from a string key or a user callable."""
    if callable(s):
        return s
    if s in STAT_REGISTRY:
        return STAT_REGISTRY[s]
    raise ValueError(
        f"Unknown stat {s!r}.  Built-ins: {list(STAT_REGISTRY)}.  "
        "Pass a callable fn(samples, weights) → dict instead."
    )


# ── Simple summary stats (always computed for the summary table) ───────────────

def compute_stats(samples: np.ndarray, weights=None) -> dict:
    """
    Compute a standard fixed set of posterior statistics.

    Returns: mean, median, std, p16, p84.
    """
    s = samples[np.isfinite(samples)]
    if len(s) == 0:
        nan = float("nan")
        return dict(mean=nan, median=nan, std=nan, p16=nan, p84=nan)

    if weights is not None:
        w = weights[np.isfinite(samples)] if len(weights) == len(samples) else None
        if w is not None:
            w = w / w.sum()
            mean   = float(np.average(s, weights=w))
            cdf    = np.cumsum(w[np.argsort(s)])
            median = float(np.sort(s)[np.searchsorted(cdf, 0.5)])
            std    = float(np.sqrt(np.average((s - mean) ** 2, weights=w)))
        else:
            w = None

    if weights is None or w is None:
        mean   = float(np.mean(s))
        median = float(np.median(s))
        std    = float(np.std(s))

    p16 = float(np.quantile(s, 0.16))
    p84 = float(np.quantile(s, 0.84))

    return dict(mean=mean, median=median, std=std, p16=p16, p84=p84)


# ── Summary table ──────────────────────────────────────────────────────────────

class SummaryTable:
    """
    Tabular summary of posterior statistics, one row per (parameter, chain).

    Columns: median, p16, p84, mean, std.
    Renders as a formatted string in the terminal and as an HTML table in Jupyter.
    """

    def __init__(self, data: dict[str, dict]):
        # data: {param: {chain_label: stats_dict}}
        self._data   = data
        self._params = list(data.keys())
        self._chains = list(next(iter(data.values())).keys()) if data else []

    def __getitem__(self, param: str) -> dict:
        return self._data[param]

    def params(self) -> list[str]:
        return self._params

    def chains(self) -> list[str]:
        return self._chains

    def _rows(self) -> list[list[str]]:
        hdr  = ["param", "chain", "median", "p16", "p84", "mean", "std"]
        rows = [hdr]
        for p in self._params:
            for chain, st in self._data[p].items():
                rows.append([
                    p, chain,
                    f"{st['median']:.4g}",
                    f"{st['p16']:.4g}",
                    f"{st['p84']:.4g}",
                    f"{st['mean']:.4g}",
                    f"{st['std']:.4g}",
                ])
        return rows

    def __str__(self) -> str:
        rows   = self._rows()
        widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
        sep    = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        lines  = [sep]
        for i, row in enumerate(rows):
            line = "| " + " | ".join(cell.ljust(widths[j])
                                     for j, cell in enumerate(row)) + " |"
            lines.append(line)
            if i == 0:
                lines.append(sep)
        lines.append(sep)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def _repr_html_(self) -> str:
        rows = self._rows()
        hdr  = rows[0]
        body = rows[1:]
        th   = "".join(
            f"<th style='padding:4px 10px;text-align:left'>{h}</th>" for h in hdr)
        trs  = []
        for i, row in enumerate(body):
            bg = "#f7f7fa" if i % 2 == 0 else "#ffffff"
            td = "".join(
                f"<td style='padding:3px 10px;font-family:monospace'>{c}</td>"
                for c in row)
            trs.append(f"<tr style='background:{bg}'>{td}</tr>")
        return (
            "<table style='border-collapse:collapse;font-size:13px'>"
            f"<thead><tr style='background:#ebebf5'>{th}</tr></thead>"
            f"<tbody>{''.join(trs)}</tbody></table>"
        )


def build_summary_table(
    chains: list[dict[str, np.ndarray]],
    params: list[str],
    weights_list: list,
    chain_labels: list,
) -> SummaryTable:
    data: dict[str, dict] = {}
    for p in params:
        data[p] = {}
        for ch_idx, (ch, w) in enumerate(zip(chains, weights_list)):
            lbl = chain_labels[ch_idx] or f"chain_{ch_idx}"
            data[p][lbl] = compute_stats(ch.get(p, np.array([])), weights=w)
    return SummaryTable(data)


# ── Misc helpers ───────────────────────────────────────────────────────────────

def overlap_integral(pdf1: np.ndarray, pdf2: np.ndarray, dx: float) -> float:
    overlap = float(np.sum(np.minimum(pdf1, pdf2)) * dx)
    norm    = min(float(np.sum(pdf1) * dx), float(np.sum(pdf2) * dx))
    return 0.0 if norm == 0 else min(overlap / norm, 1.0)


def find_pdf_peaks(x: np.ndarray, pdf: np.ndarray,
                   min_prominence: float = 0.15) -> np.ndarray:
    if pdf.max() == 0:
        return np.array([])
    peaks, _ = _find_peaks(pdf, prominence=min_prominence * float(pdf.max()))
    return x[peaks]
