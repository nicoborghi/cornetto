"""cornetto.kde — KDExpress FFT-KDE wrappers."""

from __future__ import annotations
import warnings
import numpy as np
import KDExpress as kdx


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def kde1d(
    data: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray | None = None,
    bandwidth: float | None = None,
) -> np.ndarray:
    data, grid = _np(data), _np(grid)
    mask = np.isfinite(data)
    data = data[mask]
    if weights is not None:
        weights = _np(weights)[mask]
        s = weights.sum()
        weights = weights / s if s > 0 else None
    if len(data) < 2:
        return np.zeros_like(grid)
    try:
        # Extend evaluation grid to prevent FFT wrap-around artifacts at tails.
        # FFT-KDE assumes periodic boundaries, so density leaking off one edge
        # folds onto the other, lifting both tails. Computing on a wider grid
        # and interpolating back keeps the returned values clean.
        n = len(grid)
        span = grid[-1] - grid[0]
        pad_frac = 0.4
        n_pad = max(int(n * pad_frac), 32)
        ext_lo = grid[0]  - pad_frac * span
        ext_hi = grid[-1] + pad_frac * span
        ext_grid = np.linspace(ext_lo, ext_hi, n + 2 * n_pad)
        raw = np.clip(_np(kdx.fft_kde1d(ext_grid, data, weights=weights, bw=bandwidth)), 0, None)
        return np.interp(grid, ext_grid, raw)
    except Exception as e:
        warnings.warn(f"KDExpress 1D KDE failed ({e}); falling back to histogram.", RuntimeWarning, stacklevel=2)
        counts, _ = np.histogram(data, bins=len(grid), range=(grid[0], grid[-1]),
                                 weights=weights, density=True)
        return counts


def kde2d(
    data_x: np.ndarray,
    data_y: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    weights: np.ndarray | None = None,
    bandwidth: tuple[float, float] | float | None = None,
) -> np.ndarray:
    """Returns density shape (My, Mx) — matplotlib convention."""
    data_x, data_y = _np(data_x), _np(data_y)
    x_grid, y_grid = _np(x_grid), _np(y_grid)
    mask = np.isfinite(data_x) & np.isfinite(data_y)
    data_x, data_y = data_x[mask], data_y[mask]
    if weights is not None:
        weights = _np(weights)[mask]
        s = weights.sum()
        weights = weights / s if s > 0 else None
    if len(data_x) < 4:
        return np.zeros((len(y_grid), len(x_grid)))
    n = len(data_x)
    w = np.ones(n, dtype=np.float64) / n if weights is None else weights
    data_2d = np.column_stack([data_x, data_y])
    if isinstance(bandwidth, (int, float)):
        bandwidth = (float(bandwidth), float(bandwidth))
    try:
        raw = _np(kdx.fft_kde2d(x_grid, y_grid, data_2d, weights=w, bw=bandwidth))
        return np.clip(raw.T, 0, None)   # (Mx,My).T → (My,Mx)
    except Exception as e:
        warnings.warn(f"KDExpress 2D KDE failed ({e}); falling back to histogram.", RuntimeWarning, stacklevel=2)
        H, _, _ = np.histogram2d(data_x, data_y,
                                  bins=[len(x_grid), len(y_grid)],
                                  range=[[x_grid[0], x_grid[-1]], [y_grid[0], y_grid[-1]]],
                                  weights=w, density=True)
        return H.T
