"""cornetto.styles — colour palettes and rcParam helpers."""

from __future__ import annotations
import colorsys
import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_hex, to_rgba

MAX_CHAINS: int = 10

# Original color — 10 hues spaced by the golden angle (137.5°) starting
# from #0072b2, with per-hue lightness/saturation tuned for equal perceived
# luminance across the full cycle.
CORNETTO_PALETTE: list[str] = [
    "#0072b2",  # ocean blue      (201.6° — anchor)
    "#aa1348",  # crimson rose    (339.1°)
    "#158b0e",  # forest green    (116.6°)
    "#421eb8",  # deep indigo     (254.1°)
    "#c06b0c",  # amber           ( 31.6°)
    "#0da086",  # sea teal        (169.1°)
    "#af1d9f",  # fuchsia         (306.6°)
    "#5e9014",  # olive           ( 84.2°)
    "#1848b4",  # slate blue      (221.7°)
    "#b21012",  # vermillion      (359.2°)
]

# Backward-compatible named aliases → hex colour (just the base colour now;
# all level/line variants are derived algorithmically at render time).
PALETTES: dict[str, str] = {
    "cornetto": "#0072b2",
    "indigo":   "#421eb8",
    "coral":    "#aa1348",
    "teal":     "#0da086",
    "gold":     "#c06b0c",
    "ink":      "#1e1b4b",
}

DEFAULT_PALETTE = "cornetto"

_LIGHT = {
    "figure.facecolor":  "#ffffff",
    "figure.dpi":        100,
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#111111",
    "axes.grid":         True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.4,
    "grid.alpha":        0.8,
    "xtick.color":       "#111111",
    "ytick.color":       "#111111",
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "text.color":        "#111111",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1,
}

_DARK = {
    "figure.facecolor":  "#0d0d1a",
    "figure.dpi":        100,
    "axes.facecolor":    "#0d0d1a",
    "axes.edgecolor":    "#2a2a45",
    "axes.labelcolor":   "#d8d6f5",
    "axes.grid":         True,
    "grid.color":        "#1e1e35",
    "grid.linewidth":    0.4,
    "grid.alpha":        0.8,
    "xtick.color":       "#d8d6f5",
    "ytick.color":       "#d8d6f5",
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "text.color":        "#d8d6f5",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1,
}


def get_theme(dark: bool = False) -> dict:
    return dict(_DARK if dark else _LIGHT)


def apply_theme(dark: bool = False) -> dict:
    theme = get_theme(dark)
    safe = {k: v for k, v in theme.items() if not k.startswith("font.")}
    mpl.rcParams.update(safe)
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CORNETTO_PALETTE)
    return theme


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _mix_rgb(c1: tuple[float, float, float],
             c2: tuple[float, float, float],
             t: float) -> tuple[float, float, float]:
    t = _clamp01(t)
    return (
        c1[0] * (1.0 - t) + c2[0] * t,
        c1[1] * (1.0 - t) + c2[1] * t,
        c1[2] * (1.0 - t) + c2[2] * t,
    )


def _adjust_hls(
    color: tuple[float, float, float],
    *,
    lightness_delta: float = 0.0,
    saturation_delta: float = 0.0,
) -> tuple[float, float, float]:
    h, l, s = colorsys.rgb_to_hls(*color)
    l = _clamp01(l + lightness_delta)
    s = _clamp01(s + saturation_delta)
    return colorsys.hls_to_rgb(h, l, s)


def _is_valid_color(value: str) -> bool:
    try:
        to_rgba(value)
        return True
    except Exception:
        return False


def _sample_cmap(name: str, n: int) -> list[str]:
    cmap = mpl.colormaps.get(name)
    if n <= 1:
        return [to_hex(cmap(0.62), keep_alpha=False)]
    xs = np.linspace(0.12, 0.88, n)
    return [to_hex(cmap(float(x)), keep_alpha=False) for x in xs]


def _derived_family(base_color: str, n: int, dark: bool = False) -> list[str]:
    rgb = to_rgba(base_color)[:3]
    out: list[str] = []
    if n <= 1:
        return [to_hex(rgb, keep_alpha=False)]
    for i in range(n):
        t = i / (n - 1)
        if dark:
            mix_t = 0.18 + 0.46 * t
            shifted = _mix_rgb(rgb, (1.0, 1.0, 1.0), mix_t)
            shifted = _adjust_hls(shifted,
                                  lightness_delta=0.03 * t,
                                  saturation_delta=-0.10 * (1.0 - t))
        else:
            mix_t = 0.10 + 0.34 * (1.0 - t)
            shifted = _mix_rgb(rgb, (0.0, 0.0, 0.0), mix_t)
            shifted = _adjust_hls(shifted,
                                  lightness_delta=-0.03 * (1.0 - t),
                                  saturation_delta=0.08 * t)
        out.append(to_hex(shifted, keep_alpha=False))
    return out


class ColorManager:
    """Resolve chain colors and contour level colors from a color spec."""

    def __init__(self, *, dark: bool = False):
        self.dark = dark

    def resolve_chain_colors(self, color: str | list[str], n_chains: int) -> list[str]:
        if n_chains <= 0:
            return []

        if isinstance(color, list):
            if not color:
                return (CORNETTO_PALETTE * n_chains)[:n_chains]
            return (color * n_chains)[:n_chains]

        spec = str(color)

        # Named color: single chain → use exact color;
        # multiple chains → cycle CORNETTO_PALETTE from that color's position.
        if spec in PALETTES:
            base = PALETTES[spec]
            if n_chains == 1:
                return [base]
            try:
                start = CORNETTO_PALETTE.index(base)
            except ValueError:
                return ([base] + CORNETTO_PALETTE)[:n_chains]
            rotated = CORNETTO_PALETTE[start:] + CORNETTO_PALETTE[:start]
            return (rotated * n_chains)[:n_chains]

        # matplotlib colormap name: sample n evenly spaced colors.
        if spec in mpl.colormaps:
            return _sample_cmap(spec, n_chains)

        # Single explicit hex/color: single chain → use as-is;
        # multiple chains → derive a perceptual family from that base color.
        if _is_valid_color(spec):
            if n_chains == 1:
                return [spec]
            return _derived_family(spec, n_chains, dark=self.dark)

        # Fallback: cycle CORNETTO_PALETTE.
        return (CORNETTO_PALETTE * n_chains)[:n_chains]

    def contour_level_colors(
        self,
        base_color: str,
        n_levels: int,
    ) -> tuple[list[tuple[float, float, float, float]],
               list[tuple[float, float, float, float]]]:
        """Return (fill_colors, line_colors) for *n_levels* contour bands.

        Both lists are ordered **innermost-first** (index 0 = innermost,
        index -1 = outermost), because core.py passes them as ``[::-1]``
        to contourf/contour, which expects outermost band at index 0.

        Design principles
        -----------------
        * Inner bands: rich saturation, base lightness, higher opacity.
        * Outer bands: desaturated, washed toward white, near-transparent.
        * Fill alpha follows a quadratic ease-in toward the centre.
        * A gentle hue micro-rotation adds perceptual depth.
        """
        if n_levels <= 0:
            return [], []

        base = to_rgba(base_color)[:3]
        h, l, s = colorsys.rgb_to_hls(*base)

        fill_colors: list[tuple[float, float, float, float]] = []
        line_colors: list[tuple[float, float, float, float]] = []

        for i in range(n_levels):
            # t=1 → innermost band (stored at index 0 so [::-1] puts it last)
            # t=0 → outermost band (stored at index n-1 so [::-1] puts it first)
            t = 1.0 - (i / max(n_levels - 1, 1)) if n_levels > 1 else 1.0

            # Quadratic ease-in: density accumulates visually toward centre.
            t_ease = t ** 2

            # Subtle hue micro-rotation for depth (±~1.4° over full range).
            hue_shift = (0.008 * t - 0.004) * (-1 if self.dark else 1)
            hi = (h + hue_shift) % 1.0

            if self.dark:
                # Dark canvas: inner glows at base brightness; outer fades pale.
                fill_l = _clamp01(l + (1.0 - l) * 0.38 * (1.0 - t))
                fill_s = _clamp01(s * (0.50 + 0.50 * t_ease))
                fill_a = 0.07 + 0.28 * t_ease

                line_l = _clamp01(l + (1.0 - l) * 0.20 * (1.0 - t))
                line_s = _clamp01(s * (0.70 + 0.30 * t))
                line_a = 0.40 + 0.55 * t
            else:
                # Light canvas: inner stays at base hue+dark; outer washes white.
                fill_l = _clamp01(l - 0.05 * t + (1.0 - l) * 0.55 * (1.0 - t))
                fill_s = _clamp01(s * (0.38 + 0.62 * t_ease))
                fill_a = 0.32 + 0.18 * t_ease  # inner ≈ 0.50, outer ≈ 0.32

                line_l = _clamp01(l - l * 0.18 * (1.0 - t))
                line_s = _clamp01(s * (0.65 + 0.35 * t))
                line_a = 0.45 + 0.50 * t

            fill_colors.append((*colorsys.hls_to_rgb(hi, fill_l, fill_s), fill_a))
            line_colors.append((*colorsys.hls_to_rgb(hi, line_l, line_s), line_a))

        return fill_colors, line_colors


def make_contour_colors(
    base_color: str,
    n_levels: int,
    dark: bool = False,
) -> tuple[list[tuple], list[tuple]]:
    return ColorManager(dark=dark).contour_level_colors(base_color, n_levels)
