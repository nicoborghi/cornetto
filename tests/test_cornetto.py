"""
tests/test_cornetto.py
----------------------
Test suite for the cornetto package.

Run with:
    python -m pytest tests/ -v
or directly:
    MPLBACKEND=Agg python tests/test_cornetto.py
"""

import os
import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cornetto import Cornetto, corner
from cornetto.styles import ColorManager

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
from cornetto.stats import hdi, sigmas_to_levels, density_to_levels, compute_stats


# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
N   = 5_000


def _single_data():
    return {
        "mass": RNG.normal(30, 5, N),
        "spin": RNG.normal(0.3, 0.1, N),
        "chi":  RNG.normal(0.1, 0.05, N),
    }


def _multi_data(n_chains=2):
    means = [30 + 20 * i for i in range(n_chains)]
    return {
        "mass": np.vstack([RNG.normal(m, 5, N) for m in means]),
        "spin": np.vstack([RNG.normal(0.3 + 0.2 * i, 0.1, N) for i in range(n_chains)]),
        "chi":  np.vstack([RNG.normal(0.1 * (i + 1), 0.05, N) for i in range(n_chains)]),
    }


# ── Stats ──────────────────────────────────────────────────────────────────────

class TestStats:
    def test_hdi_symmetric(self):
        s = RNG.normal(0, 1, 100_000)
        lo, hi = hdi(s, prob=0.68)
        assert abs(lo + hi) < 0.05          # near-symmetric around 0
        assert abs((hi - lo) - 2.0) < 0.05  # width ≈ 2σ

    def test_hdi_skewed(self):
        s = RNG.exponential(1, 100_000)
        lo, hi = hdi(s, prob=0.68)
        assert lo >= 0

    def test_sigmas_to_levels(self):
        levels = sigmas_to_levels((1, 2, 3))
        assert abs(levels[0] - 0.3935) < 1e-3
        assert abs(levels[1] - 0.8647) < 1e-3
        assert abs(levels[2] - 0.9889) < 1e-3

    def test_density_to_levels_correct(self):
        """density_to_levels must enclose >= requested probability mass."""
        x = np.linspace(-3, 3, 200)
        pdf = np.exp(-0.5 * x**2)
        pdf /= pdf.sum()
        thresholds = density_to_levels(pdf, (0.68,))
        enclosed = float(pdf[pdf >= thresholds[0]].sum())
        assert enclosed >= 0.68 - 1e-3

    def test_compute_stats_keys(self):
        st = compute_stats(RNG.normal(0, 1, N))
        for key in ("mean", "median", "std", "p16", "p84"):
            assert key in st

    def test_compute_stats_empty(self):
        st = compute_stats(np.array([]))
        assert np.isnan(st["mean"])


# ── Cornetto init ──────────────────────────────────────────────────────────────

class TestCornettoInit:
    def test_single_chain(self):
        c = Cornetto(_single_data())
        assert c._n_chains == 1
        assert set(c._param_list) == {"mass", "spin", "chi"}

    def test_multi_chain(self):
        c = Cornetto(_multi_data(2))
        assert c._n_chains == 2

    def test_params_subset(self):
        c = Cornetto(_single_data(), params=["mass", "spin"])
        assert c._param_list == ["mass", "spin"]
        assert "chi" not in c._param_list

    def test_params_missing_raises(self):
        with pytest.raises(ValueError, match="not found in data"):
            Cornetto(_single_data(), params=["mass", "nonexistent"])

    def test_invalid_stat_raises(self):
        with pytest.raises(ValueError, match="Unknown stat"):
            Cornetto(_single_data(), stat="bogus")

    def test_max_chains_enforced(self):
        data = _multi_data(3)
        with pytest.raises(ValueError, match="max_chains"):
            Cornetto(data, max_chains=2)

    def test_subsample(self):
        c = Cornetto(_single_data(), subsample=500)
        for p in c._param_list:
            assert len(c._chains[0][p]) <= 500

    def test_ax_lims(self):
        c = Cornetto(_single_data(), ax_lims={"mass": (20.0, 40.0)})
        assert c._ranges["mass"] == (20.0, 40.0)

    def test_truths_stored(self):
        c = Cornetto(_single_data(), truths={"mass": 30.0})
        assert c._truths["mass"] == 30.0

    def test_repr(self):
        c = Cornetto(_single_data())
        r = repr(c)
        assert "Cornetto" in r
        assert "n_chains=1" in r


# ── Cornetto.plot ──────────────────────────────────────────────────────────────

class TestCornettoPlot:
    def test_returns_fig_axes(self):
        c = Cornetto(_single_data())
        fig, axes = c.plot()
        assert axes.shape == (3, 3)

    def test_dark_theme(self):
        c = Cornetto(_multi_data(2), chain_labels=["A", "B"])
        fig, axes = c.plot(dark=True)
        assert fig is not None

    def test_truths_single(self):
        c = Cornetto(_single_data(), truths={"mass": 30.0, "spin": 0.3})
        fig, _ = c.plot()
        assert fig is not None

    def test_truths_multi_array(self):
        c = Cornetto(
            _multi_data(2),
            truths={"mass": np.array([30.0, 50.0])},
        )
        fig, _ = c.plot()
        assert fig is not None

    def test_kde_cache_reuse(self):
        """Second .plot() with different visual kwargs must skip KDE recomputation."""
        c = Cornetto(_single_data())
        c.plot(n_grid=64)
        key_after_first = c._cache_key
        c.plot(dark=True, n_grid=64)   # same grid, different visual → same key
        assert c._cache_key == key_after_first

    def test_kde_cache_invalidated_on_grid_change(self):
        c = Cornetto(_single_data())
        c.plot(n_grid=64)
        c.plot(n_grid=128)
        assert c._cache_key[0] == 128

    def test_all_builtin_stats(self):
        import functools
        from cornetto import stat_median_hdi
        for s in ("median", "median_mad", "median_hdi", "mean"):
            c = Cornetto(_single_data(), stat=s)
            fig, _ = c.plot()
            assert fig is not None
        hdi90 = functools.partial(stat_median_hdi, prob=0.90)
        c = Cornetto(_single_data(), stat=hdi90)
        fig, _ = c.plot()
        assert fig is not None

    def test_custom_callable_stat(self):
        def my_stat(samples, weights=None):
            s = samples[np.isfinite(samples)]
            m = float(np.median(s))
            return dict(center=m, lo=float(np.quantile(s, 0.1)),
                        hi=float(np.quantile(s, 0.9)), label="80%CI")
        c = Cornetto(_single_data(), stat=my_stat)
        fig, _ = c.plot()
        assert fig is not None

    def test_no_fill_no_smooth(self):
        c = Cornetto(_single_data())
        fig, _ = c.plot(fill_1d=False, fill_contours=False, smooth=False)
        assert fig is not None

    def test_custom_palette_list(self):
        c = Cornetto(_multi_data(2))
        fig, _ = c.plot(color=["#ff0000", "#0000ff"])
        assert fig is not None

    def test_show_tension(self):
        c = Cornetto(_multi_data(2))
        fig, _ = c.plot(show_tension=True)
        assert fig is not None

    def test_params_subset_plot(self):
        c = Cornetto(_single_data(), params=["mass", "spin"])
        fig, axes = c.plot()
        assert axes.shape == (2, 2)

    def test_subsample_then_plot(self):
        c = Cornetto(_single_data(), subsample=300)
        fig, _ = c.plot()
        assert fig is not None

    def test_bandwidth_scale(self):
        c = Cornetto(_single_data())
        fig, _ = c.plot(bandwidth_scale=0.5)
        assert fig is not None

    def test_into_existing_fig(self):
        import matplotlib.pyplot as plt
        c = Cornetto(_single_data())
        fig_pre, _ = plt.subplots(3, 3, figsize=(6, 6))
        fig, axes = c.plot(fig=fig_pre)
        assert fig is fig_pre


# ── Cornetto.pairplot ─────────────────────────────────────────────────────────

class TestCornettoPlairplot:
    def test_basic(self):
        d1 = _single_data()
        d2 = {k: v + 2 for k, v in d1.items()}
        c1 = Cornetto(d1, chain_labels=["A"])
        c2 = Cornetto(d2, chain_labels=["B"])
        fig, axes = c1.pairplot(c2, other_label="B", other_color="coral")
        assert axes.shape == (4, 4)  # (N+1) × (N+1) with blank_diagonal=True

    def test_basic_legacy(self):
        d1 = _single_data()
        d2 = {k: v + 2 for k, v in d1.items()}
        c1 = Cornetto(d1, chain_labels=["A"])
        c2 = Cornetto(d2, chain_labels=["B"])
        fig, axes = c1.pairplot(c2, blank_diagonal=False)
        assert axes.shape == (3, 3)  # N × N with blank_diagonal=False

    def test_upper_triangle_visible(self):
        d1 = _single_data()
        d2 = {k: v + 1 for k, v in d1.items()}
        c1 = Cornetto(d1)
        c2 = Cornetto(d2)
        fig, axes = c1.pairplot(c2)
        # With blank_diagonal=True: (N+1)×(N+1), other's 2D at col > row+1
        grid_sz = 4  # N+1 = 4 for N=3
        other_cells = [axes[r, c].get_visible()
                       for r in range(grid_sz) for c in range(grid_sz)
                       if c > r + 1]
        assert any(other_cells)

    def test_other_missing_param_raises(self):
        d1 = _single_data()
        d2 = {"mass": d1["mass"], "extra": RNG.normal(0, 1, N)}
        c1 = Cornetto(d1)
        c2 = Cornetto(d2)
        with pytest.raises(ValueError, match="not in self"):
            c1.pairplot(c2)


# ── Cornetto.summary ──────────────────────────────────────────────────────────

class TestCornettoSummary:
    def test_returns_summary_table(self):
        from cornetto.stats import SummaryTable
        c = Cornetto(_single_data())
        tbl = c.summary()
        assert isinstance(tbl, SummaryTable)

    def test_summary_params(self):
        c = Cornetto(_single_data())
        tbl = c.summary()
        assert set(tbl.params()) == {"mass", "spin", "chi"}

    def test_summary_multi_chain(self):
        c = Cornetto(_multi_data(2), chain_labels=["A", "B"])
        tbl = c.summary()
        assert tbl.chains() == ["A", "B"]

    def test_summary_str(self):
        c = Cornetto(_single_data())
        s = str(c.summary())
        assert "mass" in s
        assert "median" in s.lower()

    def test_summary_html(self):
        c = Cornetto(_single_data())
        html = c.summary()._repr_html_()
        assert "<table" in html


# ── Cornetto.latex ────────────────────────────────────────────────────────────

class TestCornettoLatex:
    def test_latex_aastex(self):
        pytest.importorskip("astropy")
        c = Cornetto(_single_data())
        src = c.latex(style="aastex")
        assert "mass" in src

    def test_latex_plain(self):
        pytest.importorskip("astropy")
        c = Cornetto(_single_data())
        src = c.latex(style="latex")
        assert r"\begin{table}" in src

    def test_latex_caption_label(self):
        pytest.importorskip("astropy")
        c = Cornetto(_single_data())
        src = c.latex(caption="My table", label="tbl:posterior", style="latex")
        assert "My table" in src
        assert "tbl:posterior" in src

    def test_latex_no_astropy(self, monkeypatch):
        import sys
        for mod in ("astropy", "astropy.table", "astropy.io", "astropy.io.ascii"):
            monkeypatch.setitem(sys.modules, mod, None)
        c = Cornetto(_single_data())
        with pytest.raises((ImportError, ModuleNotFoundError)):
            c.latex()


# ── corner() convenience ──────────────────────────────────────────────────────

class TestCornerFunction:
    def test_basic(self):
        fig, axes = corner(_single_data())
        assert axes.shape == (3, 3)

    def test_with_truths(self):
        fig, axes = corner(_single_data(), truths={"mass": 30.0})
        assert fig is not None

    def test_params_subset(self):
        fig, axes = corner(_single_data(), params=["mass", "chi"])
        assert axes.shape == (2, 2)

    def test_dark(self):
        fig, axes = corner(_single_data(), dark=True)
        assert fig is not None


# ── Color management ──────────────────────────────────────────────────────────

class TestColorManager:
    def test_named_colormap_resolves_n_chains(self):
        mgr = ColorManager(dark=False)
        cols = mgr.resolve_chain_colors("viridis", 4)
        assert len(cols) == 4
        assert len(set(cols)) == 4

    def test_single_base_color_builds_family(self):
        mgr = ColorManager(dark=False)
        cols = mgr.resolve_chain_colors("#1f77b4", 3)
        assert len(cols) == 3
        assert len(set(cols)) == 3

    def test_named_palette_cycles_for_multi_chain(self):
        mgr = ColorManager(dark=False)
        cols = mgr.resolve_chain_colors("indigo", 2)
        assert len(cols) == 2
        assert cols[0] != cols[1], "Two chains with named palette should get distinct colors"

    def test_contour_colors_change_rgb_not_only_alpha(self):
        mgr = ColorManager(dark=False)
        fill, line = mgr.contour_level_colors("#F2545B", 3)
        fill_rgbs = [tuple(c[:3]) for c in fill]
        line_rgbs = [tuple(c[:3]) for c in line]
        assert len(set(fill_rgbs)) > 1
        assert len(set(line_rgbs)) > 1


# ── Entry point for running without pytest ────────────────────────────────────

if __name__ == "__main__":
    import sys
    failed = 0
    suites = [
        TestStats, TestCornettoInit, TestCornettoPlot,
        TestCornettoPlairplot, TestCornettoSummary,
        TestCornerFunction,
    ]
    # Skip latex tests if astropy not installed
    try:
        import astropy  # noqa: F401
        suites.append(TestCornettoLatex)
    except ImportError:
        print("astropy not installed — skipping latex tests")

    for suite_cls in suites:
        suite = suite_cls()
        print(f"\n{suite_cls.__name__}")
        for name in dir(suite_cls):
            if not name.startswith("test_"):
                continue
            method = getattr(suite, name)
            try:
                method()
                print(f"  ✓ {name}")
            except Exception as exc:
                print(f"  ✗ {name}: {exc}")
                failed += 1

    if failed:
        print(f"\n{failed} test(s) FAILED")
        sys.exit(1)
    else:
        print("\nAll tests passed.")
