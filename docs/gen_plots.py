"""
MkDocs hook - regenerates static guide illustrations on every build.
Runs before any page is built so images are always current.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def on_pre_build(config) -> None:
    docs_dir = Path(config["docs_dir"])
    _gen_multi_dark(docs_dir / "multi_dark.svg")


def _gen_multi_dark(out: Path) -> None:
    from cornetto import corner

    rng = np.random.default_rng(0)
    N = 10_000
    data = {
        "mass_1": np.stack([rng.normal(30, 4, N), rng.normal(85, 8, N)]),
        "mass_2": np.stack([rng.normal(25, 3, N), rng.normal(66, 6, N)]),
        "chi":    np.stack([rng.uniform(-1, 1, N), rng.uniform(-1, 1, N)]),
    }
    labels = {
        "mass_1": r"$m_1\,[M_\odot]$",
        "mass_2": r"$m_2\,[M_\odot]$",
        "chi":    r"$\chi_{\mathrm{eff}}$",
    }
    fig, _ = corner(
        data,
        labels=labels,
        truths={"mass_1": np.array([30.0, 85.0]), "mass_2": np.array([25.0, 66.0])},
        chain_labels=["Chain 1", "Chain 2"],
        dark=True,
    )
    fig.savefig(out, bbox_inches="tight", format="svg")
    plt.close(fig)
