# cornetto

[![Tests](https://img.shields.io/github/actions/workflow/status/nicoborghi/cornetto/tests.yml?label=Tests&logo=github&logoColor=white)](https://github.com/nicoborghi/cornetto/actions)
[![Coverage](https://img.shields.io/codecov/c/github/nicoborghi/cornetto)](https://codecov.io/gh/nicoborghi/cornetto)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/nicoborghi/cornetto/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cornetto)](https://pypi.org/project/cornetto/)
[![Built with Claude](https://img.shields.io/badge/built_with-Claude-orange?logo=anthropic&logoColor=white)](https://claude.ai)


**Fast, smooth corner plots for MCMC chains.**

!!! note "Early release"
    cornetto is built for research use and works well for everyday analysis,
    but the API may still evolve and some features are still maturing.
    Feedback and contributions are very welcome — open an
    [issue](https://github.com/nicoborghi/cornetto/issues) or a PR.

Cornetto is a Python library for visualising Bayesian posterior samples —
the kind you get out of samplers like
[bilby](https://lscsoft.docs.ligo.org/bilby/),
[emcee](https://emcee.readthedocs.io/), or
[dynesty](https://dynesty.readthedocs.io/).
Feed it a plain `dict` of sample arrays and it returns a `(fig, axes)` pair
you can save or embed in a notebook.

The speed comes from [KDExpress](https://github.com/mtagliazucchi/KDExpress),
a JAX-based FFT-KDE library that computes smooth 1-D and 2-D densities in
O(N log N) time. At 20k samples a full corner plot renders in well under a
second on CPU; `quick_corner` (histograms only) is faster still.

<p align="center">
  <img src="https://raw.githubusercontent.com/nicoborghi/cornetto/refs/heads/main/docs/assets/example_cornetto.svg" alt="cornetto corner plot" width="500">
</p>

## Install

```bash
pip install cornetto
```

## Quick example

```python
import numpy as np
from cornetto import corner

data = {
    "mass_1": chain["mass_1"],    # 1-D array of posterior samples
    "mass_2": chain["mass_2"],
    "chi_eff": chain["chi_eff"],
}

fig, axes = corner(
    data,
    labels={"mass_1": r"$m_1\,[M_\odot]$",
            "mass_2": r"$m_2\,[M_\odot]$",
            "chi_eff": r"$\chi_{\mathrm{eff}}$"},
    truths={"mass_1": 35.6},
    chain_labels=["GW200129"],
)
fig.savefig("posterior.pdf", bbox_inches="tight")
```

For fast iteration during analysis, skip the KDE entirely:

```python
from cornetto import quick_corner
fig, axes = quick_corner(data)   # histograms only, sub-second
```

## What you get

| Feature | Detail |
|---|---|
| **FFT-KDE contours** | JAX-accelerated, O(N log N). Smooth at 10k–100k samples. |
| **`quick_corner`** | Histogram-only path, ~7× faster. Same dict API. |
| **Multi-chain** | Pass `(N_chains, N_samples)` arrays. Palette and legend are automatic. |
| **Sparse chains** | Parameters can be absent for some chains - pass a shorter array or `None`. |
| **Summary tables** | `c.summary()` renders HTML in Jupyter; `c.latex()` produces AASTeX output. |
| **Reproducible style** | Plots look identical regardless of prior `rcParams` changes in the notebook. |

## Where to go next

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart](quickstart.md)**
  First plot in under a minute.
- :material-book-open-variant: **[User guide](guide.md)**
  Multi-chain, sparse chains, weights, statistics, dark theme.
- :material-flash: **[Fast mode](quick-corner.md)**
  `quick_corner` - the sub-second path.
- :material-code-tags: **[API reference](api.md)**
  Every function, every argument.

</div>
