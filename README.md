# cornetto

[![Tests](https://img.shields.io/github/actions/workflow/status/nicoborghi/cornetto/tests.yml?label=Tests&logo=github&logoColor=white)](https://github.com/nicoborghi/cornetto/actions)
[![Coverage](https://img.shields.io/codecov/c/github/nicoborghi/cornetto)](https://codecov.io/gh/nicoborghi/cornetto)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/nicoborghi/cornetto/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cornetto)](https://pypi.org/project/cornetto/)
[![Built with Claude](https://img.shields.io/badge/built_with-Claude-orange?logo=anthropic&logoColor=white)](https://claude.ai)

**Fast, smooth corner plots for MCMC chains.**

> [!NOTE]
> **Early release** - cornetto is built for research use and works well for everyday analysis, but the API may still evolve and some features are still maturing. Feedback and contributions are very welcome.

Cornetto takes a `dict[str, array]` of posterior samples and produces
publication-ready corner plots. Under the hood it uses
[KDExpress](https://github.com/mtagliazucchi/KDExpress) - a JAX-based
FFT-KDE library - so contours are smooth and rendering stays fast even at
50k+ samples.

<p align="center">
  <img src="https://raw.githubusercontent.com/nicoborghi/cornetto/refs/heads/main/docs/assets/example_cornetto.svg" alt="cornetto corner plot" width="500">
</p>

## Install

```bash
pip install cornetto
```

Requires Python ≥ 3.10, NumPy, Matplotlib, SciPy, and
[KDExpress](https://pypi.org/project/KDExpress/).

## Usage

```python
import numpy as np
from cornetto import corner

# data is a plain dict: parameter name → 1-D sample array
data = {
    "mass_1": chain["mass_1"],   # shape (N_samples,)
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

Multiple chains (e.g. two events, or prior vs posterior) are just 2-D arrays:

```python
data = {
    "mass_1": np.stack([chain_A["mass_1"], chain_B["mass_1"]]),  # (2, N)
    "mass_2": np.stack([chain_A["mass_2"], chain_B["mass_2"]]),
}
corner(data, chain_labels=["GW150914", "GW190521"])
```

For fast iteration during analysis, `quick_corner` skips KDE entirely:

```python
from cornetto import quick_corner
fig, axes = quick_corner(data)   # histograms only, sub-second
```

## Documentation

Full guide, API reference, and benchmarks at
**[cornetto.readthedocs.io](https://cornetto.readthedocs.io)**.

## License

MIT - see [LICENSE](LICENSE).
