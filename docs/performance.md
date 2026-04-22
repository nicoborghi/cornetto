# Performance

## Benchmarks

All numbers: Python 3.12, matplotlib Agg backend, median of 5 warm runs.

### 8-parameter, 50 000-sample chain

| Path                  | Wall time | vs. `corner` |
| ---                   | ---       | ---          |
| `corner()`            | 3.20 s    | 1.0×         |
| `quick_corner()`      | 0.46 s    | **7.0× faster** |

### Scaling with dimension (50 000 samples)

| `N_params` | `corner` | `quick_corner` |
| ---        | ---      | ---            |
| 4          | ~0.9 s   | 0.12 s         |
| 8          | ~3.2 s   | 0.46 s         |
| 12         | ~8.5 s   | 1.4 s          |

At low dimensions the bottleneck is **matplotlib axis setup**; at higher
dimensions it shifts to **2-D KDE evaluations**. `quick_corner` addresses both.

## Where the time goes

For the default `corner()` path on an 8-param × 50 k-sample chain, cProfile
reports:

- ~70 % in KDExpress FFT-KDE evaluations + bandwidth estimation.
- ~15 % in matplotlib contour / line construction.
- ~10 % in matplotlib axis / tick / transform setup.
- ~5  % in cornetto glue (parsing, stat evaluation, titles).

`quick_corner` eliminates the first bucket entirely, shrinks the second with
lower `bins` and fewer contour levels, and shrinks the third by only creating
visible axes and sharing x per column.

## Speed tips

### For `corner()`

- **Thin the chain.** `subsample=10_000` is indistinguishable from the full
  posterior in most cases and roughly halves the KDE cost.
- **Use `params=[...]` early.** Cornetto only KDEs what it draws.
- **Smaller grid.** `n_grid=64` is about 2.5× faster than the default
  `n_grid=128` and barely affects the contour shapes.
- **Reuse the object.** `Cornetto.plot()` caches KDEs and statistics - call
  `.plot()` multiple times with different styling kwargs for free.

### For `quick_corner()`

- **Fewer bins.** `bins=20` produces a blocky but fast preview.
- **One contour level.** `sigmas=(2,)` skips the 1σ band.
- **Smaller figure.** `fig_size_per_dim=1.6` trims matplotlib setup time.

## Reproducing the numbers

Run this notebook to reproduce the benchmark on your machine.

```python
# %%
import time
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from cornetto import corner, quick_corner

rng = np.random.default_rng(0)
data = {f"p{i}": rng.normal(i, 1 + 0.1 * i, 50_000) for i in range(8)}

# %%  warm-ups (first call JIT-compiles KDExpress)
corner({"a": data["p0"][:1000], "b": data["p1"][:1000]})
plt.close("all")
quick_corner(data)
plt.close("all")

# %%
for fn in (corner, quick_corner):
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        fig, _ = fn(data)
        times.append(time.perf_counter() - t0)
        plt.close(fig)
    print(f"{fn.__name__:13s}  min={min(times):.3f}s  median={sorted(times)[2]:.3f}s")
```
