"""Profile cornetto plot paths. Run: python /tmp/profile_cornetto.py"""
import os, cProfile, pstats, io, time
os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cornetto import corner, quick_corner

RNG = np.random.default_rng(0)

def make_data(n_params: int, n_samples: int, n_chains: int = 1):
    names = [f"p{i}" for i in range(n_params)]
    if n_chains == 1:
        return {n: RNG.normal(i, 1 + 0.1 * i, n_samples) for i, n in enumerate(names)}
    return {
        n: np.vstack([RNG.normal(i + k, 1 + 0.1 * i, n_samples)
                      for k in range(n_chains)])
        for i, n in enumerate(names)
    }

def bench(name, fn, repeats=5, warmup=1):
    for _ in range(warmup):
        fig, _ = fn()
        plt.close(fig)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fig, _ = fn()
        times.append(time.perf_counter() - t0)
        plt.close(fig)
    times.sort()
    med = times[len(times) // 2]
    print(f"  {name:35s}  median {med*1000:7.1f} ms   "
          f"min {min(times)*1000:7.1f}   max {max(times)*1000:7.1f}   "
          f"(n={repeats})")
    return med

def cprofile(name, fn, repeats=3):
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeats):
        fig, _ = fn()
        plt.close(fig)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(25)
    print(f"\n=== cProfile: {name} (top 25 by cumulative, n={repeats}) ===")
    print(s.getvalue())

print("\n── Wall-clock (Agg backend) ──")

# Scenario A: README's "fast-path" claim shape
d_fast = make_data(8, 50_000)
print("\n[A] 8 params × 50_000 samples, 1 chain")
bench("corner(smooth=True)",  lambda: corner(d_fast))
bench("corner(smooth=False)", lambda: corner(d_fast, smooth=False))
bench("quick_corner()",       lambda: quick_corner(d_fast))

# Scenario B: typical corner use
d_typ = make_data(5, 10_000)
print("\n[B] 5 params × 10_000 samples, 1 chain")
bench("corner(smooth=True)",  lambda: corner(d_typ))
bench("quick_corner()",       lambda: quick_corner(d_typ))

# Scenario C: multi-chain
d_mc = make_data(5, 10_000, n_chains=2)
print("\n[C] 5 params × 10_000 samples, 2 chains")
bench("corner(smooth=True)",  lambda: corner(d_mc))
bench("quick_corner()",       lambda: quick_corner(d_mc))

# Scenario D: larger smooth corner (KDE-dominated)
d_big = make_data(8, 20_000)
print("\n[D] 8 params × 20_000 samples, 1 chain")
bench("corner(smooth=True)",  lambda: corner(d_big))

print("\n── cProfile drilldown ──")
cprofile("corner(smooth=True) 8x50k", lambda: corner(d_fast), repeats=3)
cprofile("quick_corner 8x50k", lambda: quick_corner(d_fast), repeats=3)
