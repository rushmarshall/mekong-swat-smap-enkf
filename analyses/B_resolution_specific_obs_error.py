#!/usr/bin/env python3
"""Analysis B. Product-specific residual diagnostics and observation-error sensitivity.

The residual-product calculation averages analysis residuals times forecast
innovations on update days. An iterated version supplies a product-specific
observation-error setting. This diagnostic depends on filter and error-covariance
assumptions; it does not independently estimate retrieval truth or prove that
downscaling necessarily increases error.

Each product is evaluated under that setting and a common observation-error grid.
Every final arm uses 50 seeds. Scores compare analyses with their own assimilated
corrected inputs on their own update days. The ordering is conditional on error
settings, targets, and calendars. It cannot establish independent product accuracy.

Author: Sebastian R.O. Marshall."""
import sys
import pathlib
import numpy as np
import pandas as pd

TOOL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                      # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "analyses"
OUT.mkdir(parents=True, exist_ok=True)
RES = ["9km", "1km", "400m"]
SEEDS = list(range(42, 92))
GRID = [0.02, 0.04, 0.06, 0.08, 0.12]


def cfg_with(**kw):
    return EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, **kw})


def load():
    out = {}
    for r in RES:
        f = pd.read_csv(TOOL / "data/enkf" / f"inputs_{r}.csv")
        out[r] = dict(ol=f["baseline"].to_numpy(float),
                      obs=f["observation_corrected"].to_numpy(float),
                      mask=f["obs_available"].to_numpy(bool))
    return out


def score(dat, sigma, seeds=SEEDS):
    """Mean RMSE across seeds, scored on the arm's own assimilated days."""
    rmse = []
    for s in seeds:
        r = run_enkf(dat["ol"], dat["obs"], dat["mask"],
                     config=cfg_with(obs_error_std=sigma, seed=s))
        m = dat["mask"]
        rmse.append(float(np.sqrt(np.mean((r.analysis[m] - dat["obs"][m]) ** 2))))
    return float(np.mean(rmse)), float(np.std(rmse, ddof=1))


def desroziers(dat, sigma, seeds=SEEDS):
    """E[(y - Hx_a)(y - Hx_b)] over update days, averaged across seeds."""
    vals = []
    for s in seeds:
        r = run_enkf(dat["ol"], dat["obs"], dat["mask"],
                     config=cfg_with(obs_error_std=sigma, seed=s))
        upd = np.isfinite(r.innovation)
        d_b = r.innovation[upd]                       # y - Hx_b, the prior innovation
        d_a = dat["obs"][upd] - r.analysis[upd]       # y - Hx_a, the analysis residual
        vals.append(float(np.mean(d_a * d_b)))
    v = float(np.mean(vals))
    return (np.sqrt(v) if v > 0 else np.nan), v


def estimate(dat, start=0.04, iters=8, tol=1e-4):
    """Iterate the residual-product diagnostic to a fixed point under the stated update rule."""
    s = start
    path = [s]
    for _ in range(iters):
        est, _ = desroziers(dat, s, seeds=SEEDS[:10])   # 10 seeds inside the loop, 50 at the end
        if not np.isfinite(est):
            break
        if abs(est - s) < tol:
            s = est
            path.append(s)
            break
        s = 0.5 * (s + est)                             # damped, the map can oscillate
        path.append(s)
    return s, path


def main():
    d = load()
    print("=" * 78)
    print("ANALYSIS B. Resolution-specific observation error")
    print("=" * 78)

    print("\nStep 1. sigma_o diagnosed from each product's filter residuals")
    print("(Iterated residual-product diagnostic; its assumptions are not independently verified).")
    print(f"  {'product':<10s}{'sigma_o assumed':>17s}{'sigma_o estimated':>19s}{'ratio':>8s}")
    est = {}
    rows = []
    for r in RES:
        s, path = estimate(d[r])
        est[r] = s
        print(f"  {r:<10s}{0.04:>17.4f}{s:>19.4f}{s/0.04:>8.2f}")
        rows.append(dict(product=r, sigma_assumed=0.04, sigma_estimated=s,
                         ratio=s / 0.04, iterations=len(path) - 1))
    pd.DataFrame(rows).to_csv(OUT / "B_sigma_estimates.csv", index=False)

    order = sorted(RES, key=lambda r: est[r])
    print(f"\n  ordering by residual-diagnostic sigma_o, smallest first: {' < '.join(order)}")
    print("  (these values do not independently identify retrieval-error variance)")

    print("\nStep 2. Each product re-run under ITS OWN estimated sigma_o.")
    print(f"  {'product':<10s}{'sigma_o':>9s}{'RMSE':>9s}{'sd':>8s}")
    srows = []
    own = {}
    for r in RES:
        m, sd = score(d[r], est[r])
        own[r] = m
        print(f"  {r:<10s}{est[r]:>9.4f}{m:>9.4f}{sd:>8.5f}")
        srows.append(dict(product=r, sigma=est[r], setting="own estimate",
                          rmse_mean=m, rmse_sd=sd))
    best = min(own, key=own.get)
    print(f"  lowest own-target RMSE under residual-diagnostic sigma_o: {best}")

    print("\nStep 3. Common grid, every product at every sigma_o.")
    hdr = "".join(f"{s:>10.2f}" for s in GRID)
    print(f"  {'product':<10s}{hdr}")
    table = {}
    for r in RES:
        line = []
        for s in GRID:
            m, sd = score(d[r], s)
            line.append(m)
            srows.append(dict(product=r, sigma=s, setting="grid",
                              rmse_mean=m, rmse_sd=sd))
        table[r] = line
        print(f"  {r:<10s}" + "".join(f"{v:>10.4f}" for v in line))
    pd.DataFrame(srows).to_csv(OUT / "B_obs_error_sensitivity.csv", index=False)

    print("\n" + "-" * 78)
    print("Conditional ordering of own-target RMSE across the tested common sigma_o grid")
    allbest = True
    for i, s in enumerate(GRID):
        col = {r: table[r][i] for r in RES}
        b = min(col, key=col.get)
        allbest &= (b == "9km")
        print(f"  sigma_o = {s:.2f}: lowest own-target RMSE = {b:<6s} "
              f"(9km {col['9km']:.4f}, 1km {col['1km']:.4f}, 400m {col['400m']:.4f})")
    print(f"\n  9 km has lowest own-target RMSE at every tested grid point: {allbest}")
    print(f"  9 km has lowest own-target RMSE under product-specific diagnostic settings: {best == '9km'}")
    print(f"\nwrote {OUT/'B_sigma_estimates.csv'} and {OUT/'B_obs_error_sensitivity.csv'}")


if __name__ == "__main__":
    main()
