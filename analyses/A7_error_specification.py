#!/usr/bin/env python3
"""Analysis A7. Sensitivity to scalar model-error standard deviation.

For each product, bisection locates the seed-42 model-error setting where the
declared normalized innovation variance (NIV) equals one. The denominator uses
uninflated forecast variance and the stated observation-error variance. This
single moment condition does not establish full filter consistency.

The common parameter grid and each product's NIV-one setting use 50 seeds.
Post-analysis RMSE is the root of the seed-and-day mean square on each product's
update days against its own corrected input. The script also reports mean gain,
NIV, error reduction, and the post-to-baseline ratio. Product ordering is
conditional on the error specification, target series, and scored calendar.

Author: Sebastian R.O. Marshall."""
import sys, os, time, pathlib, multiprocessing as mp, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from A_temporal_subsampling import load, TOOL, SEEDS, RES  # noqa: E402
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                      # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "analyses"
R = OPTIMAL_CONFIG.obs_error_std ** 2
GRID = [0.015, 0.020, 0.025, 0.030, 0.040]
NIV_PF = "uninflated forecast ensemble variance (ddof=0) as in A6"
OPEN_REF = {"9km": 0.0475, "1km": 0.0580, "400m": 0.0630}   # own-day open-loop RMSE reported in the paper
_DATA = {}


def run_stats(product, overrides, seed):
    """One filter run: own-day squared-error sum, mean gain and NIV (inputs loaded once per process)."""
    if not _DATA:
        d, _ = load()
        _DATA.update({r: (d[r]["baseline"].to_numpy(float), d[r]["observation_corrected"].to_numpy(float),
                          d[r]["obs_available"].to_numpy(bool)) for r in RES})
    ol, y, m = _DATA[product]
    res = run_enkf(ol, y, m, config=EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, **overrides, "seed": seed}))
    ok = np.isfinite(res.innovation)
    return dict(sse=float(((res.analysis[m] - y[m]) ** 2).sum()), n_days=int(m.sum()), n_updates=int(ok.sum()),
                K=float(res.kalman_gain[ok].mean()),
                niv=float(res.innovation[ok].var() / ((res.forecast_spread[ok] ** 2).mean() + R)),
                open_rmse=float(np.sqrt(((ol[m] - y[m]) ** 2).mean())))


def _task(a):
    return run_stats(*a)


def summarise(runs):
    """Collapse per-seed runs (first run = seed 42) into one row."""
    n = runs[0]["n_days"]
    assert all(u["n_updates"] == n for u in runs), "update days differ from obs_available days"
    sse = np.array([u["sse"] for u in runs]); per_seed = np.sqrt(sse / n)
    post = float(np.sqrt(sse.sum() / (n * len(runs)))); ol = runs[0]["open_rmse"]
    return dict(n_seeds=len(runs), n_days=n, mean_K_update_days=float(np.mean([u["K"] for u in runs])),
                NIV_mean_over_seeds=float(np.mean([u["niv"] for u in runs])), NIV_seed42=runs[0]["niv"],
                NIV_forecast_variance=NIV_PF, post_rmse=post,
                post_rmse_sd_across_seeds=float(per_seed.std(ddof=1)) if len(runs) > 1 else np.nan,
                post_rmse_mean_of_per_seed=float(per_seed.mean()), open_loop_rmse=ol,
                reduction_pct=100.0 * (1.0 - post / ol), post_to_open_ratio=post / ol)


def sweep(pool, rows, seeds=SEEDS):
    """Run every (product, overrides) row over the seeds in one pool pass; one summary per row."""
    assert seeds[0] == 42
    flat = pool.map(_task, [(p, ov, s) for p, ov in rows for s in seeds], chunksize=1)
    k = len(seeds)
    return [summarise(flat[i * k:(i + 1) * k]) for i in range(len(rows))]


def niv_at(product, sigma):
    return run_stats(product, {"model_error_std": sigma}, 42)["niv"]


def bisect_niv1(product, lo=0.001, hi=0.040, tol=1e-6):
    """sigma_m with NIV = 1 at seed 42. NIV falls as sigma_m rises, so NIV(lo) > 1 > NIV(hi) brackets the root."""
    flo, fhi = niv_at(product, lo), niv_at(product, hi)
    assert flo > 1.0 > fhi, f"{product}: NIV({lo}) = {flo:.3f}, NIV({hi}) = {fhi:.3f} does not bracket 1"
    it = 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); it += 1
        lo, hi = (mid, hi) if niv_at(product, mid) > 1.0 else (lo, mid)
    root = 0.5 * (lo + hi); s4 = round(root, 4)
    return dict(product=product, root=root, sigma4=s4, iterations=it, bracket=(flo, fhi),
                niv_below=niv_at(product, s4 - 1e-4), niv_above=niv_at(product, s4 + 1e-4))


def main():
    t0 = time.perf_counter(); n_proc = os.cpu_count()
    assert len(SEEDS) == 50 and SEEDS[0] == 42
    print(f"A7 error specification | {n_proc} processes | seeds {SEEDS[0]} to {SEEDS[-1]} | R = {R:.4f} | NIV Pf: {NIV_PF}")
    with mp.Pool(n_proc) as pool:
        bis = {b["product"]: b for b in pool.map(bisect_niv1, RES)}
        rows_a = sweep(pool, [(r, {"model_error_std": bis[r]["sigma4"]}) for r in RES], seeds=[42])
        plan = [(r, s, bool(np.isclose(s, bis[r]["sigma4"]))) for r in RES for s in sorted(set(GRID) | {bis[r]["sigma4"]})]
        rows_b = sweep(pool, [(r, {"model_error_std": s}) for r, s, _ in plan])

    print("\n(a) Declared NIV = 1 by bisection on sigma_m; this is a moment diagnostic (seed 42, all else published; bracket 0.001 to 0.040, tolerance 1e-6)")
    for r, x in zip(RES, rows_a):
        b = bis[r]
        print(f"  {r:>4s}: sigma_m = {b['sigma4']:.4f}  NIV achieved {x['NIV_seed42']:.4f}  (root {b['root']:.6f} after {b['iterations']} halvings; "
              f"NIV at the bracket ends {b['bracket'][0]:.3f} / {b['bracket'][1]:.3f}; at {b['sigma4'] - 1e-4:.4f}: {b['niv_below']:.4f}, "
              f"at {b['sigma4'] + 1e-4:.4f}: {b['niv_above']:.4f})")

    recs = [dict(part="a_bisection_seed42", product=r, sigma_m=bis[r]["sigma4"], sigma_m_root_unrounded=bis[r]["root"],
                 is_niv1_setting=True, **x) for r, x in zip(RES, rows_a)]
    recs += [dict(part="b_grid_50_seeds", product=r, sigma_m=s, sigma_m_root_unrounded=np.nan, is_niv1_setting=f, **x)
             for (r, s, f), x in zip(plan, rows_b)]
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "A7_error_specification.csv", index=False)

    b = df[df.part == "b_grid_50_seeds"]
    print("\n(b) sigma_m grid over 50 seeds; post RMSE = root of the seed-and-day mean squared error on own observation days")
    print(f"  {'product':>7s} {'sigma_m':>8s} {'NIV=1':>6s} {'n_days':>6s} {'mean K':>7s} {'NIV mean':>8s} {'NIV s42':>8s} "
          f"{'post':>8s} {'sd':>9s} {'open':>7s} {'red %':>6s} {'post/open':>9s}")
    for _, x in b.iterrows():
        print(f"  {x['product']:>7s} {x.sigma_m:>8.4f} {'yes' if x.is_niv1_setting else '':>6s} {x.n_days:>6d} {x.mean_K_update_days:>7.4f} "
              f"{x.NIV_mean_over_seeds:>8.4f} {x.NIV_seed42:>8.4f} {x.post_rmse:>8.5f} {x.post_rmse_sd_across_seeds:>9.2e} "
              f"{x.open_loop_rmse:>7.5f} {x.reduction_pct:>6.1f} {x.post_to_open_ratio:>9.3f}")

    pub = b[np.isclose(b.sigma_m, 0.040)].set_index("product")
    own = b[b.is_niv1_setting].set_index("product")
    order = lambda t: " < ".join(f"{r} {t.loc[r, 'post_rmse']:.4f}" for r in sorted(RES, key=lambda r: t.loc[r, "post_rmse"]))
    print("\n(c) conditional ordering of own-target post RMSE among the products")
    print("  common published sigma_m 0.040:", order(pub))
    print("  own NIV = 1 sigma_m (" + ", ".join(f"{r} {own.loc[r, 'sigma_m']:.4f}" for r in RES) + "):", order(own))
    print("wrote", OUT / "A7_error_specification.csv")

    # self-check
    a6 = pd.read_csv(OUT / "A6_filter_consistency.csv").set_index("product")
    a = df[df.part == "a_bisection_seed42"].set_index("product")
    assert len(b) >= 15 and (b.n_seeds == 50).all()
    assert abs(pub.loc["9km", "post_rmse"] - 0.0195) < 2e-4, pub.loc["9km", "post_rmse"]
    assert abs(pub.loc["9km", "NIV_seed42"] - 0.484) < 0.01, pub.loc["9km", "NIV_seed42"]
    for r in RES:
        assert abs(pub.loc[r, "NIV_seed42"] - a6.loc[r, "normalised_innovation_variance"]) < 1e-9, r   # A6 reproduced
        assert pub.loc[r, "n_days"] == a6.loc[r, "n_updates"], r
        assert abs(pub.loc[r, "open_loop_rmse"] - OPEN_REF[r]) <= 1e-4, (r, pub.loc[r, "open_loop_rmse"])
        assert abs(a.loc[r, "NIV_seed42"] - 1.0) < 0.01, (r, a.loc[r, "NIV_seed42"])
        assert bis[r]["niv_below"] > 1.0 > bis[r]["niv_above"], r      # the 4-decimal value sits on the crossing
        assert own.loc[r, "NIV_seed42"] == a.loc[r, "NIV_seed42"], r    # grid NIV = 1 row reruns the bisection value
        assert np.isclose(pub.loc[r, "reduction_pct"], 100.0 * (1.0 - pub.loc[r, "post_to_open_ratio"])), r
    print(f"self-check passed | runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
