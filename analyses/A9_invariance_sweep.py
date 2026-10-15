#!/usr/bin/env python3
"""Analysis A9. Product ordering across a bounded filter-parameter sweep.

Eight configurations vary persistence, inflation, model error, or ensemble size.
Every configuration and product uses 50 seeds. A7 supplies the unchanged run
and summary functions. Post-analysis RMSE is the root of the seed-and-day mean
square against each product's corrected input on its own observation days.

The reported ordering applies only to this grid. It does not establish
invariance across untested error specifications or independent retrieval accuracy.
The script compares selected rows with historical diagnostic values and checks
matching rows against the A7 grid. Historical table labels remain data keys.

Author: Sebastian R.O. Marshall."""
import sys, os, time, pathlib, multiprocessing as mp, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from A_temporal_subsampling import SEEDS, RES           # noqa: E402
from A7_error_specification import OUT, sweep           # noqa: E402

CONFIGS = [("published", "none", np.nan),
           ("phi 0.00", "persistence_phi", 0.00),
           ("phi 0.30", "persistence_phi", 0.30),
           ("phi 0.70", "persistence_phi", 0.70),
           ("phi 0.90", "persistence_phi", 0.90),
           ("inflation 1.20", "inflation_factor", 1.20),
           ("model error 0.02", "model_error_std", 0.02),
           ("ensemble size 50", "ensemble_size", 50)]
# Printed by the reference brief (21_adversarial_check.md lines 422 and 428); used for the comparison only.
BRIEF = {"published": ({"9km": 0.0195, "1km": 0.0230, "400m": 0.0248}, {"9km": 0.410, "1km": 0.396, "400m": 0.394}),
         "model error 0.02": ({"9km": 0.0332, "1km": 0.0400, "400m": 0.0433}, {"9km": 0.700, "1km": 0.689, "400m": 0.687})}


def main():
    t0 = time.perf_counter(); n_proc = os.cpu_count()
    print(f"A9 bounded parameter sweep | {n_proc} processes | {len(CONFIGS)} configurations x {len(RES)} products x {len(SEEDS)} seeds (42 to 91)")
    plan = [(lab, par, val, r) for lab, par, val in CONFIGS for r in RES]
    with mp.Pool(n_proc) as pool:
        out = sweep(pool, [(r, {} if par == "none" else {par: val}) for _, par, val, r in plan])
    df = pd.DataFrame([dict(config=lab, parameter_changed=par, value=val, product=r, n_seeds=x["n_seeds"], n_days=x["n_days"],
                            post_rmse=x["post_rmse"], post_rmse_sd_across_seeds=x["post_rmse_sd_across_seeds"],
                            reduction_pct=x["reduction_pct"], post_to_open_ratio=x["post_to_open_ratio"],
                            NIV_mean_over_seeds=x["NIV_mean_over_seeds"], mean_K_update_days=x["mean_K_update_days"],
                            open_loop_rmse=x["open_loop_rmse"], post_rmse_mean_of_per_seed=x["post_rmse_mean_of_per_seed"],
                            NIV_seed42=x["NIV_seed42"], NIV_forecast_variance=x["NIV_forecast_variance"])
                       for (lab, par, val, r), x in zip(plan, out)])
    df.to_csv(OUT / "A9_invariance_sweep.csv", index=False)

    h = lambda name, w: f"{name + ' 9km':>{w + 6}s}{'1km':>{w}s}{'400m':>{w}s}"
    print(f"\n{'config':<18s}{h('post', 8)}{h('sd', 9)}{h('NIV', 6)}{h('K', 6)}{h('red %', 6)}{h('post/open', 8)}   conditional ordering (own-target RMSE)")
    for lab, _, _ in CONFIGS:
        t = df[df.config == lab].set_index("product")
        cells = lambda col, w, fmt, pad: f"{'':>{pad}s}" + "".join(f"{t.loc[r, col]:>{w}{fmt}}" for r in RES)
        print(f"{lab:<18s}{cells('post_rmse', 8, '.4f', 6)}{cells('post_rmse_sd_across_seeds', 9, '.1e', 6)}"
              f"{cells('NIV_mean_over_seeds', 6, '.3f', 6)}{cells('mean_K_update_days', 6, '.3f', 6)}"
              f"{cells('reduction_pct', 6, '.1f', 6)}{cells('post_to_open_ratio', 8, '.3f', 6)}   "
              + " < ".join(sorted(RES, key=lambda r: t.loc[r, "post_rmse"])))
    print("open-loop RMSE on own days: " + "  ".join(f"{r} {df[df['product'] == r].open_loop_rmse.iloc[0]:.5f}" for r in RES))
    print("wrote", OUT / "A9_invariance_sweep.csv")

    print("\nComparison with the reference brief at the fourth decimal (the brief scores the mean of per-seed RMSEs)")
    agree = {"root-mean-square": [], "mean of per-seed": []}
    for lab, (bp, br) in BRIEF.items():
        t = df[df.config == lab].set_index("product")
        for r in RES:
            rms, mps = t.loc[r, "post_rmse"], t.loc[r, "post_rmse_mean_of_per_seed"]
            agree["root-mean-square"].append(f"{rms:.4f}" == f"{bp[r]:.4f}")
            agree["mean of per-seed"].append(f"{mps:.4f}" == f"{bp[r]:.4f}")
            print(f"  {lab:<17s}{r:>5s}  brief {bp[r]:.4f}  here: root-mean-square {rms:.4f} ({'agrees' if agree['root-mean-square'][-1] else 'differs'}), "
                  f"mean of per-seed {mps:.4f} ({'agrees' if agree['mean of per-seed'][-1] else 'differs'})  |  ratio brief {br[r]:.3f}, "
                  f"here {t.loc[r, 'post_to_open_ratio']:.3f} (rms) / {mps / t.loc[r, 'open_loop_rmse']:.3f} (mean of per-seed)")
    for k, v in agree.items():
        print(f"  {k}: {sum(v)} of {len(v)} values agree to the fourth decimal -> {'AGREE' if all(v) else 'DO NOT ALL AGREE'}")

    # self-check
    pub = df[df.config == "published"].set_index("product")
    assert len(df) == len(CONFIGS) * len(RES) and (df.n_seeds == 50).all()
    assert abs(pub.loc["9km", "post_rmse"] - 0.0195) < 2e-4, pub.loc["9km", "post_rmse"]
    assert abs(pub.loc["9km", "NIV_seed42"] - 0.484) < 0.01, pub.loc["9km", "NIV_seed42"]
    for lab, _, _ in CONFIGS[1:]:   # negative control: every changed setting reached the filter
        t = df[df.config == lab].set_index("product")
        assert all(t.loc[r, "post_rmse"] != pub.loc[r, "post_rmse"] for r in RES), lab
    a7 = OUT / "A7_error_specification.csv"
    if a7.exists():
        g = pd.read_csv(a7); g = g[g.part == "b_grid_50_seeds"]
        for lab, sig in (("published", 0.040), ("model error 0.02", 0.020)):
            t = df[df.config == lab].set_index("product")
            for r in RES:
                ref = g[(g["product"] == r) & np.isclose(g.sigma_m, sig)]["post_rmse"].item()
                assert abs(t.loc[r, "post_rmse"] - ref) < 1e-12, (lab, r)
        print("cross-check: published and model-error-0.02 rows identical to the A7 grid rows at sigma_m 0.040 and 0.020")
    else:
        print("cross-check against A7 SKIPPED: A7_error_specification.csv not found")
    print(f"self-check passed | runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
