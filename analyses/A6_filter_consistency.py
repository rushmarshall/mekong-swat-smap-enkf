#!/usr/bin/env python3
"""Analysis A6. Innovation diagnostics for the scalar production configuration.

True innovations equal observations minus forecast ensemble means on update days.
The script reports mean innovations by period, lag-one correlation, posterior
spread, and normalized innovation variance (NIV). The declared NIV denominator
uses mean uninflated forecast variance plus observation-error variance.

A value near one is a diagnostic reference under those conventions. Agreement
with that reference alone does not establish filter consistency or independent
state accuracy. A10 checks additional lags and pairing conventions.

Author: Sebastian R.O. Marshall."""
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from A_temporal_subsampling import load, TOOL, RES  # noqa: E402
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                      # noqa: E402
from mekong_da.config import OPTIMAL_CONFIG              # noqa: E402
OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "analyses"


def main():
    d, dates = load(); rows = []
    R = OPTIMAL_CONFIG.obs_error_std ** 2
    for r in RES:
        f = d[r]; mask = f["obs_available"].to_numpy(bool)
        res = run_enkf(f["baseline"].to_numpy(float), f["observation_corrected"].to_numpy(float), mask, config=OPTIMAL_CONFIG)
        ok = np.isfinite(res.innovation); inn = res.innovation[ok]; pf = res.forecast_spread[ok] ** 2
        tr = pd.to_datetime(dates[ok]) < pd.Timestamp("2020-01-01")
        niv = inn.var() / (pf.mean() + R)
        ac1 = np.corrcoef(inn[:-1], inn[1:])[0, 1]
        post_spread = float(np.nanmean(res.spread))
        rows.append(dict(product=r, n_updates=int(ok.sum()), mean_posterior_spread=post_spread, mean_innovation=inn.mean(), mean_innovation_2015_2019=inn[tr].mean(),
                         mean_innovation_2020_2024=inn[~tr].mean(), innovation_var=inn.var(), mean_forecast_var=pf.mean(), R=R,
                         normalised_innovation_variance=niv, innovation_ac1=ac1))
        print(f"{r}: n={ok.sum()} mean d {inn.mean():+.4f} (2015-19 {inn[tr].mean():+.4f}, 2020-24 {inn[~tr].mean():+.4f}) var(d) {inn.var():.6f} "
              f"mean Pf {pf.mean():.6f} R {R:.4f} NIV {niv:.3f} AC1(d) {ac1:+.3f} posterior spread {post_spread:.4f}")
    pd.DataFrame(rows).to_csv(OUT / "A6_filter_consistency.csv", index=False); print("wrote", OUT / "A6_filter_consistency.csv")


if __name__ == "__main__":
    main()
