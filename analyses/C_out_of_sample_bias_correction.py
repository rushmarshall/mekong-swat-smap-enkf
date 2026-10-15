#!/usr/bin/env python3
"""Analysis C. Sensitivity to the empirical bias-correction training window.

Two candidate cumulative-distribution-function mappings are compared with the
archived corrected column. The smaller residual identifies the closer tested
reconstruction, rather than proving exact historical implementation. The remaining
residual is reported explicitly.

The archived mapping and both reconstructions are assimilated over 50 seeds.
Evaluation uses available 2020 to 2024 observations. A held-out mapping period
does not make these targets independent: each evaluated target is assimilated.
RMSE reduction therefore measures input agreement. Analysis-minus-baseline RMS
measures state displacement and likewise does not establish independent accuracy.

Author: Sebastian R.O. Marshall."""
import sys
import pathlib
import numpy as np
import pandas as pd

TOOL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                      # noqa: E402
from mekong_da.biascorrect import cdf_match_rank         # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "analyses"
OUT.mkdir(parents=True, exist_ok=True)
RES = ["9km", "1km", "400m"]
SEEDS = list(range(42, 92))
SPLIT = "2020-01-01"


def cfg(seed):
    return EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, "seed": seed})


def filt(ol, obs, mask, evald):
    a_o, a_ol = [], []
    for s in SEEDS:
        r = run_enkf(ol, obs, mask, config=cfg(s))
        a_o.append(float(np.sqrt(np.mean((r.analysis[evald] - obs[evald]) ** 2))))
        a_ol.append(float(np.sqrt(np.mean((r.analysis[evald] - ol[evald]) ** 2))))
    return float(np.mean(a_o)), float(np.std(a_o, ddof=1)), float(np.mean(a_ol))


def main():
    print("=" * 78)
    print("ANALYSIS C. The training window, and what the soil-moisture metric measures")
    print("=" * 78)
    rows = []
    for r in RES:
        f = pd.read_csv(TOOL / "data/enkf" / f"inputs_{r}.csv")
        date = pd.to_datetime(f["date"])
        ol = f["baseline"].to_numpy(float)
        raw = f["observation"].to_numpy(float)
        pub = f["observation_corrected"].to_numpy(float)
        mask = f["obs_available"].to_numpy(bool)
        pre = (date < SPLIT).to_numpy()
        test = (date >= SPLIT).to_numpy()
        evald = mask & test
        fin = np.isfinite(raw) & np.isfinite(pub)

        # ---- Step 1. Compare two candidate reconstructions of the archived column ----------------------
        cand = {
            "trained on the whole record": cdf_match_rank(raw, ol),
            "trained on 2015 to 2019 only": cdf_match_rank(raw, ol, train_mask=pre),
        }
        print(f"\n--- {r} ---")
        print("  which tested training window is closer to the archived corrected column?")
        resid = {}
        for lab, got in cand.items():
            mx = float(np.nanmax(np.abs(got[fin] - pub[fin])))
            resid[lab] = mx
            print(f"    {lab:<32s} max |difference| = {mx:.3e} m3/m3")
        won = min(resid, key=resid.get)
        print(f"    closest tested reconstruction: {won} "
              f"(smaller by a factor of {resid[max(resid, key=resid.get)]/resid[won]:.0f})")

        # ---- Step 2. Three corrections that differ materially from one another -------------
        arms = {
            "published (2015 to 2019)": pub,
            "rebuilt on 2015 to 2019": cand["trained on 2015 to 2019 only"],
            "rebuilt on the whole record": cand["trained on the whole record"],
        }
        spread = float(np.nanmax(np.abs(arms["rebuilt on the whole record"][fin]
                                        - arms["published (2015 to 2019)"][fin])))
        print(f"\n  the three targets differ from one another by up to {spread:.3e} m3/m3.")
        print(f"  scored on the {int(evald.sum())} held-out observations from {SPLIT}:")
        print(f"    {'target series':<30s}{'open loop':>11s}{'post-DA':>10s}"
              f"{'reduction':>11s}{'sd':>9s}{'vs open loop':>14s}")
        for lab, series in arms.items():
            olr = float(np.sqrt(np.mean((ol[evald] - series[evald]) ** 2)))
            a, sd, ind = filt(ol, series, mask, evald)
            print(f"    {lab:<30s}{olr:>11.4f}{a:>10.4f}{100*(olr-a)/olr:>10.1f}%"
                  f"{sd:>9.5f}{ind:>14.4f}")
            rows.append(dict(product=r, target=lab, n_eval=int(evald.sum()),
                             rmse_open_loop=olr, rmse_post_da=a, rmse_sd=sd,
                             reduction_pct=100 * (olr - a) / olr,
                             analysis_vs_open_loop=ind,
                             resid_vs_published=float(np.nanmax(np.abs(series[fin] - pub[fin]))),
                             identified_training=won))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "C_out_of_sample.csv", index=False)

    print("\n" + "-" * 78)
    print("Input-agreement reduction across the three tested bias-corrected targets:")
    print("the following ranges are conditional on these targets and update days:")
    for r in RES:
        s = df[df["product"] == r]["reduction_pct"]
        print(f"  {r:>5s}: {s.min():.1f}% to {s.max():.1f}%  (range {s.max()-s.min():.1f} points)")
    print("\nEach target is assimilated on the evaluation day, so these scores measure")
    print("agreement with that input. They do not estimate independent retrieval accuracy.")
    print("The held-out correction period does not create an independent evaluation target.")
    print(f"\nwrote {OUT/'C_out_of_sample.csv'}")


if __name__ == "__main__":
    main()
