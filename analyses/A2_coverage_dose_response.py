#!/usr/bin/env python3
"""Analysis A2. Input-agreement RMSE under regular and random observation thinning.

The 9 km calendar retains every kth available retrieval for k in the declared
KEEP grid. Random subsets use matched counts, with ten draws per count. Both
designs run 50 ensemble seeds and score against the same corrected 9 km target
on 3,040 common observation days. Random selection does not preserve the original
clustering of outages. Regular sampling limits gaps in retrieval-index space,
rather than guaranteeing a fixed calendar-day interval.

The scored set includes days omitted by thinning but remains a selected subset
of the full 9 km calendar. R3_decomposition.csv separates updated and skipped
scored days. These results are conditional on that sample and the filter settings.
They do not establish a coverage-independent product ranking or retrieval accuracy.

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
SEEDS = list(range(42, 92))
KEEP = [1, 2, 3, 5, 7, 10, 14, 21, 30]      # keep every k-th available day


def cfg(seed):
    return EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, "seed": seed})


def gaps(mask):
    idx = np.flatnonzero(mask)
    g = np.diff(idx)
    return (float(g.mean()), float(np.percentile(g, 95)), int(g.max())) if g.size else (np.nan,) * 3


def run(ol, obs, mask, evald):
    rm = []
    for s in SEEDS:
        r = run_enkf(ol, obs, mask, config=cfg(s))
        rm.append(float(np.sqrt(np.mean((r.analysis[evald] - obs[evald]) ** 2))))
    return float(np.mean(rm)), float(np.std(rm, ddof=1))


def main():
    d = {r: pd.read_csv(TOOL / "data/enkf" / f"inputs_{r}.csv") for r in ("9km", "1km", "400m")}
    m = {r: d[r]["obs_available"].to_numpy(bool) for r in d}
    common = m["9km"] & m["1km"] & m["400m"]
    ol = d["9km"]["baseline"].to_numpy(float)
    obs = d["9km"]["observation_corrected"].to_numpy(float)
    full = m["9km"]
    avail = np.flatnonzero(full)
    ol_rmse = float(np.sqrt(np.mean((ol[common] - obs[common]) ** 2)))

    print("=" * 78)
    print("ANALYSIS A2. Coverage dose-response for the 9 km product")
    print("=" * 78)
    print(f"open loop RMSE on the {int(common.sum())} evaluation days: {ol_rmse:.4f} m3/m3")
    print(f"published post-DA RMSE, 1 km {0.0228:.4f}, 400 m {0.0248:.4f} (Analysis A)")
    print(f"\n  {'thinning':<16s}{'n upd':>7s}{'cover%':>8s}{'mean gap':>10s}"
          f"{'max gap':>9s}{'RMSE':>9s}{'sd':>8s}")
    rows = []
    for k in KEEP:
        mk = np.zeros_like(full)
        mk[avail[::k]] = True
        mg, p95, mx = gaps(mk)
        rmse, sd = run(ol, obs, mk, common)
        lab = "every day" if k == 1 else f"every {k}th day"
        rows.append(dict(pattern="regular", keep_every=k, n_updates=int(mk.sum()),
                         coverage_pct=100 * mk.sum() / mk.size, mean_gap_d=mg,
                         p95_gap_d=p95, max_gap_d=mx, rmse_mean=rmse, rmse_sd=sd))
        print(f"  {lab:<16s}{int(mk.sum()):>7d}{100*mk.sum()/mk.size:>8.1f}"
              f"{mg:>10.2f}{mx:>9d}{rmse:>9.4f}{sd:>8.5f}")

    print("\n  random thinning at matched counts (10 draws each, mean over draws and seeds)")
    print(f"  {'target n':<16s}{'n upd':>7s}{'cover%':>8s}{'mean gap':>10s}"
          f"{'max gap':>9s}{'RMSE':>9s}{'sd':>8s}")
    rng = np.random.default_rng(20260909)
    for k in KEEP:
        n = len(avail[::k])
        vals = []
        mg_l, mx_l = [], []
        for _ in range(10):
            pick = rng.choice(avail, size=n, replace=False)
            mk = np.zeros_like(full)
            mk[np.sort(pick)] = True
            mg, p95, mx = gaps(mk)
            mg_l.append(mg)
            mx_l.append(mx)
            vals.append(run(ol, obs, mk, common)[0])
        rows.append(dict(pattern="random", keep_every=k, n_updates=n,
                         coverage_pct=100 * n / full.size, mean_gap_d=float(np.mean(mg_l)),
                         p95_gap_d=np.nan, max_gap_d=int(np.mean(mx_l)),
                         rmse_mean=float(np.mean(vals)), rmse_sd=float(np.std(vals, ddof=1))))
        print(f"  {'n = '+str(n):<16s}{n:>7d}{100*n/full.size:>8.1f}"
              f"{np.mean(mg_l):>10.2f}{int(np.mean(mx_l)):>9d}"
              f"{np.mean(vals):>9.4f}{np.std(vals, ddof=1):>8.5f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A2_dose_response.csv", index=False)

    reg = df[df["pattern"] == "regular"].sort_values("keep_every")
    base = float(reg.iloc[0]["rmse_mean"])
    gap_1km = 0.0228 - base
    print("\n" + "-" * 78)
    print("Where does common-day thinning change RMSE by the own-target product difference?")
    print(f"  that difference is {gap_1km:+.4f} m3/m3")
    hit = reg[reg["rmse_mean"] - base >= gap_1km]
    if len(hit):
        h = hit.iloc[0]
        print(f"  first reached at every {int(h['keep_every'])}th day "
              f"(mean gap {h['mean_gap_d']:.1f} d, coverage {h['coverage_pct']:.1f}%)")
    else:
        print(f"  never reached, even at every {int(reg.iloc[-1]['keep_every'])}th day "
              f"(mean gap {reg.iloc[-1]['mean_gap_d']:.1f} d): "
              f"largest conditional thinning change is {reg['rmse_mean'].max()-base:+.4f}")
    print(f"\nwrote {OUT/'A2_dose_response.csv'}")


if __name__ == "__main__":
    main()
