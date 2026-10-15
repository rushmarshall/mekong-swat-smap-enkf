#!/usr/bin/env python3
"""Analysis A. Conditional input agreement under reduced observation schedules.

The 9 km observations are assimilated on their full calendar and on intersections
with the 1 km and 400 m calendars. Observation values, bias correction, baseline,
filter parameters, and the 3,040 common scored days remain fixed. Each arm uses
50 seeds, 42 to 91, and reports the mean and spread of per-seed RMSE.

These common-day scores omit direct error on missed update days. They retain
effects of update history and sample selection. R2_evalsets.csv supplies the
complementary fixed-calendar comparison, which includes omitted update days.
Cross-product scores use different corrected targets and do not rank independent
soil-moisture accuracy. Ratios to those differences are descriptive normalizations.

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
SEEDS = list(range(42, 92))          # 50 realisations, starting at the published seed


def load():
    d = {}
    for r in RES:
        f = pd.read_csv(TOOL / "data/enkf" / f"inputs_{r}.csv")
        f["date"] = pd.to_datetime(f["date"])
        d[r] = f
    dates = d[RES[0]]["date"].to_numpy()
    for r in RES[1:]:
        assert (d[r]["date"].to_numpy() == dates).all(), f"{r} is not day-aligned with 9km"
    return d, dates


def gap_stats(mask):
    """Calendar-day gaps between consecutive assimilations; this is one availability diagnostic."""
    idx = np.flatnonzero(mask)
    gaps = np.diff(idx)
    return dict(n_obs=int(mask.sum()),
                coverage_pct=100.0 * mask.sum() / mask.size,
                mean_gap_d=float(gaps.mean()) if gaps.size else np.nan,
                p95_gap_d=float(np.percentile(gaps, 95)) if gaps.size else np.nan,
                max_gap_d=int(gaps.max()) if gaps.size else 0)


def arm(frame, mask, evald, seeds=SEEDS, cfg=OPTIMAL_CONFIG):
    """Run one assimilation schedule over every seed; score on the common evaluation days."""
    ol = frame["baseline"].to_numpy(float)
    obs = frame["observation_corrected"].to_numpy(float)
    rmses, rs = [], []
    for s in seeds:
        r = run_enkf(ol, obs, mask, config=EnKFConfig(**{**cfg.__dict__, "seed": s}))
        a = r.analysis[evald]
        y = obs[evald]
        rmses.append(float(np.sqrt(np.mean((a - y) ** 2))))
        rs.append(float(np.corrcoef(a, y)[0, 1]))
    return np.array(rmses), np.array(rs)


def main():
    d, dates = load()
    m = {r: d[r]["obs_available"].to_numpy(bool) for r in RES}
    common = m["9km"] & m["1km"] & m["400m"]

    print("=" * 78)
    print("ANALYSIS A. Common-day input agreement under reduced schedules")
    print("=" * 78)
    print(f"record: {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}, "
          f"{len(dates)} days")
    print("\nObservation schedules, as published:")
    print(f"  {'schedule':<22s}{'n':>6s}{'cover%':>9s}{'mean gap':>10s}{'p95 gap':>9s}{'max gap':>9s}")
    sched = {r: m[r] for r in RES}
    sched["9km AND 1km dates"] = m["9km"] & m["1km"]
    sched["9km AND 400m dates"] = m["9km"] & m["400m"]
    sched["common to all three"] = common
    grows = []
    for k, mk in sched.items():
        g = gap_stats(mk)
        grows.append(dict(schedule=k, **g))
        print(f"  {k:<22s}{g['n_obs']:>6d}{g['coverage_pct']:>9.1f}"
              f"{g['mean_gap_d']:>10.2f}{g['p95_gap_d']:>9.1f}{g['max_gap_d']:>9d}")
    pd.DataFrame(grows).to_csv(OUT / "A_schedules.csv", index=False)

    ol_rmse = {}
    for r in RES:
        ol_rmse[r] = float(np.sqrt(np.mean(
            (d[r]["baseline"].to_numpy(float)[common]
             - d[r]["observation_corrected"].to_numpy(float)[common]) ** 2)))

    # ---- PRIMARY: the 9 km product under three schedules, one target, one day set ---------
    print(f"\nPrimary test. The 9 km product only. Same observations, same target series,")
    print(f"same {int(common.sum())} evaluation days. Only the update schedule changes.")
    print(f"  open loop RMSE on those days: {ol_rmse['9km']:.4f} m3/m3")
    print(f"\n  {'9 km run under':<24s}{'n upd':>7s}{'RMSE':>9s}{'sd':>8s}{'vs full':>9s}")
    prim, rows = {}, []
    for lab, mk in (("its own dates", m["9km"]),
                    ("the 1 km dates", m["9km"] & m["1km"]),
                    ("the 400 m dates", m["9km"] & m["400m"])):
        rm, rr = arm(d["9km"], mk, common)
        prim[lab] = rm
        delta = rm.mean() - prim["its own dates"].mean()
        rows.append(dict(product="9km", schedule=lab, n_updates=int(mk.sum()),
                         rmse_mean=rm.mean(), rmse_sd=rm.std(ddof=1),
                         r_mean=rr.mean(), delta_vs_full=delta))
        print(f"  {lab:<24s}{int(mk.sum()):>7d}{rm.mean():>9.4f}{rm.std(ddof=1):>8.5f}"
              f"{delta:>+9.4f}")

    # ---- SECONDARY: all three products at their own schedules, same evaluation days -------
    print(f"\nSecondary. Each product at its own schedule, scored on the same days against")
    print(f"its own corrected series (the published comparison, restricted to common days).")
    print(f"  {'product':<24s}{'n upd':>7s}{'RMSE':>9s}{'sd':>8s}{'open loop':>11s}")
    sec = {}
    for r in RES:
        rm, rr = arm(d[r], m[r], common)
        sec[r] = rm
        rows.append(dict(product=r, schedule="its own dates (published)",
                         n_updates=int(m[r].sum()), rmse_mean=rm.mean(),
                         rmse_sd=rm.std(ddof=1), r_mean=rr.mean(), delta_vs_full=np.nan))
        print(f"  {r:<24s}{int(m[r].sum()):>7d}{rm.mean():>9.4f}{rm.std(ddof=1):>8.5f}"
              f"{ol_rmse[r]:>11.4f}")

    # ---- Descriptive normalization of the conditional common-day change -------------------------
    print("\n" + "-" * 78)
    print("Common-day schedule change relative to the own-target cross-product difference")
    for fine in ("1km", "400m"):
        lab = "the 1 km dates" if fine == "1km" else "the 400 m dates"
        gap = sec[fine].mean() - sec["9km"].mean()
        closed = prim[lab].mean() - prim["its own dates"].mean()
        frac = 100.0 * closed / gap if gap != 0 else np.nan
        print(f"  9 km vs {fine:>4s}: own-target gap {gap:+.4f} m3/m3; "
              f"degrading 9 km to the {fine} schedule moves it {closed:+.4f}, "
              f"a descriptive {frac:.1f}% of that gap; not an accuracy attribution")
        rows.append(dict(product=f"9km_vs_{fine}", schedule="gap decomposition",
                         n_updates=np.nan, rmse_mean=np.nan, rmse_sd=np.nan,
                         r_mean=np.nan, delta_vs_full=np.nan,
                         published_gap=gap, closed_by_sampling=closed, pct_of_gap=frac))
    pd.DataFrame(rows).to_csv(OUT / "A_subsampling.csv", index=False)
    print(f"\nwrote {OUT/'A_subsampling.csv'} and {OUT/'A_schedules.csv'}")


if __name__ == "__main__":
    main()
