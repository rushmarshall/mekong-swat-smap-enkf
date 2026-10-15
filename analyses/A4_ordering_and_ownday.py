#!/usr/bin/env python3
"""Analysis A4. Separate scoring-calendar and update-schedule contributions.

The full 9 km arm is rescored on each reduced arm's update days. The difference
from its full calendar measures sample selection. The reduced-minus-full contrast
on those same days measures the conditional schedule contribution.

Paired intervals use 2,000 moving-block draws of 30 scored days. Daily squared
errors are averaged over 50 seeds before resampling. Cross-product comparisons
use 3,040 common days but retain each product's own corrected target. Their
ordering therefore concerns input agreement under the specified filter, not
independent retrieval accuracy. A11 provides block-length sensitivity.

Author: Sebastian R.O. Marshall."""
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from A_temporal_subsampling import load, TOOL, SEEDS, RES  # noqa: E402
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                      # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG  # noqa: E402
OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "analyses"; B, L = 2000, 30


def seed_mean_sqerr(frame, mask, y_all):
    """Per-day squared error against y_all on EVERY day, averaged over the seeds."""
    ol = frame["baseline"].to_numpy(float); obs = frame["observation_corrected"].to_numpy(float)
    acc = np.zeros_like(ol)
    for s in SEEDS:
        a = run_enkf(ol, obs, mask, config=EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, "seed": s})).analysis
        acc += (a - y_all) ** 2
    return acc / len(SEEDS)


def boot_diff(e_a, e_b, rng):
    """95 % interval of RMSE(a) - RMSE(b) from paired 30-day moving blocks."""
    n = len(e_a); nb = int(np.ceil(n / L)); d = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - L + 1, nb); idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        d[b] = np.sqrt(e_a[idx].mean()) - np.sqrt(e_b[idx].mean())
    return np.sqrt(e_a.mean()) - np.sqrt(e_b.mean()), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    d, _ = load(); m = {r: d[r]["obs_available"].to_numpy(bool) for r in RES}; common = m["9km"] & m["1km"] & m["400m"]
    rng = np.random.default_rng(42); rows = []
    y9 = d["9km"]["observation_corrected"].to_numpy(float)
    e_full = seed_mean_sqerr(d["9km"], m["9km"], y9)
    # (1) own-day decomposition
    gap50 = pd.read_csv(OUT / "R2_evalsets.csv", float_precision="round_trip").dropna(subset=["published_gap"]).groupby("arm")["published_gap"].first().to_dict()   # 50-seed own-day gaps
    for tag, mk in (("the 1 km dates", m["9km"] & m["1km"]), ("the 400 m dates", m["9km"] & m["400m"])):
        e_deg = seed_mean_sqerr(d["9km"], mk, y9)
        full_all = np.sqrt(e_full[m["9km"]].mean()); full_on_deg = np.sqrt(e_full[mk].mean()); deg_on_deg = np.sqrt(e_deg[mk].mean())
        delta, lo, hi = boot_diff(e_deg[mk], e_full[mk], rng)
        ac1 = np.corrcoef((e_deg[mk] - e_full[mk])[:-1], (e_deg[mk] - e_full[mk])[1:])[0, 1]
        rows.append(dict(block="own-day decomposition", schedule=tag, n_days=int(mk.sum()), rmse_full_own3447=full_all, rmse_full_on_degraded_days=full_on_deg,
                         rmse_degraded=deg_on_deg, total=deg_on_deg - full_all, part_dayset=full_on_deg - full_all, part_schedule=delta, ci_lo=lo, ci_hi=hi,
                         pct_total=100 * (deg_on_deg - full_all) / gap50[tag], pct_dayset=100 * (full_on_deg - full_all) / gap50[tag], pct_schedule=100 * delta / gap50[tag], ac1_paired_diff=ac1))
        print(f"{tag}: total {100*(deg_on_deg-full_all)/gap50[tag]:+.1f} % of the gap = day set {100*(full_on_deg-full_all)/gap50[tag]:+.1f} + schedule {100*delta/gap50[tag]:+.1f} [{100*lo/gap50[tag]:+.1f}, {100*hi/gap50[tag]:+.1f}]; AC1 of paired diff {ac1:+.3f}")
    # AC1 of the common-day paired difference in the common-day schedule diagnostic
    e_deg1 = seed_mean_sqerr(d["9km"], m["9km"] & m["1km"], y9); dd = (e_deg1 - e_full)[common]
    print(f"Common-day paired daily difference (1 km calendar, common days): AC1 {np.corrcoef(dd[:-1], dd[1:])[0,1]:+.3f}")
    # (2) ordering intervals on the common days
    prior = {r: ((d[r]["observation_corrected"] - d[r]["baseline"]).to_numpy(float) ** 2)[common] for r in RES}
    post = {r: seed_mean_sqerr(d[r], m[r], d[r]["observation_corrected"].to_numpy(float))[common] for r in RES}
    for kind, E in (("prior disagreement", prior), ("post-assimilation error", post)):
        for r in RES: rows.append(dict(block=kind, schedule=r, n_days=int(common.sum()), value=float(np.sqrt(E[r].mean()))))
        for a, b in (("1km", "9km"), ("400m", "1km"), ("400m", "9km")):
            delta, lo, hi = boot_diff(E[a], E[b], rng)
            rows.append(dict(block=kind, schedule=f"{a} minus {b}", n_days=int(common.sum()), value=delta, ci_lo=lo, ci_hi=hi))
            print(f"{kind} {a} minus {b}: {delta:+.4f} [{lo:+.4f}, {hi:+.4f}] {'interval excludes zero' if lo > 0 or hi < 0 else 'interval includes zero'}")
    pd.DataFrame(rows).to_csv(OUT / "A4_ordering_and_ownday.csv", index=False); print("wrote", OUT / "A4_ordering_and_ownday.csv")


if __name__ == "__main__":
    main()
