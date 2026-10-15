#!/usr/bin/env python3
"""Analysis A3. Paired block intervals for the common-day schedule contrast.

Daily squared errors are averaged across 50 seeds before resampling. Each of
2,000 paired moving-block draws concatenates blocks of 30 consecutive scored
days and truncates the result to the original length. The score uses 3,040
common days against the corrected 9 km target. Its RMSE is the root of the
seed-and-day mean square, rather than the mean of per-seed RMSE values.

The interval describes temporal sampling within this selected set. It excludes
direct error on omitted update days outside that set and does not measure
independent soil-moisture accuracy. A11 examines longer scored-day blocks.

Author: Sebastian R.O. Marshall."""
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from A_temporal_subsampling import load, TOOL, SEEDS, RES  # noqa: E402
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                      # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "analyses"
B, L = 2000, 30


def seed_mean_sqerr(frame, mask, sel, y):
    """Per-day squared error against y on the selected days, averaged over the seeds; also the per-seed RMSEs."""
    ol = frame["baseline"].to_numpy(float); obs = frame["observation_corrected"].to_numpy(float)
    acc = np.zeros(int(sel.sum())); per_seed = []
    for s in SEEDS:
        a = run_enkf(ol, obs, mask, config=EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, "seed": s})).analysis[sel]
        e2 = (a - y) ** 2; acc += e2; per_seed.append(float(np.sqrt(e2.mean())))
    return acc / len(SEEDS), np.array(per_seed)


def block_boot(x, rng):
    n = len(x); nb = int(np.ceil(n / L)); out = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n - L + 1, nb)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        out[b] = x[idx].mean()
    return out


def main():
    d, _ = load()
    m = {r: d[r]["obs_available"].to_numpy(bool) for r in RES}
    common = m["9km"] & m["1km"] & m["400m"]
    y = d["9km"]["observation_corrected"].to_numpy(float)[common]
    e_full, ps_full = seed_mean_sqerr(d["9km"], m["9km"], common, y)
    rng = np.random.default_rng(42)
    rows = []
    for tag, mk in (("the 1 km dates", m["9km"] & m["1km"]), ("the 400 m dates", m["9km"] & m["400m"])):
        e_deg, ps_deg = seed_mean_sqerr(d["9km"], mk, common, y)
        rmse_full, rmse_deg = np.sqrt(e_full.mean()), np.sqrt(e_deg.mean())
        # bootstrap the RMSE difference on paired blocks
        n = len(y); nb = int(np.ceil(n / L)); diffs = np.empty(B)
        for b in range(B):
            starts = rng.integers(0, n - L + 1, nb)
            idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
            diffs[b] = np.sqrt(e_deg[idx].mean()) - np.sqrt(e_full[idx].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append(dict(schedule=tag, n_days=int(common.sum()), rmse_full=rmse_full, rmse_degraded=rmse_deg,
                         delta=rmse_deg - rmse_full, ci_lo=lo, ci_hi=hi, block_days=L, B=B, seeds=len(SEEDS),
                         per_seed_rmse_mean_full=ps_full.mean(), per_seed_rmse_mean_degraded=ps_deg.mean(),
                         per_seed_delta_mean=(ps_deg - ps_full).mean()))
        print(f"{tag}: n={common.sum()} full {rmse_full:.5f} degraded {rmse_deg:.5f} delta {rmse_deg-rmse_full:+.2e} "
              f"95% paired block-bootstrap [{lo:+.2e}, {hi:+.2e}] -> {'interval excludes zero' if (lo > 0 or hi < 0) else 'interval includes zero under this block design'}")
    pd.DataFrame(rows).to_csv(OUT / "A3_daylevel_interval.csv", index=False)
    print("wrote", OUT / "A3_daylevel_interval.csv")


if __name__ == "__main__":
    main()
