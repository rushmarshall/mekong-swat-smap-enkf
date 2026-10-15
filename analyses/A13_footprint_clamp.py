#!/usr/bin/env python3
"""Analysis A13. Prior disagreement under alternate subbasin supports.

The script compares unweighted subbasin means on each product's own support,
the 9 km product's maximal support, and its daily support. Bias correction uses
2015 to 2019, and evaluation retains the same available dates across supports.
These subbasin composites differ from the extraction-box inputs to the scalar
filter. The fixed intersection also changes seasonal representation.

Fifty random subsets match the maximal support's subbasin count. This comparison
tests a restricted averaging-count explanation. Spatially incoherent draws do not
rule out effects of spatial coherence, location, land cover, or sample selection.
The output describes support sensitivity; it does not identify a unique cause
or rank independent retrieval accuracy. Assertions preserve the recorded comparison
and should be interpreted within these limits.

Author: Sebastian R.O. Marshall."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT
OUT = ROOT / "outputs" / "analyses"
sys.path.insert(0, str(T / "src"))
from mekong_da.biascorrect import cdf_match_rank  # noqa: E402

RES = ("9km", "1km", "400m")
DOWN = ("1km", "400m")
TRAIN_END = pd.Timestamp("2019-12-31")
N_DRAWS = 50
SEED = 42


def load():
    base = pd.read_csv(T / "data/enkf/inputs_9km.csv", parse_dates=["date"]).set_index("date")["baseline"]
    sub = {}
    for r in RES:
        path = T / f"data/smap_subbasin_daily/smap_{r}_subbasin_daily.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found: this analysis needs your own subbasin "
                             "SMAP series (see docs/03_data.md).")
        s = pd.read_csv(path,
                        parse_dates=["date"]).set_index("date")
        sub[r] = s[[c for c in s.columns if c.startswith("sub_")]].reindex(base.index)
    return base, sub


def composite(df, mask=None):
    """Unweighted mean over valid subbasins; NaN on days with no valid subbasin."""
    a = df.to_numpy(dtype=float)
    if mask is not None:
        a = np.where(mask, a, np.nan)
    n = (~np.isnan(a)).sum(axis=1)
    v = np.full(len(a), np.nan)
    v[n > 0] = np.nanmean(a[n > 0], axis=1)
    return v


def disagreement(v, base_arr, train_mask, day_mask):
    c = cdf_match_rank(v, base_arr, train_mask=train_mask)
    return float(np.sqrt(np.nanmean((c[day_mask] - base_arr[day_mask]) ** 2)))


def main():
    t0 = time.perf_counter()
    base, sub = load()
    b = base.to_numpy()
    train = base.index <= TRAIN_END
    cols9 = [c for c in sub["9km"].columns if sub["9km"][c].notna().any()]
    m9 = sub["9km"][cols9].notna().to_numpy()
    print(f"subbasins the 9 km chain ever populates: {len(cols9)} of {sub['9km'].shape[1]}")
    daily = m9.sum(axis=1)
    print(f"9 km daily footprint on days it observes: median {int(np.median(daily[daily > 0]))} "
          f"of {len(cols9)} subbasins")

    variants = {"9km|own field": composite(sub["9km"][cols9])}
    for r in DOWN:
        variants[f"{r}|own field"] = composite(sub[r])
        variants[f"{r}|maximal clamp"] = composite(sub[r][cols9])
        variants[f"{r}|daily clamp"] = composite(sub[r][cols9], mask=m9)

    fixed = np.ones(len(base), dtype=bool)
    for v in variants.values():
        fixed &= ~np.isnan(v)
    n_fixed = int(fixed.sum())
    print(f"day set held fixed at n = {n_fixed}\n")

    rows, val = [], {}
    for key, v in variants.items():
        product, support = key.split("|")
        val[key] = disagreement(v, b, train, fixed)
        rows.append(dict(part="a_supports", product=product, support=support,
                         rmse=val[key], n_days=n_fixed))
        print(f"  {product:>5}  {support:<15} {val[key]:.4f} m3/m3")

    print()
    for r in DOWN:
        for support in ("own field", "maximal clamp", "daily clamp"):
            gap = val[f"{r}|{support}"] - val["9km|own field"]
            rows.append(dict(part="b_gaps", product=r, support=support, rmse=gap, n_days=n_fixed))
            print(f"  {r:>5} {support:<15} minus 9 km: {gap:+.4f}")

    print(f"\nnegative control: {N_DRAWS} random {len(cols9)}-subbasin draws per downscaled product")
    rng = np.random.default_rng(SEED)
    control = {}
    for r in DOWN:
        own = [c for c in sub[r].columns if sub[r][c].notna().any()]
        draws = np.array([disagreement(composite(sub[r][list(rng.choice(own, size=len(cols9),
                                                                       replace=False))]),
                                       b, train, fixed) for _ in range(N_DRAWS)])
        control[r] = draws
        rows.append(dict(part="c_random_support_control", product=r, support=f"median of {N_DRAWS}",
                         rmse=float(np.median(draws)), n_days=n_fixed))
        rows.append(dict(part="c_random_support_control", product=r, support="min",
                         rmse=float(draws.min()), n_days=n_fixed))
        rows.append(dict(part="c_random_support_control", product=r, support="max",
                         rmse=float(draws.max()), n_days=n_fixed))
        print(f"  {r:>5}: own field {val[f'{r}|own field']:.4f} | clamped {val[f'{r}|maximal clamp']:.4f}"
              f" | random {len(cols9)} of {len(own)}: median {np.median(draws):.4f}, "
              f"range {draws.min():.4f} to {draws.max():.4f}")

    df = pd.DataFrame(rows)
    dest = OUT / "A13_footprint_clamp.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest.relative_to(ROOT)}  ({len(df)} rows)")

    # --- self-check -------------------------------------------------------------------
    # 1. Check the recorded narrowing under the maximal support restriction.
    gap_own = val["1km|own field"] - val["9km|own field"]
    gap_clamp = val["1km|maximal clamp"] - val["9km|own field"]
    assert gap_own > gap_clamp, f"clamp did not narrow the 1 km gap: {gap_own:+.4f} -> {gap_clamp:+.4f}"
    # 2. Compare random subsets with the own-support and clamped values.
    #    This tests averaging count within the random-subset design.
    #    It does not separate location, spatial coherence, or other spatial covariates.
    for r in DOWN:
        lo, hi = control[r].min(), control[r].max()
        assert not (lo <= val[f"{r}|maximal clamp"] <= hi), (
            f"{r}: clamped value {val[f'{r}|maximal clamp']:.4f} lies INSIDE the random-support "
            f"range {lo:.4f} to {hi:.4f}, so the recorded separation from random subsets is not reproduced")
        assert lo <= val[f"{r}|own field"] <= hi, (
            f"{r}: own-field value {val[f'{r}|own field']:.4f} lies outside the random-support "
            f"range {lo:.4f} to {hi:.4f}, so the control is not measuring what it claims")
    # 3. Verify the recorded own-support ordering for this diagnostic sample.
    assert val["9km|own field"] < val["1km|own field"] < val["400m|own field"], "own-support ordering moved"
    print(f"self-check passed | runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
