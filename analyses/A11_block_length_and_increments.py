#!/usr/bin/env python3
"""Analysis A11. Block-length sensitivity and scalar analysis displacements.

Daily squared errors are averaged across 50 seeds before paired resampling.
The ordering comparison uses 3,040 common days and 10,000 moving-block draws
for scored-day lengths 7, 15, 30, 60, 90, 120, and 180. An explicit replay
checks the original A4 random stream and 2,000-draw intervals. Prefix-sum
resampling is checked against direct indexed resampling.

The own-day diagnostic repeats RMSE, correlation, and reduction intervals for
seed 42 and the 50-seed ensemble at three block lengths. PRINTED_S16 contains
historical scalar-table values for a precision check. Cross-product scores retain different corrected input targets.

The increment output records analysis-minus-baseline displacements and corrected
observation-minus-baseline differences by period and calendar. Scalar displacements
are not coupled SWAT water-budget increments. The interval-width criterion is
a declared grid diagnostic; it does not establish an optimal block length.

Author: Sebastian R.O. Marshall."""
import os, sys, time, pathlib, multiprocessing
sys.dont_write_bytecode = True                     # this run writes the three CSVs and nothing else
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"        # inherited by the joblib workers
import numpy as np, pandas as pd
from joblib import Parallel, delayed
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from A_temporal_subsampling import load, TOOL, SEEDS, RES  # noqa: E402
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                      # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG  # noqa: E402

OUT = HERE.parent / "outputs" / "analyses"
FULL_CORES = multiprocessing.cpu_count()
B, L_GRID, L_S16, LAGS = 10_000, (7, 15, 30, 60, 90, 120, 180), (7, 30, 90), (1, 5, 10, 20, 30)
B_A4, L_A4 = 2000, 30
KINDS = ("prior disagreement", "post-assimilation error")
PAIRS_A4 = (("1km", "9km"), ("400m", "1km"), ("400m", "9km"))      # A4's order, which fixes its rng stream
PAIRS_C = (("1km", "9km"), ("400m", "1km"))                         # the four common-day cross-product contrasts
PRINTED_S16 = {   # historical scalar Table S16: (point, lo, hi)
    "9km": dict(rmse=(0.0195, 0.019, 0.020), r=(0.935, 0.929, 0.941), reduction_pct=(58.9, 57.9, 59.8)),
    "1km": dict(rmse=(0.0229, 0.022, 0.024), r=(0.907, 0.899, 0.915), reduction_pct=(60.5, 59.7, 61.3)),
    "400m": dict(rmse=(0.0248, 0.024, 0.026), r=(0.899, 0.890, 0.907), reduction_pct=(60.6, 59.9, 61.4))}
DP = dict(rmse=(4, 3), r=(3, 3), reduction_pct=(1, 1))              # printed decimals: (point, interval)
SRC42, SRC50, SRCOBS = "seed 42 production run", "50-seed mean", "observation series (seed-independent)"


def one(ol, y, mask, s):
    return run_enkf(ol, y, mask, config=EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, "seed": s})).analysis


def starts(rng, n, L, b=B):
    return rng.integers(0, n - L + 1, (b, -(-n // L)))


def gather_idx(st, n, L):
    """A3/A4's resample index: the blocks concatenated and truncated to n."""
    return (st[:, :, None] + np.arange(L)).reshape(len(st), -1)[:, :n]


def bmean(x, st, L):
    """Mean of x over each resample, from prefix sums: nb - 1 full blocks plus the first n - (nb - 1) L days of the last."""
    n = len(x); c = np.concatenate(([0.0], np.cumsum(x))); head, tail = st[:, :-1], st[:, -1]; r = n - head.shape[1] * L
    return ((c[head + L] - c[head]).sum(1) + c[tail + r] - c[tail]) / n


def acf(x, lags):
    x = x - x.mean(); v = x @ x
    return [float(x[:-k] @ x[k:] / v) for k in lags]


def stabilised(Ls, w):
    for i in range(len(Ls) - 1):
        if all(abs(w[j] / w[i] - 1) < 0.10 for j in range(i + 1, len(Ls))):
            return Ls[i]
    return None


def boot_corr(S, y, st, L):
    """Mean over the rows of S of the per-resample correlation with y (both centred first; R is shift-invariant)."""
    yc = y - y.mean(); my, myy = bmean(yc, st, L), bmean(yc * yc, st, L); acc = np.zeros(len(st))
    for a in S:
        ac = a - a.mean(); ma = bmean(ac, st, L)
        acc += (bmean(ac * yc, st, L) - ma * my) / np.sqrt((bmean(ac * ac, st, L) - ma ** 2) * (myy - my ** 2))
    return acc / len(S)


def fmt(x, d):
    return f"{x:.{d}f}"


def main():
    t0 = time.perf_counter()
    d, dates = load(); N = len(dates); dates = pd.DatetimeIndex(dates)
    M = {r: d[r]["obs_available"].to_numpy(bool) for r in RES}
    OL = {r: d[r]["baseline"].to_numpy(float) for r in RES}
    Y = {r: d[r]["observation_corrected"].to_numpy(float) for r in RES}
    common = M["9km"] & M["1km"] & M["400m"]; nc = int(common.sum())
    assert SEEDS[0] == 42 and len(SEEDS) == 50
    for r in RES:
        assert np.array_equal(np.isfinite(Y[r]), M[r]), f"{r}: observation_corrected finite != obs_available"
    print(f"A11 | {FULL_CORES} cores | record {dates[0].date()} to {dates[-1].date()} ({N} days) | common days {nc} | "
          f"own days " + ", ".join(f"{r} {int(M[r].sum())}" for r in RES))

    # ---- 150 filter runs: each product on its own calendar, seeds 42..91 --------------------------------------------
    runs = Parallel(n_jobs=FULL_CORES)(delayed(one)(OL[r], Y[r], M[r], s) for r in RES for s in SEEDS)
    A = {r: np.array(runs[i * len(SEEDS):(i + 1) * len(SEEDS)]) for i, r in enumerate(RES)}
    for r in RES:
        ref = pd.read_csv(TOOL / "data/enkf" / f"reference_{r}.csv")["analysis"].to_numpy(float)
        assert np.abs(A[r][0] - ref).max() < 1e-12, f"{r}: seed 42 is not the archived production run"
    print(f"filter runs done in {time.perf_counter() - t0:.1f} s; seed 42 equals the archived production series (all three)")

    # ---- (1) Common-day cross-product contrasts: block length -------------------------------------------------------------------------
    E = {KINDS[0]: {r: ((Y[r] - OL[r]) ** 2)[common] for r in RES},
         KINDS[1]: {r: ((A[r] - Y[r]) ** 2).mean(0)[common] for r in RES}}
    a4 = pd.read_csv(OUT / "A4_ordering_and_ownday.csv")
    # A4 replay on its own stream (per-iteration draws, the two own-day calls first) and its gather form
    rng = np.random.default_rng(42)
    draw = lambda n: np.stack([rng.integers(0, n - L_A4 + 1, -(-n // L_A4)) for _ in range(B_A4)])
    draw(int((M["9km"] & M["1km"]).sum())); draw(int((M["9km"] & M["400m"]).sum()))
    replay = {}
    print("\nA4 replay (L = 30, B = 2,000, default_rng(42)) against A4_ordering_and_ownday.csv")
    for kind in KINDS:
        for r in RES:
            v = float(a4.loc[(a4.block == kind) & (a4.schedule == r), "value"].iloc[0])
            assert abs(np.sqrt(E[kind][r].mean()) - v) < 1e-9, (kind, r)
        for a, b in PAIRS_A4:
            idx = gather_idx(draw(nc), nc, L_A4)
            dd = np.sqrt(E[kind][a][idx].mean(1)) - np.sqrt(E[kind][b][idx].mean(1))
            pt = np.sqrt(E[kind][a].mean()) - np.sqrt(E[kind][b].mean()); lo, hi = np.percentile(dd, [2.5, 97.5])
            row = a4[(a4.block == kind) & (a4.schedule == f"{a} minus {b}")].iloc[0]
            dl, dh = lo - row.ci_lo, hi - row.ci_hi
            print(f"  {kind:24s} {a:>4s} minus {b:<4s} {pt:+.6f} [{lo:+.6f}, {hi:+.6f}]  vs A4 d_lo {dl:+.1e} d_hi {dh:+.1e}")
            assert abs(pt - row.value) < 1e-9 and abs(dl) < 2e-4 and abs(dh) < 2e-4, (kind, a, b)
            replay[(kind, a, b)] = (lo, hi)
    print("  ASSERT PASSED: 30-day block-(C) intervals reproduce A4 within 2e-4; point values within 1e-9")

    # check the prefix-sum resample mean against the direct gather on one case with many blocks
    st = starts(np.random.default_rng([99, 7]), nc, 7, 500); x = E[KINDS[0]]["1km"]
    assert np.abs(bmean(x, st, 7) - x[gather_idx(st, nc, 7)].mean(1)).max() < 1e-14

    rows = []
    for L in L_GRID:
        st = starts(np.random.default_rng([11, L]), nc, L)            # one index per L, shared by the four rows
        for kind in KINDS:
            for a, b in PAIRS_C:
                ea, eb = E[kind][a], E[kind][b]
                dd = np.sqrt(bmean(ea, st, L)) - np.sqrt(bmean(eb, st, L)); lo, hi = np.percentile(dd, [2.5, 97.5])
                rows.append(dict(comparison=kind, first=a, second=b, n_days=nc, first_rmse=np.sqrt(ea.mean()),
                                 second_rmse=np.sqrt(eb.mean()), difference=np.sqrt(ea.mean()) - np.sqrt(eb.mean()),
                                 block_days=L, n_blocks=st.shape[1], B=B, ci_lo=lo, ci_hi=hi, width=hi - lo,
                                 excludes_zero=bool(lo > 0 or hi < 0)))
    c = pd.DataFrame(rows); key = ["comparison", "first", "second"]
    for k, g in c.groupby(key, sort=False):
        w = g["width"].to_numpy(); i = g.index
        ea, eb = E[k[0]][k[1]], E[k[0]][k[2]]; ac = acf(ea - eb, LAGS); sl = stabilised(L_GRID, w)
        c.loc[i, "width_ratio_to_L30"] = w / w[L_GRID.index(30)]
        c.loc[i, "width_ratio_to_prev"] = np.r_[np.nan, w[1:] / w[:-1]]
        c.loc[i, "excludes_zero_at_every_length"] = bool(g["excludes_zero"].all())
        for lag, v in zip(LAGS, ac):
            c.loc[i, f"acf_lag{lag}"] = v
        c.loc[i, "acf_white_band"] = 1.96 / np.sqrt(nc)
        c.loc[i, "stabilised_block_days"] = np.nan if sl is None else sl
        j = i[g["block_days"].to_numpy() == 30]
        c.loc[j, "A4_replay_ci_lo_B2000"], c.loc[j, "A4_replay_ci_hi_B2000"] = replay[k]
    per = c.groupby(key, sort=False)["stabilised_block_days"].first()
    c["stabilised_block_days_all_four"] = np.nan if per.isna().any() else per.max()
    c.to_csv(OUT / "A11_ordering_blocks.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(f"\n(1) Common-day cross-product contrasts, 3,040 common days, 95 % paired moving-block intervals, B = {B:,}")
    print(c[key + ["block_days", "n_blocks", "difference", "ci_lo", "ci_hi", "width", "width_ratio_to_L30", "width_ratio_to_prev",
                   "excludes_zero"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nACF of the paired daily difference (scored-day index; white band "
          f"+/-{1.96 / np.sqrt(nc):.3f}) and stabilised length (every longer width within 10 %)")
    for k, g in c.groupby(key, sort=False):
        s = g.iloc[0]; sl = s.stabilised_block_days
        g30 = g[g.block_days == 30].iloc[0]
        print(f"  {k[0]:24s} {k[1]:>4s} minus {k[2]:<4s} ACF " + " ".join(f"{s[f'acf_lag{l}']:+.3f}" for l in LAGS)
              + f" | stabilised {'not within grid' if np.isnan(sl) else int(sl)} | excludes zero at every length "
              + f"{bool(s.excludes_zero_at_every_length)} | grid L30 minus A4: d_lo {g30.ci_lo - g30.A4_replay_ci_lo_B2000:+.1e} "
              + f"d_hi {g30.ci_hi - g30.A4_replay_ci_hi_B2000:+.1e}")
    sall = c["stabilised_block_days_all_four"].iloc[0]
    print(f"  stabilised block length, all four: {'not within grid' if np.isnan(sall) else int(sall)} scored days")

    # ---- (2) Historical scalar table check at 7, 30, 90 scored days ----------------------------------------------------------------------
    s16 = []
    for pi, r in enumerate(RES):
        k = M[r]; n = int(k.sum()); y = Y[r][k]; e_ol = (OL[r][k] - y) ** 2; rm_ol = np.sqrt(e_ol.mean())
        for L in L_S16:
            st = starts(np.random.default_rng([16, pi, L]), n, L); ol_b = np.sqrt(bmean(e_ol, st, L))
            for src, S in ((SRC42, A[r][:1, k]), (SRC50, A[r][:, k])):
                e2 = ((S - y) ** 2).mean(0); rm = np.sqrt(e2.mean()); rm_b = np.sqrt(bmean(e2, st, L))
                pts = dict(rmse=rm, r=float(np.mean([np.corrcoef(a, y)[0, 1] for a in S])), reduction_pct=100 * (1 - rm / rm_ol))
                bs = dict(rmse=rm_b, r=boot_corr(S, y, st, L), reduction_pct=100 * (1 - rm_b / ol_b))
                for met in pts:
                    lo, hi = np.percentile(bs[met], [2.5, 97.5]); p, plo, phi = PRINTED_S16[r][met]; dp, di = DP[met]
                    s16.append(dict(source=src, product=r, n_days=n, block_days=L, n_blocks=st.shape[1], B=B, metric=met,
                                    point=pts[met], ci_lo=lo, ci_hi=hi, width=hi - lo, open_loop_rmse=rm_ol,
                                    printed_point=p, printed_ci_lo=plo, printed_ci_hi=phi,
                                    point_matches_printed=fmt(pts[met], dp) == fmt(p, dp),
                                    ci_matches_printed_at_printed_precision=(fmt(lo, di), fmt(hi, di)) == (fmt(plo, di), fmt(phi, di))
                                    if L == 7 else np.nan))
    t = pd.DataFrame(s16)
    for (src, r, met), g in t.groupby(["source", "product", "metric"], sort=False):
        w = g.set_index("block_days")["width"]; t.loc[g.index, "width_ratio_to_L7"] = g["width"].to_numpy() / w[7]
    for (src, L), g in t[t.metric == "rmse"].groupby(["source", "block_days"], sort=False):
        iv = g.set_index("product")[["ci_lo", "ci_hi"]]
        ov = lambda a, b: bool(iv.loc[a, "ci_lo"] <= iv.loc[b, "ci_hi"] and iv.loc[b, "ci_lo"] <= iv.loc[a, "ci_hi"])
        m = (t.source == src) & (t.block_days == L)
        t.loc[m, "rmse_ci_overlap_9km_1km"], t.loc[m, "rmse_ci_overlap_1km_400m"] = ov("9km", "1km"), ov("1km", "400m")
    bad = t[(t.source == SRC42) & (t.block_days == 7) & ~t.point_matches_printed.astype(bool)]
    assert bad.empty, f"seed-42 point values do not match printed Table S16:\n{bad}"
    t.to_csv(OUT / "A11_table_s16.csv", index=False)

    print(f"\n(2) Historical scalar Table S16 check, own observation days, 95 % moving-block intervals, B = {B:,}")
    print("  ASSERT PASSED: the seed-42 production run reproduces all nine printed point values")
    moved = t[(t.source == SRC50) & (t.block_days == 7) & ~t.point_matches_printed.astype(bool)]
    print("  at 50 seeds the printed digit moves for: " + ("; ".join(
        f"{q.product} {q.metric} {fmt(q.point, DP[q.metric][0])} (printed {fmt(q.printed_point, DP[q.metric][0])})"
        for q in moved.itertuples()) or "none"))
    for src in (SRC42, SRC50):
        print(f"  -- {src}")
        for L in L_S16:
            g = t[(t.source == src) & (t.block_days == L)]
            cells = []
            for r in RES:
                h = g[g["product"] == r].set_index("metric")
                cells.append(f"{r} RMSE {h.point['rmse']:.4f} [{h.ci_lo['rmse']:.4f}, {h.ci_hi['rmse']:.4f}] "
                             f"R {h.point['r']:.3f} [{h.ci_lo['r']:.3f}, {h.ci_hi['r']:.3f}] "
                             f"red {h.point['reduction_pct']:.1f} [{h.ci_lo['reduction_pct']:.1f}, {h.ci_hi['reduction_pct']:.1f}]")
            print(f"   L={L:>2d}: " + " | ".join(cells))
            print(f"         RMSE intervals overlap: 9 km/1 km {bool(g.rmse_ci_overlap_9km_1km.iloc[0])}; "
                  f"1 km/400 m {bool(g.rmse_ci_overlap_1km_400m.iloc[0])}; width ratio to L7 (RMSE) "
                  + ", ".join(f"{r} {g[(g['product'] == r) & (g.metric == 'rmse')].width_ratio_to_L7.iloc[0]:.2f}" for r in RES))
        h7 = t[(t.source == src) & (t.block_days == 7)]
        print("   7-day intervals reproduce the printed ones at printed precision: " + "; ".join(
            f"{q.product} {q.metric} {'yes' if q.ci_matches_printed_at_printed_precision else 'no'} "
            f"[{fmt(q.ci_lo, DP[q.metric][1])}, {fmt(q.ci_hi, DP[q.metric][1])}] vs [{fmt(q.printed_ci_lo, DP[q.metric][1])}, "
            f"{fmt(q.printed_ci_hi, DP[q.metric][1])}]" for q in h7.itertuples()))
    print("   (RMSE printed at 3 decimals cannot discriminate; R and reduction can)")

    # ---- (3) increments by window --------------------------------------------------------------------------
    win = {"2015-04-01 to 2024-12-31": np.ones(N, bool), "2015-04-01 to 2019-12-31": np.asarray(dates.year <= 2019),
           "2020-01-01 to 2024-12-31": np.asarray(dates.year >= 2020)}
    inc = []
    for r in RES:
        ia, io = A[r] - OL[r], Y[r] - OL[r]
        for wn, wm in win.items():
            for ds, dm in (("update days", M[r]), ("all days", np.ones(N, bool))):
                sel = wm & dm; base = dict(product=r, window=wn, day_set=ds)
                x = ia[0, sel]
                inc.append(dict(**base, source=SRC42, quantity="analysis minus open loop", n_days=int(sel.sum()),
                                mean=x.mean(), rms=np.sqrt((x ** 2).mean())))
                X = ia[:, sel]
                inc.append(dict(**base, source=SRC50, quantity="analysis minus open loop", n_days=int(sel.sum()),
                                mean=X.mean(), rms=np.sqrt((X ** 2).mean()), rms_of_seed_mean_series=np.sqrt((X.mean(0) ** 2).mean())))
                so = sel & np.isfinite(Y[r]); x = io[so]
                inc.append(dict(**base, source=SRCOBS, quantity="observation_corrected minus open loop", n_days=int(so.sum()),
                                mean=x.mean(), rms=np.sqrt((x ** 2).mean())))
    inc = pd.DataFrame(inc); inc.to_csv(OUT / "A11_increments_by_window.csv", index=False)
    print("\n(3) Increments (m3/m3): mean and RMS")
    print(inc.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))

    for f in ("A11_ordering_blocks.csv", "A11_table_s16.csv", "A11_increments_by_window.csv"):
        print("wrote", OUT / f)
    print(f"runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
