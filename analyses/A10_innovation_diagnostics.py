#!/usr/bin/env python3
"""Analysis A10. Innovation autocorrelation under two pairing conventions.

True innovations are observations minus forecast means on update days. An
update-index lag pairs entries k updates apart; a calendar-day lag pairs valid
innovations k calendar days apart. Pearson correlation summarizes each pair set.
The script reports lags 1 to 16 and selected-lag distributions across 50 seeds.

Ljung-Box statistics use the update-index series, lag degrees of freedom, and no
fitted-parameter correction. They are diagnostics under the stated sampling and
stationarity assumptions, rather than proof of independent observation errors.

The inflation-persistence sweep uses the archived grid and all available update
days from April 2015 through December 2024. The tuning implementation's validation
date restricts its validation metrics, but not innovation autocorrelation. A
through-2020 subset provides a separate period comparison for two configurations.
Checks reproduce A6 lag one and the seed-42 member of the ensemble.

Author: Sebastian R.O. Marshall."""
import sys
import pathlib
import multiprocessing as mp
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from A_temporal_subsampling import load, RES, SEEDS, OUT        # noqa: E402  (also puts src/ on sys.path)
from mekong_da.enkf import run_enkf                               # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG, DEFAULT_CONFIG  # noqa: E402
from mekong_da.tuning import sweep, INFLATION_GRID, PHI_GRID     # noqa: E402

LEGACY = pathlib.Path(__file__).resolve().parents[1] / "data" / "enkf_tuning"
FALLBACK_GRID = ((1.00, 1.02, 1.05, 1.10), (0.30, 0.50, 0.70, 0.90))  # used only if the legacy CSV is unreachable
A6_AC1 = {"9km": 0.0450, "1km": 0.0666, "400m": 0.1319}
PAIRINGS = ("update_index", "calendar_day")
LAGS = range(1, 17)
SEED_LAGS = (1, 3, 5, 8)
LB_LAGS = (8, 16)
FLAGS = {"legacy_default": (1.05, 0.70), "adopted": (1.02, 0.50)}
FULL_CORES = mp.cpu_count()


def acf(d, k, pairing):
    """Pearson lag-k autocorrelation of a daily innovation array d (NaN off update days); returns (r, n_pairs)."""
    if pairing == "update_index":
        x = d[np.isfinite(d)]
        a, b = x[:-k], x[k:]
    else:
        a, b = d[:-k], d[k:]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
    return float(np.corrcoef(a, b)[0, 1]), int(a.size)


def arrays(frame):
    return (frame["baseline"].to_numpy(float), frame["observation_corrected"].to_numpy(float),
            frame["obs_available"].to_numpy(bool))


def innovation(ol, y, mask, seed=42, cfg=OPTIMAL_CONFIG):
    return run_enkf(ol, y, mask, config=EnKFConfig(**{**cfg.to_dict(), "seed": seed})).innovation


def _seed_job(args):
    r, s, ol, y, m = args
    d = innovation(ol, y, m, seed=s)
    return r, s, {(p, k): acf(d, k, p)[0] for p in PAIRINGS for k in SEED_LAGS}


def _sweep_job(args):
    r, ol, y, grid = args
    t = sweep(ol, y, DEFAULT_CONFIG, {"inflation_factor": grid[0], "persistence_phi": grid[1]}, verbose=False)
    t.insert(0, "product", r)
    return t


def main():
    d, dates = load()
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    a6 = pd.read_csv(OUT / "A6_filter_consistency.csv").set_index("product")
    print(f"A10 | products {RES} | seeds {SEEDS[0]}..{SEEDS[-1]} ({len(SEEDS)}) | cores {FULL_CORES}")
    print(f"published configuration: {OPTIMAL_CONFIG}")

    # ---------------- (a) production run, seed 42 ----------------
    rows, prod_inn = [], {}
    for r in RES:
        ol, y, m = arrays(d[r])
        inn = innovation(ol, y, m)
        prod_inn[r] = inn
        n_upd = int(np.isfinite(inn).sum())
        base = dict(product=r, ensemble="seed42", sd_across_seeds=np.nan, n_update_days=n_upd)
        print(f"\n[{r}] update days {n_upd}, first {dates[np.isfinite(inn)][0].date()}, last {dates[np.isfinite(inn)][-1].date()}")
        print("  lag  update_index (n)      calendar_day (n)")
        tab = {}
        for p in PAIRINGS:
            for k in LAGS:
                v, n = acf(inn, k, p)
                tab[(p, k)] = (v, n)
                rows.append({**base, "statistic": "acf", "pairing": p, "lag": k, "value": v, "n_pairs": n})
        for k in LAGS:
            (u, nu), (c, nc) = tab[("update_index", k)], tab[("calendar_day", k)]
            print(f"  {k:>3d}  {u:+.4f} ({nu:>4d})       {c:+.4f} ({nc:>4d})")
        lb = acorr_ljungbox(inn[np.isfinite(inn)], lags=list(LB_LAGS))
        for k in LB_LAGS:
            q, pv = float(lb.loc[k, "lb_stat"]), float(lb.loc[k, "lb_pvalue"])
            rows.append({**base, "statistic": "ljung_box_Q", "pairing": "update_index", "lag": k, "value": q, "n_pairs": n_upd})
            rows.append({**base, "statistic": "ljung_box_p", "pairing": "update_index", "lag": k, "value": pv, "n_pairs": n_upd})
            print(f"  Ljung-Box lag {k:>2d}: Q = {q:.1f}, p = {pv:.3g} (df = {k})")
        ac1 = tab[("update_index", 1)][0]
        assert abs(ac1 - A6_AC1[r]) < 1e-3 and abs(ac1 - a6.loc[r, "innovation_ac1"]) < 1e-3, (r, ac1)
        print(f"  CHECK lag-1 update_index {ac1:+.4f} reproduces A6_filter_consistency.csv {a6.loc[r, 'innovation_ac1']:+.4f}: PASS")

    # ---------------- legacy grid ----------------
    s1b = LEGACY / "stage1b_inflation_phi.csv"
    if s1b.is_file():
        leg = pd.read_csv(s1b)
        grid = (tuple(float(v) for v in sorted(leg["inflation_factor"].unique())),
                tuple(float(v) for v in sorted(leg["persistence_phi"].unique())))
        grid_source = str(s1b)
    else:
        leg, grid, grid_source = None, FALLBACK_GRID, "fallback grid (legacy stage1b CSV unreachable)"
    print(f"\ngrid source: {grid_source}\n  inflation {grid[0]}\n  phi       {grid[1]}")
    print(f"  equals tuning.INFLATION_GRID x tuning.PHI_GRID: {grid == (tuple(INFLATION_GRID), tuple(PHI_GRID))}")

    # ---------------- 50 seeds and the sweep, on all cores ----------------
    seed_jobs = [(r, s, *arrays(d[r])) for r in RES for s in SEEDS]
    ser = {}
    for r in RES:
        ol, y, m = arrays(d[r])
        ser[r] = (pd.Series(ol, index=dates), pd.Series(np.where(m, y, np.nan), index=dates))
    sweep_jobs = [(r, *ser[r], grid) for r in RES]
    with mp.Pool(min(FULL_CORES, len(seed_jobs))) as pool:
        seed_out = pool.map(_seed_job, seed_jobs)
        tun = pd.concat(pool.map(_sweep_job, sweep_jobs), ignore_index=True)

    print("\n50-seed ensemble (mean +/- sd), published configuration")
    for r in RES:
        got = {s: v for rr, s, v in seed_out if rr == r}
        assert len(got) == len(SEEDS)
        for p in PAIRINGS:
            for k in SEED_LAGS:
                assert abs(got[42][(p, k)] - acf(prod_inn[r], k, p)[0]) < 1e-12
                vals = np.array([got[s][(p, k)] for s in SEEDS])
                rows.append(dict(product=r, statistic="acf", pairing=p, lag=k, ensemble=f"seeds{SEEDS[0]}_{SEEDS[-1]}",
                                 value=float(vals.mean()), sd_across_seeds=float(vals.std(ddof=1)),
                                 n_pairs=acf(prod_inn[r], k, p)[1], n_update_days=int(np.isfinite(prod_inn[r]).sum())))
            print(f"  [{r}] {p:<12s} " + "  ".join(
                f"lag{k} {np.mean([got[s][(p, k)] for s in SEEDS]):+.3f}+/-{np.std([got[s][(p, k)] for s in SEEDS], ddof=1):.3f}"
                for k in SEED_LAGS))
    acf_df = pd.DataFrame(rows)[["product", "statistic", "pairing", "lag", "ensemble", "value", "sd_across_seeds", "n_pairs", "n_update_days"]]
    acf_df.to_csv(OUT / "A10_innovation_acf.csv", index=False)
    print("wrote", OUT / "A10_innovation_acf.csv", len(acf_df), "rows")

    # ---------------- (b) sweep table ----------------
    def flag(row):
        for lab, (inf, phi) in FLAGS.items():
            if np.isclose(row.inflation_factor, inf) and np.isclose(row.persistence_phi, phi):
                return lab
        return "grid"
    tun["config_label"] = tun.apply(flag, axis=1)
    tun["in_legacy_panel_a"] = (tun["product"] == "9km") & (np.isclose(tun["inflation_factor"], 1.05) | np.isclose(tun["inflation_factor"], 1.02))
    tun["grid_source"] = grid_source
    tun["legacy_archived_ac1"] = np.nan
    tun["legacy_archived_file"] = ""
    if leg is not None:
        for i, row in tun[tun["product"] == "9km"].iterrows():
            hit = leg[np.isclose(leg["inflation_factor"], row.inflation_factor) & np.isclose(leg["persistence_phi"], row.persistence_phi)]
            if len(hit):
                tun.loc[i, ["legacy_archived_ac1", "legacy_archived_file"]] = [float(hit["innov_autocorr_1"].iloc[0]), "stage1b_inflation_phi.csv"]
        val = pd.read_csv(LEGACY / "validation_default_vs_optimal.csv")
        for i, row in tun[(tun["product"] != "9km") & (tun["config_label"] != "grid")].iterrows():
            hit = val[(val["resolution"] == row["product"]) & (val["config"] == {"legacy_default": "DEFAULT", "adopted": "OPTIMAL"}[row.config_label])]
            tun.loc[i, ["legacy_archived_ac1", "legacy_archived_file"]] = [float(hit["innov_autocorr_1"].iloc[0]), "validation_default_vs_optimal.csv"]

    tun["n_update_days"] = 0
    tun["ac1_sample_first_day"] = ""
    tun["ac1_sample_last_day"] = ""
    tun["ac1_update_days_through_2020"] = np.nan
    for r in RES:
        ol, y = ser[r]
        upd = dates[y.notna().to_numpy()]
        sel = tun["product"] == r
        tun.loc[sel, ["n_update_days", "ac1_sample_first_day", "ac1_sample_last_day"]] = [len(upd), str(upd[0].date()), str(upd[-1].date())]
        for lab, (inf, phi) in FLAGS.items():
            cfg = EnKFConfig(**{**DEFAULT_CONFIG.to_dict(), "inflation_factor": inf, "persistence_phi": phi})
            inn = run_enkf(ol.to_numpy(float), y.to_numpy(float), None, config=cfg).innovation
            i = tun.index[sel & (tun["config_label"] == lab)][0]
            full = acf(inn, 1, "update_index")[0]
            assert abs(full - tun.loc[i, "innov_autocorr_1"]) < 1e-12, (r, lab)
            tun.loc[i, "ac1_update_days_through_2020"] = acf(np.where(dates <= pd.Timestamp("2020-12-31"), inn, np.nan), 1, "update_index")[0]

    keep = ["product", "inflation_factor", "persistence_phi", "config_label", "in_legacy_panel_a", "innov_autocorr_1",
            "legacy_archived_ac1", "legacy_archived_file", "ac1_update_days_through_2020", "n_update_days",
            "ac1_sample_first_day", "ac1_sample_last_day", "mean_K", "innov_var_ratio", "nse_da", "kge_da",
            "obs_error_std", "model_error_std", "grid_source"]
    tun = tun[keep]
    tun.to_csv(OUT / "A10_tuning_ac1.csv", index=False)

    print("\nSweep AC1 (tuning.evaluate_config, full record of update days), legacy default vs adopted")
    for r in RES:
        t = tun[tun["product"] == r].set_index("config_label")
        ld, ad = t.loc["legacy_default"], t.loc["adopted"]
        print(f"  [{r}] legacy default (1.05, 0.70) {ld.innov_autocorr_1:+.3f} (archived {ld.legacy_archived_ac1:+.3f})  |  "
              f"adopted (1.02, 0.50) {ad.innov_autocorr_1:+.3f} (archived {ad.legacy_archived_ac1:+.3f})  |  "
              f"through-2020 subset: {ld.ac1_update_days_through_2020:+.3f} / {ad.ac1_update_days_through_2020:+.3f}  |  "
              f"sample {ld.ac1_sample_first_day} to {ld.ac1_sample_last_day}, n = {ld.n_update_days}")
        g = tun[tun["product"] == r]
        best = g.loc[g["innov_autocorr_1"].abs().idxmin()]
        print(f"         grid argmin |AC1|: ({best.inflation_factor}, {best.persistence_phi}) AC1 {best.innov_autocorr_1:+.3f}")
    print("\n9 km grid (rows = inflation, columns = phi), archived AC1:")
    print(tun[tun["product"] == "9km"].pivot(index="inflation_factor", columns="persistence_phi", values="innov_autocorr_1").round(3).to_string())
    print("wrote", OUT / "A10_tuning_ac1.csv", len(tun), "rows")


if __name__ == "__main__":
    main()
