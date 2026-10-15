#!/usr/bin/env python3
"""Analysis A8. Schedule contrasts across five scalar persistence settings.

Each setting uses the same 50 seeds and the same 9 km observation values.
The full and reduced calendars are compared on common days, reduced update
days, and the fixed 3,447-day 9 km calendar. The fixed calendar retains direct
errors on omitted update days. Common-day comparisons retain history and
selection effects but exclude missed-day error outside their selected sample.

Paired intervals use 2,000 moving-block draws of 30 scored days after averaging
daily squared errors across seeds. Each comparison resets the stated random
stream, matching A3 and A4 at persistence 0.50. Fixed-calendar scores and
normalizing gaps follow R2. Mean gain is recorded on update and common days.

Checks reproduce A3, A4, and R2 within the stated tolerances. A changed-persistence
negative control confirms that the comparison detects a different configuration.
All scores concern input agreement conditional on these settings and targets.

Author: Sebastian R.O. Marshall."""
import sys
sys.dont_write_bytecode = True            # touch no file beside the CSV, not even __pycache__
import os, time, pathlib, multiprocessing as mp, numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from A_temporal_subsampling import load, TOOL, SEEDS, RES   # noqa: E402
from A4_ordering_and_ownday import boot_diff, B, L          # noqa: E402  the A3/A4 bootstrap, verbatim
sys.path.insert(0, str(TOOL / "src"))
from mekong_da.enkf import run_enkf                         # noqa: E402
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG     # noqa: E402

OUT = HERE.parent / "outputs" / "analyses"
PHIS = (0.00, 0.30, 0.50, 0.70, 0.90)
TOL = 2e-6


def run(task):
    """One filter run: per-day squared error against the product's corrected series, and the gain."""
    phi, seed, ol, y, mask = task
    r = run_enkf(ol, y, mask, config=EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, "seed": seed, "persistence_phi": phi}))
    assert r.n_updates == int(mask.sum())
    return (r.analysis - y) ** 2, r.kalman_gain


def boots(pairs):
    """A3 and A4 each start one stream at 42 and draw the 1 km schedule, then the 400 m schedule."""
    rng = np.random.default_rng(42)
    return [boot_diff(a, b, rng) for a, b in pairs]


def main():
    t0 = time.time(); n_cores = os.cpu_count()
    d, _ = load()
    m = {r: d[r]["obs_available"].to_numpy(bool) for r in RES}
    OL = {r: d[r]["baseline"].to_numpy(float) for r in RES}
    Y = {r: d[r]["observation_corrected"].to_numpy(float) for r in RES}
    m9, common = m["9km"], m["9km"] & m["1km"] & m["400m"]
    SCHED = {"the 1 km dates": ("1km", m9 & m["1km"]), "the 400 m dates": ("400m", m9 & m["400m"])}
    ARMS = {"full": ("9km", m9), "the 1 km dates": ("9km", SCHED["the 1 km dates"][1]),
            "the 400 m dates": ("9km", SCHED["the 400 m dates"][1]), "own 1km": ("1km", m["1km"]),
            "own 400m": ("400m", m["400m"])}
    print(f"A8 schedule x persistence: cores {n_cores}, seeds {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)}), phi {PHIS}, "
          f"days {len(m9)}, 9 km calendar {int(m9.sum())}, common {int(common.sum())}, "
          f"1 km schedule {int(SCHED['the 1 km dates'][1].sum())}, 400 m schedule {int(SCHED['the 400 m dates'][1].sum())}")

    keys = [(phi, a, s) for phi in PHIS for a in ARMS for s in SEEDS]
    tasks = ((phi, s, OL[ARMS[a][0]], Y[ARMS[a][0]], ARMS[a][1]) for phi, a, s in keys)
    E = {}; P = {}; G = {}; GC = {}
    with mp.Pool(n_cores) as pool:
        for i, ((phi, a, s), (e2, K)) in enumerate(zip(keys, pool.imap(run, tasks, chunksize=2)), 1):
            k = (phi, a); mk = ARMS[a][1]
            E[k] = E.get(k, 0.0) + e2                     # seed order 42..91, as A3 and A4 accumulate
            P.setdefault(k, []).append(np.sqrt(e2[mk].mean()))
            G.setdefault(k, []).append(np.nanmean(K[mk])); GC.setdefault(k, []).append(np.nanmean(K[common]))
            if i % 250 == 0:
                print(f"  filter runs {i}/{len(keys)}  {time.time() - t0:.0f} s", flush=True)
        E = {k: v / len(SEEDS) for k, v in E.items()}
        SETS = {"fixed": lambda mk: m9, "common": lambda mk: common, "own": lambda mk: mk}
        jobs = [(phi, st) for phi in PHIS for st in SETS]
        out = pool.map(boots, [[(E[(phi, sch)][SETS[st](mk)], E[(phi, "full")][SETS[st](mk)])
                                for sch, (_, mk) in SCHED.items()] for phi, st in jobs])
    BT = {(phi, st, sch): r for (phi, st), res in zip(jobs, out) for sch, r in zip(SCHED, res)}

    rows = []
    for phi in PHIS:
        ef = E[(phi, "full")]
        for sch, (prod, mk) in SCHED.items():
            ed = E[(phi, sch)]
            r9, ro = float(np.mean(P[(phi, "full")])), float(np.mean(P[(phi, f"own {prod}")]))
            gap = ro - r9
            pct = lambda x: 100.0 * x / gap
            row = dict(phi=phi, schedule=sch, n_seeds=len(SEEDS), n_updates_full=int(m9.sum()), n_updates_degraded=int(mk.sum()))
            for st, ev in (("fixed", m9), ("common", common)):
                delta, lo, hi = BT[(phi, st, sch)]
                row.update({f"{st}_n_days": int(ev.sum()), f"{st}_rmse_full": np.sqrt(ef[ev].mean()),
                            f"{st}_rmse_degraded": np.sqrt(ed[ev].mean()), f"{st}_diff": delta, f"{st}_ci_lo": lo,
                            f"{st}_ci_hi": hi, f"{st}_excludes_zero": bool(lo > 0 or hi < 0), f"{st}_pct_of_gap": pct(delta),
                            f"{st}_pct_lo": min(pct(lo), pct(hi)), f"{st}_pct_hi": max(pct(lo), pct(hi))})
            full_all, full_on_deg, deg = np.sqrt(ef[m9].mean()), np.sqrt(ef[mk].mean()), np.sqrt(ed[mk].mean())
            delta, lo, hi = BT[(phi, "own", sch)]
            row.update(own_n_days_full=int(m9.sum()), own_n_days_degraded=int(mk.sum()), own_rmse_full=full_all,
                       own_rmse_full_on_degraded_days=full_on_deg, own_rmse_degraded=deg, own_total=deg - full_all,
                       own_part_dayset=full_on_deg - full_all, own_part_schedule=delta, own_schedule_ci_lo=lo,
                       own_schedule_ci_hi=hi, own_schedule_excludes_zero=bool(lo > 0 or hi < 0),
                       own_pct_total=pct(deg - full_all), own_pct_dayset=pct(full_on_deg - full_all), own_pct_schedule=pct(delta),
                       own_pct_schedule_lo=min(pct(lo), pct(hi)), own_pct_schedule_hi=max(pct(lo), pct(hi)),
                       gap_product=prod, own_rmse_9km_product=r9, own_rmse_other_product=ro, gap_other_minus_9km=gap,
                       gain_full_update_days=float(np.mean(G[(phi, "full")])), gain_degraded_update_days=float(np.mean(G[(phi, sch)])),
                       gain_full_common_days=float(np.mean(GC[(phi, "full")])), gain_degraded_common_days=float(np.mean(GC[(phi, sch)])),
                       block_days=L, B=B)
            rows.append(row)
    df = pd.DataFrame(rows)
    num = df.select_dtypes("number")
    assert np.isfinite(num.to_numpy(float)).all(), "non-finite value in the table"

    print("\nphi   schedule         common diff [95% block]            excl0  %gap  | fixed rise  gap      | own: dayset  schedule [95% block]              | gain upd full/deg  common full/deg")
    for r in df.itertuples():
        print(f"{r.phi:.2f} {r.schedule:<16s} {r.common_diff:+.2e} [{r.common_ci_lo:+.2e}, {r.common_ci_hi:+.2e}] {str(r.common_excludes_zero):>5s} "
              f"{r.common_pct_of_gap:+6.1f} | {r.fixed_diff:+.5f} {r.gap_other_minus_9km:.5f} | {r.own_part_dayset:+.2e} "
              f"{r.own_part_schedule:+.2e} [{r.own_schedule_ci_lo:+.2e}, {r.own_schedule_ci_hi:+.2e}] {str(r.own_schedule_excludes_zero):>5s} | "
              f"{r.gain_full_update_days:.4f}/{r.gain_degraded_update_days:.4f}  {r.gain_full_common_days:.4f}/{r.gain_degraded_common_days:.4f}")

    # ---- gates at the published phi, against the shipped CSVs
    a3 = pd.read_csv(OUT / "A3_daylevel_interval.csv").set_index("schedule")
    r2 = pd.read_csv(OUT / "R2_evalsets.csv"); r2 = r2[r2.eval_set == "the 9 km calendar, 3,447 days"].set_index("arm")
    a4 = pd.read_csv(OUT / "A4_ordering_and_ownday.csv"); a4 = a4[a4.block == "own-day decomposition"].set_index("schedule")
    p5 = df[np.isclose(df.phi, 0.50)].set_index("schedule")
    print(f"\ngates at phi 0.50, tolerance {TOL:.0e}:")
    for sch in SCHED:
        for label, got, want in (("common-day diff vs A3", p5.at[sch, "common_diff"], a3.at[sch, "delta"]),
                                 ("fixed-calendar diff vs R2", p5.at[sch, "fixed_diff"], r2.at[sch, "delta_mean"]),
                                 ("own-day schedule part vs A4", p5.at[sch, "own_part_schedule"], a4.at[sch, "part_schedule"]),
                                 ("own-day gap vs R2", p5.at[sch, "gap_other_minus_9km"], r2.at[sch, "published_gap"])):
            print(f"  {sch:<16s} {label:<28s} got {got:+.6e} want {want:+.6e} |diff| {abs(got - want):.1e}")
            assert abs(got - want) < TOL, f"{sch} {label}: {got} vs {want}"
        ctl = abs(df[np.isclose(df.phi, 0.70) & (df.schedule == sch)].common_diff.iloc[0] - a3.at[sch, "delta"])
        print(f"  {sch:<16s} negative control: phi 0.70 common diff is {ctl:.1e} from A3, gate fires: {ctl >= TOL}")
        assert ctl >= TOL, "negative control failed: the gate cannot tell phi 0.70 from phi 0.50"
    df.to_csv(OUT / "A8_schedule_persistence.csv", index=False)
    print(f"\nwrote {OUT / 'A8_schedule_persistence.csv'} ({len(df)} rows, {df.shape[1]} columns); runtime {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
