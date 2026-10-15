"""Derived scalar measurements for the schedule and sensitivity results.

R1 records production input-agreement scores, constructed-series sensitivity,
and a memoryless diagnostic. The finite-ensemble ratio is compared with mean
one-minus-gain; exact equality is not assumed. These controls show dependence
on filter settings and do not quantify independent retrieval accuracy.

R2 compares schedules across common, reduced-update, and fixed observation
calendars. The fixed-calendar score includes omitted update days. The corresponding
decomposition separates updated and skipped days. R3 applies that decomposition
to the thinning experiment and checks its pooled result against A2.

The bootstrap or seed summary attached to each output must be preserved when
interpreting its uncertainty. Cross-product normalizing gaps use different
corrected input targets and remain descriptive.

Author: Sebastian R.O. Marshall."""
import sys, pathlib, multiprocessing, numpy as np, pandas as pd
from joblib import Parallel, delayed

T = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(T / "src"))
from mekong_da.enkf import run_enkf
from mekong_da.config import EnKFConfig, OPTIMAL_CONFIG

OUT = pathlib.Path(__file__).resolve().parent.parent / "outputs" / "analyses"
RES = ["9km", "1km", "400m"]
SEEDS = list(range(42, 92))
FULL_CORES = multiprocessing.cpu_count()


def cfg(**kw):
    return EnKFConfig(**{**OPTIMAL_CONFIG.__dict__, **kw})


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ac1(e):
    e = np.asarray(e, float)
    if e.std() == 0:
        return np.nan          # a pure bias has no variance, so AC1 is undefined
    return float(np.corrcoef(e[:-1], e[1:])[0, 1])


def ensemble(ol, y, mask, seeds=SEEDS, **kw):
    """Analysis series for every seed, stacked (n_seeds, n_days)."""
    return np.array(Parallel(n_jobs=-1)(
        delayed(lambda s: run_enkf(ol, y, mask, config=cfg(seed=s, **kw)).analysis)(s)
        for s in seeds))


def per_seed_rmse(A, y, evalmask):
    """RMSE per seed on `evalmask`, so arms can be differenced seed by seed."""
    return np.sqrt(((A[:, evalmask] - y[evalmask]) ** 2).mean(axis=1))


def ci95(x):
    x = np.asarray(x, float)
    return 1.959963985 * x.std(ddof=1) / np.sqrt(len(x))


inp = {r: pd.read_csv(T / "data/enkf" / f"inputs_{r}.csv", parse_dates=["date"]) for r in RES}
ref = {r: pd.read_csv(T / "data/enkf" / f"reference_{r}.csv", parse_dates=["date"]) for r in RES}
M = {r: inp[r]["obs_available"].to_numpy(bool) for r in RES}
OL = {r: inp[r]["baseline"].to_numpy(float) for r in RES}
Y = {r: inp[r]["observation_corrected"].to_numpy(float) for r in RES}
COMMON = M["9km"] & M["1km"] & M["400m"]
LO, HI = OPTIMAL_CONFIG.sm_bounds
N = len(OL["9km"])

print(f"cores {FULL_CORES}   days {N}   common evaluation days {int(COMMON.sum())}")
print(f"open loop bit-identical across products: "
      f"{np.abs(OL['9km'] - OL['1km']).max():.1e}, {np.abs(OL['9km'] - OL['400m']).max():.1e}")

# ----------------------------------------------------------------- R1 published
rows = []
for r in RES:
    m = M[r]
    e = (Y[r] - OL[r])[m]
    ol_r = rmse(OL[r][m], Y[r][m])
    post = rmse(ref[r]["analysis"].to_numpy(float)[m], Y[r][m])
    rows.append(dict(product=r, n_obs=int(m.sum()),
                     coverage_pct=100 * m.sum() / N,
                     rmse_open_loop=ol_r, rmse_post_da=post,
                     ratio=post / ol_r, reduction_pct=100 * (1 - post / ol_r),
                     sd_disagreement=float(e.std()), ac1_disagreement=ac1(e),
                     basis="each product's own observation days, archived analysis series"))
pub = pd.DataFrame(rows)
pub.to_csv(OUT / "R1_published.csv", index=False)
print("\nR1_published.csv")
print(pub[["product", "n_obs", "rmse_open_loop", "rmse_post_da", "ratio",
           "reduction_pct", "ac1_disagreement"]].to_string(index=False))

# ------------------------------------------------------- R1 ratio controls
def control(e, tag, kind, seeds=SEEDS[:20], **kw):
    """Run the real filter on a CONSTRUCTED observation series ol + e, on the 9 km mask."""
    ol, mk = OL["9km"], M["9km"]
    y = np.where(np.isfinite(Y["9km"]), np.clip(ol + e, LO + 1e-6, HI - 1e-6), np.nan)
    A = ensemble(ol, y, mk, seeds=seeds, **kw)
    ol_r = rmse(ol[mk], y[mk])
    post = float(per_seed_rmse(A, y, mk).mean())
    ee = (y - ol)[mk]
    return dict(series=tag, kind=kind, sd_disagreement=float(ee.std()),
                ac1_disagreement=ac1(ee), rmse_open_loop=ol_r,
                rmse_post_da=post, ratio=post / ol_r, n_seeds=len(seeds), **kw)


rng = np.random.default_rng(7)
e_real = {r: (Y[r] - OL[r]) for r in RES}
ctl = []
for sd in (0.0475, 0.095):
    ctl.append(control(rng.normal(0, sd, N), f"white noise, sd {sd}", "white noise"))
for rho in (0.3, 0.6, 0.9, 0.98):
    w = rng.normal(0, 1, N)
    e = np.zeros(N)
    for t in range(1, N):
        e[t] = rho * e[t - 1] + w[t]
    e = 0.0475 * e / e.std()
    ctl.append(control(e, f"AR(1) rho {rho}", "AR(1)"))
ctl.append(control(np.full(N, 0.0475), "constant offset 0.0475", "pure bias"))
for f in (0.5, 1.0, 1.5):
    ctl.append(control(f * e_real["9km"], f"real 9 km disagreement x {f}", "real, rescaled"))
ctl = pd.DataFrame(ctl)
ctl.to_csv(OUT / "R1_ratio_controls.csv", index=False)
print("\nR1_ratio_controls.csv")
print(ctl[["series", "kind", "sd_disagreement", "ac1_disagreement",
           "rmse_open_loop", "rmse_post_da", "ratio"]].to_string(index=False))

# Compare the finite-ensemble memoryless ratio with mean one-minus-gain.
mem = []
for r in RES:
    A = ensemble(OL[r], Y[r], M[r], persistence_phi=0.0)
    post = float(per_seed_rmse(A, Y[r], M[r]).mean())
    ol_r = rmse(OL[r][M[r]], Y[r][M[r]])
    K = np.array([run_enkf(OL[r], Y[r], M[r], config=cfg(seed=s, persistence_phi=0.0)).kalman_gain
                  for s in SEEDS[:10]])
    one_minus_k = float(np.nanmean(1.0 - K))
    mem.append(dict(product=r, phi=0.0, rmse_open_loop=ol_r, rmse_post_da=post,
                    ratio=post / ol_r, one_minus_kalman_gain=one_minus_k))
mem = pd.DataFrame(mem)
mem.to_csv(OUT / "R1_memoryless_limit.csv", index=False)
print("\nR1_memoryless_limit.csv  (phi = 0; compare the finite-ensemble ratio with mean 1 - K)")
print(mem.to_string(index=False))

# ----------------------------------------------------------------- R2 eval sets
ARM = {"its own dates": M["9km"],
       "the 1 km dates": M["9km"] & M["1km"],
       "the 400 m dates": M["9km"] & M["400m"]}
A_arm = {a: ensemble(OL["9km"], Y["9km"], mk) for a, mk in ARM.items()}

# the denominator: the PUBLISHED own-day gap, one denominator for all three sets, at the
# same 50 seeds as the numerators so the two are commensurable
own50 = {}
for r in RES:
    own50[r] = per_seed_rmse(ensemble(OL[r], Y[r], M[r]), Y[r], M[r])
gap = {"the 1 km dates": float(own50["1km"].mean() - own50["9km"].mean()),
       "the 400 m dates": float(own50["400m"].mean() - own50["9km"].mean())}
print(f"\npublished own-day gaps, 50 seeds: 9 km to 1 km {gap['the 1 km dates']:+.6f}, "
      f"9 km to 400 m {gap['the 400 m dates']:+.6f}")

SETS = {"the 3,040 days all three observe": lambda a: COMMON,
        "the 9 km calendar, 3,447 days": lambda a: M["9km"],
        "each arm's own update days": lambda a: ARM[a]}
rows = []
for sname, f in SETS.items():
    base = per_seed_rmse(A_arm["its own dates"], Y["9km"], f("its own dates"))
    for a in ARM:
        v = per_seed_rmse(A_arm[a], Y["9km"], f(a))
        d = v - base
        pct = 100 * d / gap[a] if a in gap else np.zeros_like(d)
        rows.append(dict(eval_set=sname, arm=a, n_scored=int(f(a).sum()),
                         n_updates=int(ARM[a].sum()),
                         rmse_mean=float(v.mean()), rmse_sd=float(v.std(ddof=1)),
                         delta_mean=float(d.mean()), delta_ci95=float(ci95(d)),
                         pct_of_published_gap=float(pct.mean()),
                         pct_ci95=float(ci95(pct)) if a in gap else 0.0,
                         published_gap=gap.get(a, np.nan), n_seeds=len(SEEDS)))
ev = pd.DataFrame(rows)
ev.to_csv(OUT / "R2_evalsets.csv", index=False)
print("\nR2_evalsets.csv")
print(ev[["eval_set", "arm", "n_scored", "rmse_mean", "delta_mean", "delta_ci95",
          "pct_of_published_gap", "pct_ci95"]].to_string(index=False))

# Separate fixed-calendar squared errors on updated and skipped days.
rows = []
for a in ("the 1 km dates", "the 400 m dates"):
    ev_m, mk = M["9km"], ARM[a]
    upd, sk = ev_m & mk, ev_m & ~mk
    A = A_arm[a]
    r_u = float(per_seed_rmse(A, Y["9km"], upd).mean())
    r_s = float(per_seed_rmse(A, Y["9km"], sk).mean())
    r_all = float(per_seed_rmse(A, Y["9km"], ev_m).mean())
    n_u, n_s = int(upd.sum()), int(sk.sum())
    rows.append(dict(arm=a, n_updated=n_u, n_skipped=n_s,
                     rmse_updated=r_u, rmse_skipped=r_s,
                     rmse_open_loop_skipped=rmse(OL["9km"][sk], Y["9km"][sk]),
                     rmse_open_loop_all=rmse(OL["9km"][ev_m], Y["9km"][ev_m]),
                     rmse_pooled=r_all,
                     pooled_check=float(np.sqrt((n_u * r_u ** 2 + n_s * r_s ** 2) / (n_u + n_s))),
                     pct_days_skipped=100 * n_s / (n_u + n_s),
                     pct_sqerr_from_skipped=100 * n_s * r_s ** 2 / (n_u * r_u ** 2 + n_s * r_s ** 2)))
dec = pd.DataFrame(rows)
dec.to_csv(OUT / "R2_decomposition.csv", index=False)
print("\nR2_decomposition.csv")
print(dec.to_string(index=False))

# ------------------------------------------------------------ R3 decomposition
a2 = pd.read_csv(OUT / "A2_dose_response.csv")
KEEP = sorted(a2[a2.pattern == "regular"].keep_every.unique().tolist())
full, avail = M["9km"], np.flatnonzero(M["9km"])
rows = []
for k in KEEP:
    mk = np.zeros_like(full)
    mk[avail[::k]] = True
    A = ensemble(OL["9km"], Y["9km"], mk)
    e_up, e_sk = COMMON & mk, COMMON & ~mk
    n_u, n_s = int(e_up.sum()), int(e_sk.sum())
    r_all = float(per_seed_rmse(A, Y["9km"], COMMON).mean())
    r_u = float(per_seed_rmse(A, Y["9km"], e_up).mean()) if n_u else np.nan
    r_s = float(per_seed_rmse(A, Y["9km"], e_sk).mean()) if n_s else np.nan
    rows.append(dict(keep_every=k, n_updates=int(mk.sum()),
                     mean_gap_d=float(np.diff(np.flatnonzero(mk)).mean()),
                     coverage_pct=100 * mk.sum() / N,
                     n_eval_updated=n_u, n_eval_skipped=n_s,
                     frac_eval_never_assimilated=n_s / (n_u + n_s),
                     rmse_pooled=r_all, rmse_updated=r_u, rmse_skipped=r_s,
                     rmse_open_loop=rmse(OL["9km"][COMMON], Y["9km"][COMMON])))
d3 = pd.DataFrame(rows)
# the pooled column must reproduce the shipped A2 curve, else the two are not the same runs
merged = d3.merge(a2[a2.pattern == "regular"][["keep_every", "rmse_mean"]], on="keep_every")
worst = float((merged.rmse_pooled - merged.rmse_mean).abs().max())
assert worst < 3e-4, f"pooled column disagrees with A2_dose_response.csv by {worst:.2e}"
print(f"\nR3 pooled column agrees with A2_dose_response.csv to {worst:.2e}")
d3.to_csv(OUT / "R3_decomposition.csv", index=False)
print("R3_decomposition.csv")
print(d3.to_string(index=False))

print("\nDONE")
