#!/usr/bin/env python3
"""A worked example of the soil-moisture experiment, start to finish, in a few seconds.

Run it after installing the package:

    python examples/quickstart.py

It filters one resolution, checks the result against the published series, prints the skill
table, and then shows a small experiment of its own: what happens to the filter when the
observation is trusted more or less.

Author: Sebastian R.O. Marshall.
"""

import numpy as np
import pandas as pd

from mekong_da import EnKFConfig, OPTIMAL_CONFIG, run_enkf_frame
from mekong_da.config import DATA_DIR, SM_PERIODS
from mekong_da.metrics import metrics, metrics_table

RESOLUTION = "9km"


def main() -> None:
    inputs = pd.read_csv(DATA_DIR / "enkf" / f"inputs_{RESOLUTION}.csv")
    print(f"Loaded {len(inputs)} days of {RESOLUTION} inputs, "
          f"{int(inputs['obs_available'].sum())} of them with a retrieval to assimilate.\n")

    # ---------------------------------------------------------------- run the filter
    print(f"Filter settings: {OPTIMAL_CONFIG}\n")
    result = run_enkf_frame(inputs, config=OPTIMAL_CONFIG)

    keep = result["obs_available"].to_numpy(bool)
    observation = result["observation_corrected"].to_numpy()[keep]
    open_loop = metrics(result["baseline"].to_numpy()[keep], observation)
    analysis = metrics(result["analysis"].to_numpy()[keep], observation)

    print("Skill against the assimilated observation, full record:")
    print(f"  open loop  RMSE {open_loop['RMSE']:.4f}  NSE {open_loop['NSE']:6.3f}  KGE {open_loop['KGE']:.3f}")
    print(f"  analysis   RMSE {analysis['RMSE']:.4f}  NSE {analysis['NSE']:6.3f}  KGE {analysis['KGE']:.3f}")
    print(f"  RMSE reduced by {100 * (1 - analysis['RMSE'] / open_loop['RMSE']):.1f} per cent\n")

    # ------------------------------------------------- check against the published series
    reference = pd.read_csv(DATA_DIR / "enkf" / f"reference_{RESOLUTION}.csv")
    diff = np.nanmax(np.abs(result["analysis"].to_numpy() - reference["analysis"].to_numpy()))
    print(f"Largest disagreement with the published series: {diff:.2e}")
    print("Reproduced." if diff < 1e-12 else "NOT reproduced, see docs/07_pitfalls.md.\n")

    # ------------------------------------------------------------ the filter's diagnostics
    innovation = result["innovation"].dropna().to_numpy()
    gain = result["kalman_gain"].dropna().to_numpy()
    ac1 = np.corrcoef(innovation[:-1], innovation[1:])[0, 1]
    print(f"\nFilter diagnostics: mean gain {gain.mean():.3f} "
          f"(range {gain.min():.3f} to {gain.max():.3f}), "
          f"innovation lag-1 autocorrelation {ac1:+.4f}.")
    print("A gain near 0.5 means model and observation are trusted about equally, and a")
    print("lag-1 autocorrelation near zero means the filter is not over-correcting.")
    print("It is not a whiteness test; see docs/05_enkf_and_tuning.md.\n")

    # ---------------------------------------------------------------- the skill table
    table = metrics_table(
        result, obs_col="observation_corrected",
        model_cols={"Open-Loop": "baseline", "EnKF": "analysis"},
        periods=SM_PERIODS, row_filter=result["obs_available"], Resolution=RESOLUTION,
    )
    print("Skill by evaluation window:")
    print(table[["Period", "Model", "N", "RMSE", "NSE", "KGE"]].round(4).to_string(index=False))

    # -------------------------------------------- a small experiment of your own
    print("\nHow the analysis responds to trusting the observation more or less:")
    print(f"  {'obs error':>10} {'mean gain':>10} {'RMSE':>8} {'NSE':>7}")
    for obs_error in (0.02, 0.04, 0.08, 0.16):
        config = EnKFConfig(**{**OPTIMAL_CONFIG.to_dict(), "obs_error_std": obs_error})
        run = run_enkf_frame(inputs, config=config)
        skill = metrics(run["analysis"].to_numpy()[keep], observation)
        print(f"  {obs_error:>10.2f} {run['kalman_gain'].mean():>10.3f} "
              f"{skill['RMSE']:>8.4f} {skill['NSE']:>7.3f}")
    print("\nSkill against the assimilated observation rises as the filter is told to trust")
    print("that observation more, which is close to circular. It is why the parameter search")
    print("scores filter health alongside skill: see docs/05_enkf_and_tuning.md.")


if __name__ == "__main__":
    main()
