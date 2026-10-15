"""Two-stage parameter search for the offline scalar diagnostic.

The first stage varies observation and process error; the second varies inflation
and persistence. The declared score combines heuristic innovation criteria with
input-agreement Nash-Sutcliffe efficiency. Its gain reference and weights are
design choices, rather than theoretical conditions guaranteeing filter correctness.

Bias correction uses the selected training period. Innovation diagnostics use all
available update days, while validation metrics use the supplied validation mask.
Those metrics still compare analyses with assimilated targets. A separate period
does not make the observation target independent. A10 reports current lag and
period comparisons; historical negative autocorrelation claims are not retained.

Author: Sebastian R.O. Marshall."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .biascorrect import cdf_match_rank
from .config import DEFAULT_CONFIG, EnKFConfig
from .enkf import run_enkf

__all__ = [
    "OBS_ERROR_GRID", "MODEL_ERROR_GRID", "INFLATION_GRID", "PHI_GRID",
    "TRAIN_END", "VALIDATION_START",
    "diagnostics", "evaluate_config", "sweep", "score", "run_tuning",
]

OBS_ERROR_GRID: Tuple[float, ...] = (0.04, 0.06, 0.08, 0.10)
MODEL_ERROR_GRID: Tuple[float, ...] = (0.02, 0.04, 0.06, 0.08)
INFLATION_GRID: Tuple[float, ...] = (1.02, 1.05, 1.10, 1.15)
PHI_GRID: Tuple[float, ...] = (0.50, 0.65, 0.70, 0.80, 0.90)

#: Last day used to build the bias-correction transfer function.
TRAIN_END = pd.Timestamp("2019-12-31")
#: First day of the held-out window the search scores on.
VALIDATION_START = pd.Timestamp("2020-01-01")


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2))


def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def diagnostics(
    innovations: np.ndarray,
    gains: np.ndarray,
    forecast_spread: np.ndarray,
    obs_error: float,
    obs_values: np.ndarray,
    open_loop_values: np.ndarray,
    analysis_values: np.ndarray,
    validation_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Innovation and input-agreement diagnostics for one configuration.

    Parameters
    ----------
    innovations:
        True innovations, ``observation - forecast_mean``, on update days only.
    gains:
        Kalman gain applied on each update day.
    forecast_spread:
        Prior ensemble standard deviation on each update day.
    obs_error:
        Observation error standard deviation of the configuration under test. It is the
        R term of the theoretical innovation variance, so it must track the swept value
        rather than be pinned at the default.
    obs_values, open_loop_values, analysis_values:
        Observation, open-loop and analysis series restricted to update days.
    validation_mask:
        Boolean mask over the update days selecting the held-out window.

    Returns
    -------
    dict
        Filter-health terms (``innov_autocorr_1``, ``mean_K``, ``innov_var_ratio`` and
        companions) followed by input-agreement terms for the full and validation windows.

    Notes
    -----
    ``innov_var_ratio`` is ``var(innovations) / mean(forecast_spread**2 + obs_error**2)``
    with ``ddof=0`` throughout, where ``forecast_spread`` is the uninflated ensemble
    standard deviation recorded before each update (:mod:`mekong_da.enkf` inflates the
    variance only inside the gain, so the denominator omits inflation). A value near 1 is a
    diagnostic reference under these variance conventions. Values below or above one
    indicate differences between realized innovation variance and this denominator.
    This single moment condition does not establish full filter consistency.
    """
    if len(innovations) > 10:
        centred = innovations - np.mean(innovations)
        autocorr_1 = float(np.corrcoef(centred[:-1], centred[1:])[0, 1])
    else:
        autocorr_1 = np.nan

    predicted_var = forecast_spread**2 + obs_error**2
    mean_pred = float(np.mean(predicted_var))
    innov_ratio = float(np.var(innovations) / mean_pred) if mean_pred > 0 else np.nan

    out = {
        "innov_autocorr_1": autocorr_1,
        "mean_K": float(np.mean(gains)),
        "median_K": float(np.median(gains)),
        "std_K": float(np.std(gains)),
        "pct_K_extreme": float(np.mean((gains > 0.95) | (gains < 0.05)) * 100),
        "innov_var_ratio": innov_ratio,
        "mean_innov": float(np.mean(innovations)),
        "std_innov": float(np.std(innovations)),
        "mean_spread": float(np.mean(forecast_spread)),
        "rmse_base": float(np.sqrt(np.mean((open_loop_values - obs_values) ** 2))),
        "rmse_da": float(np.sqrt(np.mean((analysis_values - obs_values) ** 2))),
        "r_base": float(np.corrcoef(open_loop_values, obs_values)[0, 1]),
        "r_da": float(np.corrcoef(analysis_values, obs_values)[0, 1]),
        "nse_base": _nse(obs_values, open_loop_values),
        "nse_da": _nse(obs_values, analysis_values),
        "kge_base": _kge(obs_values, open_loop_values),
        "kge_da": _kge(obs_values, analysis_values),
        "bias_da": float(np.mean(analysis_values - obs_values)),
    }
    out["rmse_improve_pct"] = (1 - out["rmse_da"] / out["rmse_base"]) * 100

    if validation_mask is not None and validation_mask.sum() > 10:
        o, a = obs_values[validation_mask], analysis_values[validation_mask]
        out.update(
            rmse_da_val=float(np.sqrt(np.mean((a - o) ** 2))),
            r_da_val=float(np.corrcoef(a, o)[0, 1]),
            nse_da_val=_nse(o, a),
            kge_da_val=_kge(o, a),
        )
    else:
        out.update(rmse_da_val=np.nan, r_da_val=np.nan, nse_da_val=np.nan, kge_da_val=np.nan)
    return out


def evaluate_config(
    open_loop: pd.Series,
    observation: pd.Series,
    config: EnKFConfig,
) -> Dict[str, float]:
    """Run one configuration and return its diagnostics."""
    result = run_enkf(
        open_loop.to_numpy(float),
        observation.to_numpy(float),
        obs_available=None,
        config=config,
    )
    update_days = np.isfinite(result.innovation)
    valid = observation.notna().to_numpy(bool)
    validation_mask = np.asarray(open_loop.index[valid] >= VALIDATION_START)

    metrics = diagnostics(
        innovations=result.innovation[update_days],
        gains=result.kalman_gain[update_days],
        forecast_spread=result.forecast_spread[update_days],
        obs_error=config.obs_error_std,
        obs_values=observation.to_numpy(float)[valid],
        open_loop_values=open_loop.to_numpy(float)[valid],
        analysis_values=result.analysis[valid],
        validation_mask=validation_mask,
    )
    metrics.update(
        obs_error_std=config.obs_error_std,
        model_error_std=config.model_error_std,
        inflation_factor=config.inflation_factor,
        persistence_phi=config.persistence_phi,
    )
    return metrics


def score(metrics: Dict[str, float]) -> float:
    """Rank the tested grid with the preserved heuristic weighted score.

    Forty percent weights lag-one correlation, gain proximity to 0.5, and the
    declared innovation-variance ratio. Sixty percent weights full-period and
    validation-period input-agreement NSE. These are chosen weights and reference
    values. The score does not establish independent accuracy or optimal covariance.
    """
    ac = max(0.0, 1 - abs(metrics["innov_autocorr_1"]) / 0.5)
    k = max(0.0, 1 - abs(metrics["mean_K"] - 0.50) / 0.5)
    ratio = max(0.0, 1 - abs(metrics["innov_var_ratio"] - 1.0) / 2.0)
    health = (ac + k + ratio) / 3

    full = max(0.0, metrics["nse_da"])
    held_out = max(0.0, metrics["nse_da_val"]) if np.isfinite(metrics["nse_da_val"]) else 0.0
    return 0.4 * health + 0.6 * (0.5 * full + 0.5 * held_out)


def sweep(
    open_loop: pd.Series,
    observation: pd.Series,
    base_config: EnKFConfig,
    grid: Dict[str, Sequence[float]],
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate every combination in ``grid`` and return one row per configuration.

    Parameters
    ----------
    grid:
        Mapping of :class:`~mekong_da.config.EnKFConfig` field name to the values to try.
        Fields absent from the grid keep their value in ``base_config``.
    """
    names = list(grid)
    rows: List[Dict[str, float]] = []
    for values in product(*(grid[n] for n in names)):
        config = EnKFConfig(**{**base_config.to_dict(), **dict(zip(names, values))})
        metrics = evaluate_config(open_loop, observation, config)
        rows.append(metrics)
        if verbose:
            spec = " ".join(f"{n}={v}" for n, v in zip(names, values))
            print(
                f"    {spec} | K={metrics['mean_K']:.3f} AC1={metrics['innov_autocorr_1']:+.3f} "
                f"IVR={metrics['innov_var_ratio']:.3f} | NSE={metrics['nse_da']:.3f} "
                f"KGE={metrics['kge_da']:.3f} RMSE={metrics['rmse_da']:.4f}"
            )
    return pd.DataFrame(rows)


def run_tuning(
    open_loop: pd.Series,
    smap_by_resolution: Dict[str, pd.Series],
    tuning_resolution: str = "9km",
    out_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the full two-stage search plus the multi-resolution validation.

    Parameters
    ----------
    open_loop:
        Daily open-loop soil moisture, m3/m3, indexed by date and gap-free.
    smap_by_resolution:
        Raw daily basin-mean SMAP series per resolution, indexed by date. Each is
        reindexed onto ``open_loop`` and CDF-matched on the training window before use.
    tuning_resolution:
        Which resolution the two search stages run on. The published search used 9 km,
        the product with the longest record and the fewest gaps.
    out_dir:
        Directory to write ``stage1a_obs_model_error.csv``, ``stage1b_inflation_phi.csv``,
        ``validation_default_vs_optimal.csv`` and ``optimal_config.json`` into.

    Returns
    -------
    dict
        ``{"stage1a": DataFrame, "stage1b": DataFrame, "validation": DataFrame,
        "optimal": EnKFConfig}``.
    """
    train_mask = np.asarray(open_loop.index <= TRAIN_END)

    def corrected(series: pd.Series) -> pd.Series:
        aligned = series.reindex(open_loop.index)
        return pd.Series(
            cdf_match_rank(aligned.to_numpy(float), open_loop.to_numpy(float), train_mask=train_mask),
            index=open_loop.index,
        )

    obs = {res: corrected(s) for res, s in smap_by_resolution.items()}
    if tuning_resolution not in obs:
        raise KeyError(f"tuning resolution '{tuning_resolution}' not in {sorted(obs)}")
    tuning_obs = obs[tuning_resolution]

    if verbose:
        print(f"Stage 1a: observation error x model error, on {tuning_resolution}")
    stage1a = sweep(
        open_loop, tuning_obs, DEFAULT_CONFIG,
        {"obs_error_std": OBS_ERROR_GRID, "model_error_std": MODEL_ERROR_GRID},
        verbose=verbose,
    )
    best_1a = stage1a.loc[stage1a.apply(score, axis=1).idxmax()]
    if verbose:
        print(f"  highest declared grid score: obs_error={best_1a.obs_error_std}, model_error={best_1a.model_error_std}")

    stage1b_base = EnKFConfig(
        **{
            **DEFAULT_CONFIG.to_dict(),
            "obs_error_std": float(best_1a.obs_error_std),
            "model_error_std": float(best_1a.model_error_std),
        }
    )
    if verbose:
        print("Stage 1b: inflation x persistence")
    stage1b = sweep(
        open_loop, tuning_obs, stage1b_base,
        {"inflation_factor": INFLATION_GRID, "persistence_phi": PHI_GRID},
        verbose=verbose,
    )
    best_1b = stage1b.loc[stage1b.apply(score, axis=1).idxmax()]

    optimal = EnKFConfig(
        ensemble_size=DEFAULT_CONFIG.ensemble_size,
        obs_error_std=float(best_1b.obs_error_std),
        model_error_std=float(best_1b.model_error_std),
        inflation_factor=float(best_1b.inflation_factor),
        persistence_phi=float(best_1b.persistence_phi),
    )
    if verbose:
        print(f"  selected: {optimal}")
        print("Stage 2: default versus selected, all resolutions")

    rows = []
    for res, series in obs.items():
        for label, config in (("DEFAULT", DEFAULT_CONFIG), ("OPTIMAL", optimal)):
            metrics = evaluate_config(open_loop, series, config)
            metrics.update(config=label, resolution=res, ensemble_size=config.ensemble_size)
            rows.append(metrics)
            if verbose:
                print(
                    f"  {res:>5s} {label:<8s} K={metrics['mean_K']:.3f} "
                    f"AC1={metrics['innov_autocorr_1']:+.3f} NSE={metrics['nse_da']:.3f} "
                    f"KGE={metrics['kge_da']:.3f}"
                )
    validation = pd.DataFrame(rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stage1a.to_csv(out_dir / "stage1a_obs_model_error.csv", index=False)
        stage1b.to_csv(out_dir / "stage1b_inflation_phi.csv", index=False)
        validation.to_csv(out_dir / "validation_default_vs_optimal.csv", index=False)
        import json

        (out_dir / "optimal_config.json").write_text(
            json.dumps(
                {
                    "ensemble_size": optimal.ensemble_size,
                    "obs_error_std": optimal.obs_error_std,
                    "model_error_std": optimal.model_error_std,
                    "inflation_factor": optimal.inflation_factor,
                    "persistence_phi": optimal.persistence_phi,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {"stage1a": stage1a, "stage1b": stage1b, "validation": validation, "optimal": optimal}
