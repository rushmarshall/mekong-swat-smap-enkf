"""Goodness-of-fit metrics, in the single definition used for the published tables.

One function serves both tables. That is deliberate: the soil-moisture table (T1) and the
streamflow table (T2) are computed with identical formulas so that the two halves of the
study are directly comparable. In particular the Kling-Gupta efficiency here uses the
standard-deviation ratio for the variability term,

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2),
    alpha = sd(sim) / sd(obs),   beta = mean(sim) / mean(obs),

which is Gupta et al. (2009) as originally stated. A coefficient-of-variation variant
(gamma = CV(sim) / CV(obs)) also circulates; it gives different numbers and is *not* what
the published tables use. See ``docs/07_pitfalls.md``.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

__all__ = ["metrics", "metrics_table"]

_NAN_RESULT = dict(
    N=0, RMSE=np.nan, MAE=np.nan, Bias=np.nan, R=np.nan, R2=np.nan,
    NSE=np.nan, KGE=np.nan, PBIAS=np.nan, ubRMSE=np.nan,
)


def metrics(pred: Iterable[float], obs: Iterable[float], min_n: int = 3) -> Dict[str, float]:
    """Compute the standard metric set on the pairwise-complete sample.

    Parameters
    ----------
    pred:
        Simulated or assimilated values.
    obs:
        Reference values, same length as ``pred``.
    min_n:
        Below this many pairwise-complete values every metric returns NaN rather than a
        number computed from too little data.

    Returns
    -------
    dict
        ``N`` (pairwise-complete count), ``RMSE``, ``MAE``, ``Bias`` (mean signed error,
        simulated minus observed), ``R``, ``R2``, ``NSE``, ``KGE``, ``PBIAS`` (per cent)
        and ``ubRMSE`` (bias-removed RMSE). Values are unrounded; round at presentation
        time using the conventions in ``docs/01_workflow.md``.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError(f"pred shape {pred.shape} does not match obs shape {obs.shape}")

    keep = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[keep], obs[keep]
    if len(obs) < min_n:
        return dict(_NAN_RESULT, N=int(len(obs)))

    err = pred - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    if pred.std(ddof=0) == 0 or obs.std(ddof=0) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(pred, obs)[0, 1])

    denom = np.sum((obs - obs.mean()) ** 2)
    nse = float(1 - np.sum((obs - pred) ** 2) / denom) if denom > 0 else np.nan

    alpha = pred.std(ddof=0) / obs.std(ddof=0) if obs.std(ddof=0) > 0 else np.nan
    beta = pred.mean() / obs.mean() if obs.mean() != 0 else np.nan
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    pbias = float(100 * np.sum(err) / np.sum(obs)) if np.sum(obs) != 0 else np.nan
    ubrmse = float(np.sqrt(max(rmse**2 - bias**2, 0.0)))

    return dict(
        N=int(len(obs)), RMSE=rmse, MAE=mae, Bias=bias, R=r, R2=float(r**2),
        NSE=nse, KGE=kge, PBIAS=pbias, ubRMSE=ubrmse,
    )


def metrics_table(
    frame: pd.DataFrame,
    obs_col: str,
    model_cols: Dict[str, str],
    periods: Dict[str, tuple],
    date_col: str = "date",
    row_filter: Optional[pd.Series] = None,
    **id_fields: str,
) -> pd.DataFrame:
    """Evaluate several model columns over several date windows.

    Parameters
    ----------
    frame:
        Long frame with a date column, one observation column and one column per model.
    obs_col:
        Name of the reference column.
    model_cols:
        Mapping of display label to column name, for example ``{"Open-Loop": "baseline"}``.
    periods:
        Mapping of display label to ``(start, end)`` date strings, inclusive.
    date_col:
        Name of the date column.
    row_filter:
        Optional boolean mask applied before windowing, for example an observation
        availability flag.
    **id_fields:
        Constant identifier columns copied onto every output row, such as
        ``Station="Kratie"`` or ``Resolution="9km"``.

    Returns
    -------
    pandas.DataFrame
        One row per (period, model) with the identifier fields and the metric set.
    """
    df = frame.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if row_filter is not None:
        df = df[row_filter.to_numpy(bool)]

    rows = []
    for period_label, (start, end) in periods.items():
        window = df[(df[date_col] >= start) & (df[date_col] <= end)]
        for model_label, col in model_cols.items():
            if col not in window.columns:
                continue
            rows.append(
                dict(
                    **id_fields,
                    Period=period_label,
                    Model=model_label,
                    **metrics(window[col].to_numpy(float), window[obs_col].to_numpy(float)),
                )
            )
    return pd.DataFrame(rows)
