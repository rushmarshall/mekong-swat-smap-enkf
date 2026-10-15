# The offline filter, and how its settings were chosen

This document covers the offline scalar filter in `src/mekong_da/enkf.py` and its parameter
search in `src/mekong_da/tuning.py`. The filter inside the coupled model is a different
implementation, described in [04_coupled_model.md](04_coupled_model.md).

---

## The problem, stated

Let `x_t` be basin-average volumetric soil-water content on day `t`, in m3/m3, and `y_t` a
CDF-matched SMAP retrieval of the same quantity. State and observation are the same scalar,
so the observation operator `H` is the identity, and the Kalman machinery collapses to
arithmetic that fits in a paragraph.

**Forecast.** The ensemble is carried forward as an AR(1) process anchored to the SWAT open
loop `b_t`, with additive noise:

```
x_i^f(t) = phi * x_i^a(t-1) + (1 - phi) * b_t + eta_i,     eta_i ~ N(0, sigma_m^2)
```

`phi` is memory of yesterday's analysis; `1 - phi` is trust in today's open loop.

**Analysis.** On days with a retrieval, the gain is formed from the inflated forecast
variance and each member is updated against its own perturbed observation:

```
P_t   = lambda * var(x^f(t))                       lambda = inflation
K_t   = P_t / (P_t + sigma_o^2)
x_i^a = x_i^f + K_t * (y_t + eps_i - x_i^f),        eps_i ~ N(0, sigma_o^2)
```

Every member is then clipped to `[0.01, 0.60]` m3/m3, and the reported analysis is the
ensemble mean, the reported uncertainty its standard deviation.

Perturbing the observation per member, rather than updating the mean and rescaling, is what
makes this a stochastic Ensemble Kalman Filter (Burgers et al., 1998): it keeps the
posterior spread from collapsing to zero.

---

## Why the forecast is a nudge and not a model

`x^f` is not produced by re-integrating SWAT. In this experiment the forecast is an AR(1)
blend toward the open loop, which makes the filter cheap, transparent and exactly
reproducible, at the cost of not letting the analysis feed back into the hydrology.

That cost is the reason for the second experiment. The Fortran system re-integrates the
model, and the discharge response is what it adds. Keeping the two separate means the
offline result can be checked in seconds, without a calibrated model.

---

## Every parameter

| Symbol | Field | Value | What moving it does |
|---|---|---|---|
| N | `ensemble_size` | 200 | Sampling of the forecast variance. Below about 50 the gain becomes noisy; above 200 the return is small. |
| sigma_o | `obs_error_std` | 0.04 | Trust in SMAP. Larger means a smaller gain and a more conservative analysis. |
| sigma_m | `model_error_std` | 0.04 | Trust in the model. Larger means a larger gain. |
| lambda | `inflation_factor` | 1.02 | Counters ensemble collapse. Too large inflates the gain and the analysis chases noise. |
| phi | `persistence_phi` | 0.50 | Memory. Too large and yesterday's correction propagates forward, which shows up as negative innovation autocorrelation. |
| bounds | `sm_bounds` | (0.01, 0.60) | Physical clip. At the tuned settings the dry bound binds on 1.57, 0.63 and 0.77 per cent of member-steps at 9 km, 1 km and 400 m, and the wet bound binds on 0.00 per cent. The exposure is one-sided and small. |
| seed | `seed` | 42 | Fixed so two runs of one configuration are comparable. |

`sigma_o = sigma_m = 0.04` says the filter trusts model and observation equally, which puts
the gain near 0.5. That is a reasonable prior for a satellite product whose stated accuracy
target is 0.04 m3/m3 against a model with no better claim. These values were selected for the
Mekong basin-average series and must be re-tuned for another basin or product.
`OPTIMAL_CONFIG` is the name of the selected configuration in the code, not a claim that it
is optimal in general.

---

## How the settings were chosen

By a two-stage grid search, which `mekong-da tune` reruns.

### The criterion, and why it is not accuracy alone

A filter can post an excellent Nash-Sutcliffe efficiency and still be badly specified, by
trusting the observation so heavily that it chases every retrieval. The signature is
autocorrelated innovations: correcting hard on one day leaves the next day's forecast
overshooting in the opposite direction, so the lag-1 autocorrelation of the innovation goes
**negative**. A well-specified filter tends to leave innovations close to white.

The search therefore ranks each configuration by a composite score (`tuning.score`): 40 per
cent innovation diagnostics (lag-1 innovation autocorrelation, gain distance from 0.5 and
innovation variance ratio) and 60 per cent agreement with the assimilated observation, split
evenly between the full period and a held-out window after 2019-12-31, so a configuration
that only fits the training window cannot win. The weights and reference values are chosen,
not derived; the innovation terms are diagnostics, not a proof that the filter is correctly
specified; and the efficiency terms measure agreement with the input, not independent
accuracy.

### The stages

**Stage 1a.** Observation error {0.04, 0.06, 0.08, 0.10} against model error
{0.02, 0.04, 0.06, 0.08}, 16 runs at the pre-tuning inflation and persistence. Winner:
0.04 and 0.04.

**Stage 1b.** Inflation {1.02, 1.05, 1.10, 1.15} against persistence
{0.50, 0.65, 0.70, 0.80, 0.90}, 20 runs at the stage 1a winner. Winner: 1.02 and 0.50.

**Stage 2.** The selected configuration against the pre-tuning default, at all three
resolutions.

### What the tuning bought

| Resolution | Configuration | Gain | Innovation AC1 | Variance ratio | NSE | KGE |
|---|---|---|---|---|---|---|
| 9 km | default | 0.568 | -0.259 | 0.482 | 0.880 | 0.850 |
| 9 km | **selected** | 0.532 | **-0.045** | 0.538 | 0.854 | **0.854** |
| 1 km | default | 0.575 | -0.320 | 0.716 | 0.824 | 0.773 |
| 1 km | **selected** | 0.536 | **-0.102** | 0.799 | 0.784 | **0.779** |
| 400 m | default | 0.575 | -0.251 | 0.814 | 0.805 | 0.738 |
| 400 m | **selected** | 0.536 | **-0.029** | 0.924 | 0.755 | **0.740** |

Nash-Sutcliffe efficiency (NSE) falls by three to five per cent and Kling-Gupta efficiency
(KGE) rises at every resolution. That trade is the intended one: the default's higher
efficiency came from a filter that over-corrected and left strongly negative lag-1
innovation autocorrelation. The dominant change is `phi` from 0.70 to 0.50; the inflation
change is small. The change moves the same way at all three resolutions and in the held-out
window.

### Reproducing it

```bash
mekong-da tune
```

About 13 s on an Apple M3 Ultra. On the example inputs it selects 0.04, 0.04, 1.02 and
0.50, and `tests/test_tuning.py` checks the stage 1a grid against the shipped one to 1e-9.
On your own series, replace the inputs in `data/enkf_tuning/inputs/` (or pass `--data-dir`)
and run it with `--resolution` set to the product to tune on; it writes the grids and
`optimal_config.json` to `outputs/enkf_tuning/`, which `mekong-da enkf --config` accepts.

The search ran on a 2015-2020 basin-mean series, which is a smaller and earlier sample than
the 2015-2024 example inputs in `data/enkf/`. Its absolute skill numbers therefore differ;
what carries over is the ranking of configurations, which is all a search is for.

That difference in sample also explains an apparent contradiction between the two tables in
this document. On the 2015-2020 tuning sample the selected configuration leaves an
innovation autocorrelation of -0.045 at 9 km; on the 2015-2024 example inputs the same
configuration leaves +0.045. Both are correct for their sample, both are small, and neither
is evidence of a change in the filter. Quote whichever matches the run you are describing,
and say which.

---

## Reading the diagnostics the filter emits

`outputs/enkf/da_results_<res>.csv` carries, per day:

| Column | Meaning |
|---|---|
| `analysis` | posterior ensemble mean, the assimilated series |
| `uncertainty` | posterior ensemble standard deviation |
| `forecast_mean`, `forecast_spread` | prior, recorded immediately before the update |
| `innovation` | `observation - forecast_mean`, the true innovation |
| `kalman_gain` | the gain actually applied, in [0, 1] |
| `update` | `analysis - baseline`, how far the analysis moved from the open loop |

Measured at the selected settings on the example inputs:

| Resolution | Gain min / mean / max | True innovation AC1 |
|---|---|---|
| 9 km | 0.390 / 0.533 / 0.627 | +0.045 |
| 1 km | 0.390 / 0.536 / 0.630 | +0.067 |
| 400 m | 0.390 / 0.536 / 0.650 | +0.132 |

Two cautions on those numbers.

First, they are the filter's own innovations, observation minus forecast mean on update
days. Observation minus open loop is a different quantity and has very different
statistics ([07_pitfalls.md](07_pitfalls.md)).

Second, a lag-1 autocorrelation near zero is not a whiteness test. Over twelve lags the
same series peaks at 0.704 at lag 8 for 9 km, and Ljung-Box rejects whiteness at all three
resolutions. That lag-8 peak is the orbital revisit cycle aliasing into the residuals.
Report it as near-white at lag 1 and no further.
