# Diagnostic analyses

`analyses/` holds the numerical analyses behind the paper's results on observation
schedules, thinning, error specification, filter sensitivity and bias correction. Each script
reruns the offline scalar filter (`mekong_da.enkf.run_enkf`) many times under controlled
changes and writes tables to `outputs/analyses/`. They produce numbers, not figures.

Two cautions apply to all of them. Scores compare each analysis with the observation it
assimilated, so they measure agreement with the input under the stated filter settings, not
independent soil-moisture accuracy. And products are compared on their own corrected
targets, so cross-product differences are conditional on those targets.

## Running them

```bash
pip install -e ".[analyses]"          # joblib and statsmodels
mekong-da enkf                        # writes outputs/enkf/
mekong-da tune                        # writes outputs/enkf_tuning/
python analyses/A_temporal_subsampling.py
python analyses/A2_coverage_dose_response.py
python analyses/R_measurements.py
python analyses/A3_daylevel_interval.py
python analyses/A4_ordering_and_ownday.py
python analyses/A6_filter_consistency.py
python analyses/A7_error_specification.py
python analyses/A8_schedule_persistence.py
python analyses/A9_invariance_sweep.py
python analyses/A10_innovation_diagnostics.py
python analyses/A11_block_length_and_increments.py
python analyses/B_resolution_specific_obs_error.py
python analyses/C_out_of_sample_bias_correction.py
python analyses/A5_gaps_and_offbasin.py        # partly; full output needs subbasin series
python analyses/A13_footprint_clamp.py         # needs subbasin series
```

The order matters where one script reads another's output (see the table). On the example
data the thirteen scripts that need no subbasin series ran in about 40 minutes on a 28-core
Apple M3 Ultra, all exiting cleanly. Most use 50 filter realisations (seeds 42 to 91), which
is where the time goes.

## What each script computes

| Script | Question | Writes | Reads from other scripts | Time |
|---|---|---|---|---|
| `A_temporal_subsampling.py` | Does assimilating the 9 km values only on the finer products' calendars change agreement on the 3,040 common days? | `A_schedules.csv`, `A_subsampling.csv` | none | 165 s |
| `A2_coverage_dose_response.py` | How does agreement change as 9 km retrievals are thinned regularly or at random? | `A2_dose_response.csv` | none | 918 s |
| `R_measurements.py` | Headline input-agreement scores, sensitivity controls, and schedule comparisons on common, reduced and fixed calendars (R1 to R3), with a decomposition into updated and skipped days | `R1_*.csv`, `R2_*.csv`, `R3_decomposition.csv` | `A2_dose_response.csv` | under 1 min |
| `A3_daylevel_interval.py` | Paired moving-block intervals for the common-day schedule contrast | `A3_daylevel_interval.csv` | none | 74 s |
| `A4_ordering_and_ownday.py` | Separates the effect of which days are scored from the effect of which days are assimilated | `A4_ordering_and_ownday.csv` | `R2_evalsets.csv` | 170 s |
| `A6_filter_consistency.py` | Innovation mean, lag-1 correlation, spread and normalized innovation variance per product | `A6_filter_consistency.csv` | none | 2 s |
| `A7_error_specification.py` | Sensitivity to model-error standard deviation, including the setting where normalized innovation variance equals one | `A7_error_specification.csv` | `A6_filter_consistency.csv` | 34 s |
| `A8_schedule_persistence.py` | Do the schedule contrasts hold across five persistence settings? | `A8_schedule_persistence.csv` | `A3_*.csv`, `A4_*.csv`, `R2_evalsets.csv` | 32 s |
| `A9_invariance_sweep.py` | Does the product ordering hold across eight filter configurations? | `A9_invariance_sweep.csv` | imports A7; checks `A7_error_specification.csv` | 28 s |
| `A10_innovation_diagnostics.py` | Innovation autocorrelation at lags 1 to 16 under two pairing conventions, with Ljung-Box statistics | `A10_innovation_acf.csv`, `A10_tuning_ac1.csv` | `A6_filter_consistency.csv`; the tuning grid in `data/enkf_tuning/` | 32 s |
| `A11_block_length_and_increments.py` | Sensitivity of the intervals to block length (7 to 180 days), and analysis displacements by period | `A11_*.csv` | `A4_ordering_and_ownday.csv` | 14 s |
| `B_resolution_specific_obs_error.py` | A product-specific observation-error estimate from residual diagnostics, and agreement under that and a common error grid | `B_sigma_estimates.csv`, `B_obs_error_sensitivity.csv` | none | 561 s |
| `C_out_of_sample_bias_correction.py` | Sensitivity to the CDF-matching training window | `C_out_of_sample.csv` | none | 217 s |
| `A5_gaps_and_offbasin.py` | Longest observation gaps, and retrieval days present in the basin box but in no subbasin | printed only | none | seconds |
| `A13_footprint_clamp.py` | Product disagreement under alternative subbasin supports | `A13_footprint_clamp.csv` | none | not measured |

Each script's module docstring states its design, sample sizes and limits in full; read it
before quoting a number from its output. Several scripts assert agreement with earlier
outputs or with values reported in the paper; on new data those checks are expected to fail
and can be removed.

## Running them on your own series

The scripts read the offline filter inputs from `data/enkf/inputs_<product>.csv` (and A11
and `R_measurements.py` also read `data/enkf/reference_<product>.csv`), relative to the
repository root. To use your own series:

1. Put your inputs in `data/enkf/` with the same columns
   ([03_data.md](03_data.md)), or edit the path at the top of each script.
2. Edit the product list `RES` (in `A_temporal_subsampling.py`, which most scripts import,
   and in the scripts that define their own) and the seed list `SEEDS` if you want fewer
   realisations.
3. Replace values the scripts compare against the Mekong results (for example `OPEN_REF` in
   `A7_error_specification.py` and `PRINTED_S16` in `A11_block_length_and_increments.py`),
   which do not apply to another basin. Labels such as "the 9 km calendar, 3,447 days" in
   `R_measurements.py` name the Mekong day counts and are also used as lookup keys by A8,
   so rename them consistently.
4. Run `mekong-da enkf` and `mekong-da tune` first, then the scripts in the order above.

A5 and A13 also read `data/smap_subbasin_daily/smap_<product>_subbasin_daily.csv`; without
those files A5 prints only the gap summary and A13 stops with a message.
