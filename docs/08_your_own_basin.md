# Using this on your own basin

The study is about the Mekong, but almost nothing in the Python is. Every basin-specific
number lives in one file, `src/mekong_da/config.py`, and the code reads those values rather
than assuming them. [01_workflow.md](01_workflow.md) gives the full sequence of commands;
this document says what to change along the way. What follows is what actually has to change, in the order it has to
change, and what breaks quietly if a step is skipped.

Two paths, and the difference in effort between them is large. Decide which one you are on
before touching anything.

---

## Path A: the filter only, on your own series

**Effort: an afternoon. No SWAT project needed, no Fortran, no changes to this repository.**

If what you want is the Ensemble Kalman Filter applied to a soil-moisture series you already
have, you do not need to adapt this repository at all. The filter takes plain arrays and has
no idea where they came from.

```python
from mekong_da import EnKFConfig, metrics, run_enkf

result = run_enkf(
    open_loop=my_model_series,        # 1-D array, your model without assimilation
    observation=my_corrected_obs,     # same length, same units, already bias-corrected
    obs_available=my_obs_mask,        # boolean, same length
    config=EnKFConfig(obs_error_std=..., model_error_std=..., sm_bounds=(lo, hi)),
)
```

Three things you must do rather than inherit:

1. **Bias-correct the observation onto your model's own distribution.** Use
   `biascorrect.cdf_match_rank(observation, model, train_mask=date <= train_end)`, fitted
   on your training window and applied forward, as the paper did with
   `tuning.TRAIN_END` (2019-12-31). Skipping this makes the filter chase the difference between two
   instruments' absolute values instead of the anomaly.
2. **Set `sm_bounds` to your variable's physical range.** The shipped `(0.01, 0.60)` is
   volumetric water content in m3/m3 and means nothing for anything else.
3. **Re-tune `obs_error_std` and `model_error_std`.** The shipped `0.04` is specific to SMAP
   soil moisture over this basin. Carrying it to a new variable, a new sensor or a new
   climate is the single most common way to get a filter that looks like it works and does
   not. `mekong-da tune` and [05_enkf_and_tuning.md](05_enkf_and_tuning.md) show the search.

That is the whole of Path A. Everything below is Path B.

---

## Path B: the coupled model, on your own SWAT project

**Effort: a project, not an afternoon.** You need a calibrated SWAT2012 project for your
basin, a satellite product mapped onto its subbasins, and the patience to run a multi-year
model several times.

### Step 1: describe your model's geometry

Everything downstream is sized from this. The command line uses `MEKONG_2024` from
`config.py` as its geometry, so set it to your project's values there (the name is kept so
every command picks it up), or pass `spec=` to `pipeline.prepare_fortran_run` and
`pipeline.harvest_fortran_run` from Python:

```python
MEKONG_2024 = SwatRunSpec(
    n_hru=...,          # count the .hru files in your TxtInOut
    n_subbasins=...,    # count the .sub files
    start_year=...,     # IYR in file.cio
    n_years=...,        # NBYR in file.cio
    nyskip=...,         # NYSKIP in file.cio
    n_days=...,         # total simulated days, leap years counted
)
```

Read the first five straight out of your project, do not estimate them:

```bash
ls TxtInOut/*.hru | wc -l          # n_hru
ls TxtInOut/*.sub | wc -l          # n_subbasins
grep -E "NBYR|IYR|NYSKIP" TxtInOut/file.cio
```

`n_days` must equal what the model itself computes from `file.cio`. Get it wrong and the run
reaches the end of an observation file hours in and dies. This is the most expensive mistake
on this page, because it costs a full run to discover.

**Careful with the counts.** Run `ls` against a directory that already holds model output
and you will count `output.hru` and `output.sub` too, giving 5,606 and 541 where the truth
is 5,605 and 540. Count in a clean project directory.

### Step 2: name your gauges and find their reaches

```python
STATIONS = (
    Station("Gauge_Name", "Gauge Name", reach_number),
    ...
)
```

The reach number is the SWAT reach that carries the gauge. **Getting it wrong produces a
complete, plausible hydrograph from the wrong river, with no error and no warning.** Confirm
each one against your model's `fig.fig` routing or the subbasin shapefile, then sanity-check
the mean annual flow against the gauge's published record before you trust a single metric.

### Step 3: supply the data, in the formats the code expects

Column names are the contract. These are the headers the code reads
([03_data.md](03_data.md) has the full description):

| File | Header | Notes |
|---|---|---|
| `data/smap_subbasin_daily/smap_<product>_subbasin_daily.csv` | `date, sub_<n>, sub_<n>, ...` | one column per subbasin, m3/m3. Column order does not matter; the name carries the subbasin number. In the Mekong application, 540 subbasin columns and 3,471 rows at 9 km, because only days with a retrieval appear |
| `data/enkf/inputs_<product>.csv` | `date, baseline, observation, observation_corrected, obs_available` | what the Python filter consumes. `baseline` is your open loop, `observation_corrected` is after CDF matching, `obs_available` is boolean |
| `data/hru_subbasin_mapping.csv` | `hru_index, subbasin, hru_in_sub` | one row per HRU (5,605 in the Mekong). This is how a subbasin observation reaches every HRU inside it |
| `data/streamflow/<Gauge>_merged_<year>.csv` | `date, swat_baseline, mrc_obs, swat_da_9km, swat_da_1km, swat_da_400m` | one column per arm plus the gauge. Keep the column name `mrc_obs` whatever the gauge network, because `streamflow_table` reads it; update `STREAMFLOW_MODELS` to match your arm columns |

Two shapes worth noticing before you build these. The subbasin file has one row per day
**with a retrieval**, not one row per simulated day, so it is shorter than the model
calendar. The merged streamflow files have one row per reported day (8,766 in the Mekong).

Set `RESOLUTIONS` to your product names. The filter, the tables and the Fortran path all
iterate over that tuple, so nothing else changes when you add or remove one.

### Step 4: set your evaluation windows

`SM_PERIODS` and `STREAMFLOW_PERIODS` are date ranges, not code. They must sit inside your
simulation period and inside your observation record, or a table comes back with a row of
`NaN` and no explanation.

### Step 5: stage and run every arm, including the control

```bash
mekong-da build-fortran
for arm in control <product> ...; do
  mekong-da prepare-run --run-dir /scratch/mybasin/$arm --resolution $arm \
                        --template /path/to/your/TxtInOut
  mekong-da run-fortran --run-dir /scratch/mybasin/$arm
  mekong-da harvest --rch /scratch/mybasin/$arm/output.rch --out outputs/mybasin_$arm.csv
done
```

The `control` arm is the matched control: same ensemble and random draws as the
assimilating arms, no observations ([04_coupled_model.md](04_coupled_model.md)).

`prepare-run` reports how many HRU-days carry an assimilable observation. For the Mekong at
9 km that is 1,742,807 of 55,276,510, which is 3.15 per cent. If your number is far larger,
suspect the missing-value screening; if it is near zero, suspect the subbasin mapping.

---

## Extending the framework

Before any of these changes, run `mekong-da reproduce` on a clean tree, so that if something
breaks later you know it was your change.

### Add a satellite product or resolution

**Touches:** `config.RESOLUTIONS`, plus the input files. Add the product name, then provide
`data/smap_subbasin_daily/smap_<name>_subbasin_daily.csv` for the coupled model and
`data/enkf/inputs_<name>.csv` for the offline filter, built as in [03_data.md](03_data.md).
The filter, the tables and the Fortran path all iterate over `RESOLUTIONS`.
**Check:** `mekong-da enkf --resolution <name>` reports a plausible assimilated day count and
error reduction.

### Add a gauge

**Touches:** `config.STATIONS`. Add a `Station(key, label, reach)`; `rch.station_frame`,
`pipeline.streamflow_table` and `pipeline.build_merged_streamflow` all read the tuple.
**Check:** `mekong-da harvest` on an existing `output.rch`, then compare the new column's
mean annual flow with the gauge record. An order-of-magnitude mismatch means the wrong reach.

### Change the filter, or try a different one

**Touches:** `enkf.py`. Read the first pitfall in [07_pitfalls.md](07_pitfalls.md) first.
The current implementation is pinned by `tests/test_enkf.py` to the expected example
output. Leave `run_enkf` untouched, add the alternative beside it returning the same
`EnKFResult`, select it behind an argument that defaults to the current filter, and add tests
for the new path without relaxing the existing ones. Everything downstream consumes
`EnKFResult`, so it works unchanged.

### Assimilate something other than soil moisture

In Python this is small: the offline filter is state-agnostic. Supply an open loop and an
observation in the same units, set `sm_bounds` to the variable's physical range, and re-tune.
In Fortran it is not small. The state vector (`no_sta_var`), the observation operator in
`init_da.f90`, the bounds in `EnKF.f90` and the write-back in `post_da.f90` are all specific
to soil-water storage, and a new variable needs new routines, a new observation branch in
`readsm.f90`, and a decision about where in the daily loop the update belongs. Budget it as a
modelling project.

### Extend the simulation period

**Touches:** `SwatRunSpec` and the SWAT project. `n_days` must equal what `maxndays.f90`
computes from `file.cio`; change `file.cio`, set `n_days` to match, then regenerate every
input file with `prepare-run`. **Check:** the first minutes of the run get past `readsm.f90`
without an end-of-file error.

---

## What must be re-tuned, and why

Not optional, and not a formality.

| Setting | Why the shipped value does not transfer |
|---|---|
| `obs_error_std` | It is the trust placed in one sensor over one basin. A different product, or the same product over different land cover, deserves a different number |
| `model_error_std` | It is your model's forecast spread, which follows your calibration, not this one |
| `persistence_phi` | 0.50 was chosen because higher values drove the innovation autocorrelation strongly negative here. Your model's memory is not this model's memory |
| `inflation_factor` | Tied to ensemble size and to how fast your ensemble collapses |
| `sm_bounds` | Physical, and specific to the variable and the soil |

Score candidates the way the tuning does, on innovation diagnostics as well as agreement
with the observation. Chasing agreement with the assimilated observation alone rewards a
filter that simply believes the observation, which is close to circular. `examples/quickstart.py` shows that effect in four
lines of output: as `obs_error_std` falls, apparent skill rises and the mean gain approaches
one.

---

## What goes wrong quietly

The failures that do not announce themselves, worth checking explicitly.

* **A gauge on the wrong reach.** Plausible hydrograph, wrong river, every metric wrong.
* **An observation file with the wrong number of days.** Runs for hours, then an end-of-file
  error deep in the model.
* **Missing-value sentinels assimilated as data.** The soil-moisture file's sentinel is
  greater than zero, so writing it without its two quality companions feeds the sentinel to
  the filter as if it were a measurement ([07_pitfalls.md](07_pitfalls.md)).
* **Uncorrected observations.** The filter chases an offset between instruments rather than
  the signal.
* **Inherited error parameters.** The filter runs, produces numbers, and those numbers are
  not about your basin.
* **The wrong control.** A `da = 0` run or a plain SWAT2012 run differs from the
  assimilating arms before any observation arrives, because those arms carry a perturbed
  ensemble. Compare every arm with the matched control (`--resolution control`) instead
  ([07_pitfalls.md](07_pitfalls.md)).

---

## Checklist before you trust a result

```
[ ] n_hru, n_subbasins, n_days read from a clean TxtInOut, not estimated
[ ] n_days agrees with what the model computes from file.cio
[ ] every gauge's reach confirmed against fig.fig or the shapefile
[ ] mean annual flow per gauge within reason against the published record
[ ] observations bias-corrected onto the model's own distribution
[ ] sm_bounds set to the variable's physical range
[ ] obs_error_std and model_error_std re-tuned, not inherited
[ ] evaluation windows inside both the simulation and the observation record
[ ] every arm, the control included, staged with da = 1 and the same sample.txt
[ ] assimilable HRU-day fraction plausible for your sensor and land cover
[ ] `pytest tests -q` still green, so nothing you changed broke the shipped contract
```

That last line matters. The tests pin the worked Mekong example, so if you have edited
only your own data directory they should still pass; if you have changed `config.py`, the
geometry and constant checks in `tests/test_claims.py` are expected to fail and can be
updated to your values.
