# Workflow, from raw data to scored results

This guide walks the whole pipeline in the order a new user runs it. Each step names the
command or function, the files it reads and writes, what to change for another basin, and a
quick check that the step worked.

## Start here

The repository is the data-assimilation framework of the paper, not its study archive. It
contains the code for every analysis step and one worked example; it does not contain the
SMAP archives, the calibrated SWAT project, gauge observations, model outputs or figure code.
Applying it to another basin means supplying those inputs ([03_data.md](03_data.md)).

Run the worked example once after installing ([02_install.md](02_install.md)):

```bash
mekong-da info        # environment, compiler and input inventory
mekong-da reproduce   # reruns the offline filter on the example and checks it
```

`reproduce` ends with `REPRODUCED` when the filter regenerates the expected example output
to within 1e-6 (in practice about 1e-16). Steps 4 and 8 below run on the example as shipped;
steps 1 to 3 and 5 to 7 need your own inputs.

Two experiments share the code and must not be confused. The **offline scalar filter**
(Python) assimilates a basin-average SMAP series into a basin-average SWAT soil-moisture
series after the model has run. The **coupled model** (Fortran, driven from Python) updates
soil water in every hydrologic response unit (HRU) inside the daily loop, so the analysis
propagates into discharge. [07_pitfalls.md](07_pitfalls.md) lists what goes wrong when the
two are mixed up.

The filter is also usable directly from Python, because it takes plain arrays:

```python
import pandas as pd
from mekong_da import EnKFConfig, metrics, run_enkf

frame = pd.read_csv("data/enkf/inputs_9km.csv", parse_dates=["date"], index_col="date")
result = run_enkf(
    open_loop=frame["baseline"].to_numpy(),                 # model, unassimilated
    observation=frame["observation_corrected"].to_numpy(),  # bias-corrected observation
    obs_available=frame["obs_available"].to_numpy(),        # True on days with a retrieval
    config=EnKFConfig(),                                    # the paper's settings
)
skill = metrics(result.analysis, frame["observation_corrected"].to_numpy())
print(result.n_updates, round(skill["RMSE"], 4), round(skill["NSE"], 3))
# 3447 0.0195 0.867
```

`metrics(prediction, reference)` takes its arguments in that order; reversing them leaves R
and RMSE unchanged and silently changes NSE, KGE and PBIAS.

---

## Data flow

Offline scalar filter:

```
SMAP granules                      SWAT open-loop run (yours)
     | step 2                            | basin-average soil moisture, m3/m3
     v                                   v
basin-mean SMAP series  ----->  biascorrect.cdf_match_rank        step 3
                                          v
                        data/enkf/inputs_<product>.csv
                                          v
                   mekong-da tune  /  mekong-da enkf               step 4
                                          v
             outputs/enkf/da_results_<product>.csv, outputs/tables/T1_*.csv
                                          v
                                    analyses/*.py                  step 8
```

Coupled model:

```
subbasin SMAP series + HRU-to-subbasin map + calibrated TxtInOut (yours)
                                          v
            mekong-da prepare-run, one directory per arm               step 5
     (observation files, sample.txt, input.da; control = no observations)
                                          v
       mekong-da build-fortran, run-fortran   (SWAT2012 + EnKF)        step 6
                                          v
                 output.rch  ->  mekong-da harvest                     step 6
                                          v
     pipeline.build_merged_streamflow + your gauge records             step 7
                                          v
           data/streamflow/<Station>_merged_2024.csv -> T2 table
```

---

## Step 1. Set up and calibrate SWAT2012

Build and calibrate a SWAT2012 project for your basin with your usual tools; this happens
outside the repository. The coupled build in `fortran/src/` is SWAT2012 revision 629, so the
project must be a SWAT2012 `TxtInOut` (not SWAT+). Record its geometry: HRU count, subbasin
count, first simulated year, number of years and warm-up years (`NYSKIP` in `file.cio`).

**Change:** create a `SwatRunSpec` with those values ([08_your_own_basin.md](08_your_own_basin.md)).
**Check:** the uncoupled project runs and its `output.rch` covers the expected days.

## Step 2. Extract SMAP onto the model's units

Average each SMAP product onto your subbasins, one value per subbasin per day, and onto the
basin for the offline filter.

* Subbasin series: `preprocessing/R/` documents the extraction used for the paper (R,
  GPL-3.0; edit the hard-coded paths at the top of each script). The coupled model expects
  `data/smap_subbasin_daily/smap_<product>_subbasin_daily.csv`, indexed by `date`, one
  column per subbasin named `sub_<id>`, in m3/m3.
* Basin-mean series: `python scripts/extract_basin_mean.py --archive 9km=/path/to/file.nc`
  (repeat `--archive` per product; needs `pip install -e ".[raw]"`). It writes
  `smap_<product>_basin_mean_daily.csv`.

**Change:** the bounding box `LAT_MIN`, `LAT_MAX`, `LON_MIN`, `LON_MAX` at the top of
`scripts/extract_basin_mean.py`, and the shapefile and paths in the R scripts.
**Check:** values lie within 0.01 to 0.60 m3/m3 and the date range matches the products.

## Step 3. Bias-correct the retrievals (CDF matching)

Satellite and model soil moisture are different quantities, so each retrieval is mapped onto
the model's own distribution by empirical rank before assimilation:

```python
from mekong_da.biascorrect import cdf_match_rank
corrected = cdf_match_rank(obs, model, train_mask=dates <= "2019-12-31")
```

Fit the mapping on a training window (the paper used 2015 to 2019) and apply it forward
without refitting. Then write one input file per product with the columns
`date, baseline, observation, observation_corrected, obs_available` (the format of
`data/enkf/inputs_9km.csv`; `baseline` is the open-loop basin soil moisture, `obs_available`
is 1 on days with a screened retrieval).

**Check:** `observation_corrected` and `baseline` have similar means and spreads over the
training window.

## Step 4. Tune and run the offline scalar filter

```bash
mekong-da tune --resolution 9km     # two-stage parameter search, writes outputs/enkf_tuning/
mekong-da enkf                      # runs every product, writes outputs/enkf/da_results_<product>.csv
mekong-da enkf --config outputs/enkf_tuning/optimal_config.json   # with your tuned settings
```

`tune` reads `data/enkf_tuning/inputs/` (an open-loop series `swat_baseline_smap_2015_2020.csv`
with column `sm_baseline`, and one `smap_<product>_basin_mean_daily.csv` per product).
[05_enkf_and_tuning.md](05_enkf_and_tuning.md) explains the filter, the search and the score.

**Change:** replace the example files in `data/enkf/` and `data/enkf_tuning/inputs/` with
yours, and `RESOLUTIONS` in `src/mekong_da/config.py` if your products differ. Remove the
example's `data/enkf/reference_*.csv` and `data/reference_metrics/` at the same time, or
better, keep the example intact and point the commands at your own folder with
`--data-dir` (or `MEKONG_DA_DATA`); otherwise your run is compared against the Mekong
expected output. Re-tune; the Mekong settings do not transfer.
**Check:** the run prints, per product, the number of assimilated days and the root mean
square error (RMSE) against the assimilated observation before and after assimilation. That
reduction measures agreement with the input, not independent accuracy.

## Step 5. Stage the coupled runs

Each arm runs in its own directory, staged from a copy of the calibrated project:

```bash
for arm in control 9km 1km 400m; do
  mekong-da prepare-run --run-dir runs/$arm --resolution $arm --template /path/to/TxtInOut
done
```

For a product, `prepare-run` expands the subbasin series onto the HRUs with
`data/hru_subbasin_mapping.csv` (columns `hru_index`, `subbasin`, `hru_in_sub`) and writes
the three observation files. For `control` it writes observation files that contain only
missing-value sentinels. Every arm then gets the same `sample.txt` (200 members, seed 42) and
an `input.da` with `da = 1`, so all four arms share the ensemble, initial states, forcing and
random draws and differ only in whether observations are present. That matched control is
the comparison the paper uses; `da = 0` is a different, deterministic run
([04_coupled_model.md](04_coupled_model.md)).

**Change:** `--members`, and any `input.da` setting through `InputDA` when calling
`pipeline.prepare_fortran_run` from Python.
**Check:** `prepare-run` prints how many HRU-days carry an assimilable retrieval (zero for
the control) and the `input.da` settings it wrote.

## Step 6. Build, run and harvest the model

```bash
mekong-da build-fortran                              # needs gfortran; about a minute
mekong-da run-fortran --run-dir runs/9km --timeout-hours 12
mekong-da harvest --rch runs/9km/output.rch --out results/9km_hydrographs.csv
```

Arms are independent and can run in parallel. Each run takes hours on a large basin and
`run-fortran` refuses to start below `--min-free-gb` (default 60) of free disk.

**Change:** `STATIONS` in `src/mekong_da/config.py` (gauge name and the SWAT reach that
carries it) before harvesting.
**Check:** `run-fortran` reports exit code 0; the harvested file has one column per gauge
and one row per simulated day after the warm-up years.

## Step 7. Score the arms against gauge observations

Merge each gauge's harvested series with its observations, then score:

```python
from mekong_da.pipeline import build_merged_streamflow, streamflow_table

frame = build_merged_streamflow(
    "Kratie",
    open_loop=control["Kratie"],                       # the matched control arm
    da_by_resolution={"9km": arm9["Kratie"], "1km": arm1["Kratie"]},
    observed=gauge_series,                             # your observations, m3/s
)
frame.to_csv("data/streamflow/Kratie_merged_2024.csv")
table = streamflow_table()                             # every gauge, arm and window
```

Once the merged files exist, `mekong-da reproduce` also writes `outputs/tables/T2_streamflow_metrics.csv`.

**Change:** `STREAMFLOW_PERIODS` in `src/mekong_da/config.py` for your evaluation windows.
**Check:** each window has the number of observed days you expect (column `N`).

## Step 8. Run the diagnostic analyses

`analyses/` holds the sampling, error-specification and sensitivity analyses behind the
paper. They run on the step 4 inputs and write tables to `outputs/analyses/`. Run order,
dependencies and runtimes are in [06_analyses.md](06_analyses.md).

```bash
pip install -e ".[analyses]"
python analyses/A_temporal_subsampling.py
```
