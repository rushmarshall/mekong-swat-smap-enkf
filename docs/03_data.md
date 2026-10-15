# Data: what ships, what you supply, and the formats

The repository ships one worked example and no study data. Every other input comes from its
public source or from your own model, in the formats below. Paths are relative to the data
directory, `data/` by default (`--data-dir` or `MEKONG_DA_DATA` point elsewhere).

---

## Shipped with the repository: the worked example

Basin-average series for the Mekong River Basin, 2015 to 2024, enough to run the offline
filter, its tuning and the diagnostic analyses, plus the expected output used to check them.
About 1.4 MB in total.

| File | Columns | Purpose |
|---|---|---|
| `enkf/inputs_{9km,1km,400m}.csv` | `date, baseline, observation, observation_corrected, obs_available` | Offline filter input per product |
| `enkf/reference_{9km,1km,400m}.csv` | `date, analysis, uncertainty` | Expected filter output (analysis mean and ensemble spread) |
| `enkf_tuning/inputs/swat_baseline_smap_2015_2020.csv` | `date, ..., sm_baseline, ...` | Open-loop soil moisture for the parameter search |
| `enkf_tuning/inputs/smap_{9km,1km,400m}_basin_mean_daily.csv` | `date, smap_<product>_basin_mean` | Raw basin-mean SMAP for the search |
| `enkf_tuning/stage1a_*.csv`, `stage1b_*.csv`, `validation_*.csv` | grid diagnostics | Expected results of `mekong-da tune` |
| `enkf_tuning/optimal_config.json`, `enkf_tuning/README.md` | | The selected settings and a description of the search |
| `reference_metrics/T1_soil_moisture_metrics.csv` | `Resolution, Period, Model, N, RMSE, ...` | Expected soil-moisture skill table |

Column meanings in the filter inputs: `baseline` is the SWAT open-loop basin-average
volumetric soil moisture (m3/m3, no gaps); `observation` is the raw basin-average SMAP
retrieval; `observation_corrected` is that retrieval after cumulative distribution function
(CDF) matching onto the model climatology; `obs_available` marks days with a retrieval that
passed screening. The baseline comes from the study's SWAT simulation and the observations
from the public SMAP products listed below.

Keep the example unchanged and put your own inputs in another folder passed with
`--data-dir`; the test suite and `mekong-da reproduce` check the example against its expected
output.

---

## What you supply

| Input | Where it goes | Source |
|---|---|---|
| SMAP L3 Enhanced, 9 km | extracted to the series below | NSIDC, SPL3SMP_E, https://nsidc.org/data/spl3smp_e |
| SMAP downscaled, 1 km | extracted to the series below | https://doi.org/10.5067/U8QZ2AXE5V7B |
| SMAP downscaled, 400 m | extracted to the series below | data https://doi.org/10.18130/V3/IVOU1T, code https://doi.org/10.18130/V3/A9UJAV |
| Calibrated SWAT2012 project | `--template` of `prepare-run` | your own model |
| Precipitation forcing | inside the SWAT project | the paper used IMERG (NASA GES DISC) |
| Terrain and soils | inside the SWAT project | the paper used MERIT DEM and HWSD v1.2 |
| Gauge discharge | step 7 of the workflow | the paper used Mekong River Commission data, https://portal.mrcmekong.org/, under the Commission's terms |

Satellite and gauge products are not redistributed here; obtain them under their own terms.

### Subbasin SMAP series (coupled model)

`smap_subbasin_daily/smap_<product>_subbasin_daily.csv`, one file per product:

* index column `date` (daily, ISO format);
* one column per subbasin named `sub_<id>`, where `<id>` matches the subbasin numbers in the
  HRU map;
* volumetric soil moisture in m3/m3, empty where there is no retrieval.

`prepare-run` drops values outside 0.01 to 0.60 m3/m3, keeps only days inside the model
calendar, and gives every HRU the value of its subbasin.

### HRU-to-subbasin map (coupled model)

`hru_subbasin_mapping.csv` with columns `hru_index` (1 to the HRU count, in the model's HRU
order), `subbasin` and `hru_in_sub`. It must list every HRU exactly once; build it from your
project's `.hru` files or its SWAT database.

### Basin-average filter inputs (offline filter)

`enkf/inputs_<product>.csv` with the five columns above, one row per day and no gaps in
`baseline`. Produce `observation_corrected` with `mekong_da.biascorrect.cdf_match_rank`,
fitting on a training window ([01_workflow.md](01_workflow.md), step 3).

### Tuning inputs (offline filter)

`enkf_tuning/inputs/swat_baseline_smap_2015_2020.csv` with at least `date` and
`sm_baseline`, and `enkf_tuning/inputs/smap_<product>_basin_mean_daily.csv` with `date` and
one value column. The file name of the baseline is fixed in `mekong-da tune`; its period can
be any common window.

### Merged streamflow (scoring)

`streamflow/<Station>_merged_2024.csv`, written by `pipeline.build_merged_streamflow`, with
columns `date, swat_baseline, mrc_obs, swat_da_<product>...`: the matched control, the gauge
observation (m3/s) and one column per assimilating arm. The `<Station>` key and the reach
come from `STATIONS` in `src/mekong_da/config.py`; the column name `mrc_obs` is kept for any
gauge network. These files, the subbasin series and the HRU map are listed in `.gitignore` so
they are never committed by accident.

---

## Rebuilding the inputs from raw data

**Basin-mean series.** `scripts/extract_basin_mean.py` reduces a SMAP NetCDF archive to a
daily basin mean over a latitude and longitude box, discarding fill and values outside the
physical range:

```bash
pip install -e ".[raw]"
python scripts/extract_basin_mean.py --archive 9km=/path/to/SPL3SMP_E.nc --archive 1km=/path/to/1km.nc
```

Edit `LAT_MIN`, `LAT_MAX`, `LON_MIN` and `LON_MAX` at the top of the script for another
basin. The Mekong archives were 84 GB (9 km), 9.5 GB (1 km) and 26 GB (400 m); a pass takes
minutes to tens of minutes per product.

**Subbasin series and observation screening.** The R scripts in `preprocessing/R/`
(GPL-3.0, from the `smap_enkf_swat` workflow of Le et al., 2022) extract SMAP over subbasin
polygons and bias-correct it. They carry hard-coded paths from their original environment
(`source('C:/Users/...')`); edit those and the shapefile paths before use. They are kept for
provenance; nothing in `src/` depends on them.
