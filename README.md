# SWAT-SMAP ensemble Kalman filter data assimilation framework

This repository contains the data-assimilation framework behind the study below: a Python
ensemble Kalman filter (EnKF) for satellite soil moisture, and a Python driver for a
Fortran SWAT2012 build in which the same kind of filter updates soil-water storage inside
the hydrological time loop. It is written so that others can rebuild the workflow on their
own basin, satellite product and SWAT project. Study data, model outputs and figure code are
not included; [docs/03_data.md](docs/03_data.md) lists where every input comes from.

## The paper

**Relative roles of spatial resolution and temporal coverage in basin-average SMAP soil
moisture assimilation over the Mekong River Basin**

Sebastian R.O. Marshall, Duc Tran, Binh Quang Nguyen, Manh-Hung Le, Bin Fang, Hesham
Elhaddad, John Eylander, Sameh A. Kantoush, John D. Bolten and Venkataraman Lakshmi

*Journal of Hydrology*, in revision. Corresponding author: Sebastian R.O. Marshall
(smarsh46@jh.edu).

### Abstract

Satellite soil moisture offers a basin-wide constraint on the water state of large monsoon
basins, where ground networks are sparse and cloud interrupts optical and thermal
observation. Downscaled Soil Moisture Active Passive (SMAP) products add spatial detail but
inherit that cloud dependence, raising the question of how resolution and temporal coverage
each contribute to assimilation. This study examined it over the Mekong River Basin using
9 km, 1 km, and 400 m products from 2015 to 2024. A 200-member ensemble Kalman filter
assimilated each product as a basin-average composite, isolating temporal sampling, while a
separate coupled experiment updated soil storage in the Soil and Water Assessment Tool
(SWAT) and evaluated discharge at five gauges against a matched control. Assimilation
reduced discrepancies with each assimilated input by 58.9–60.6%, the 9 km product retaining
the smallest residual under common error settings. Imposing the finer products' schedules
on the 9 km record left shared-day scores within sampling uncertainty but raised
full-calendar error by 0.0036 and 0.0048 m³/m³, because the omitted 8.3% and 11.7% of days
contributed 35.8% and 44.3% of squared error. Product-specific error calibration reversed
the ranking, showing that coverage alone did not determine it. Within SWAT, 1 km
assimilation produced the largest discharge gains, resolved at the two downstream gauges,
raising Nash–Sutcliffe efficiency at Kratié from 0.621 to 0.706 as monsoon storage bounds
converted proposed water additions into a net removal that reduced surface runoff and
groundwater flow. Agreement with assimilated observations and hydrological benefit
therefore ranked the products differently.

---

## The framework in brief

The study runs two assimilation experiments, and the code keeps them apart because they
answer different questions.

1. **Offline scalar filter (Python).** SMAP retrievals are averaged over the basin,
   mapped onto the model's soil-moisture climatology by cumulative distribution function
   (CDF) matching, and assimilated into the SWAT basin-average soil-moisture series with a
   200-member stochastic EnKF. The state is a single number per day, so the experiment
   isolates how often a product is observed from how finely it resolves the basin. It runs in
   seconds. Its filter settings come from a two-stage parameter search that is part of the
   code.
2. **Coupled SWAT-DA model (Fortran, driven from Python).** An EnKF inside SWAT2012
   revision 629 updates two soil-layer storages in every hydrologic response unit (HRU) on
   days with a screened retrieval, so the analysis propagates through the water balance into
   routed discharge. Volumetric retrievals are converted to plant-available storage with the
   layer depth, and each updated member is bounded between zero and saturation storage.
   Python writes the five input files the model reads, compiles and runs it, and extracts
   the gauge hydrographs. Every arm, including the matched control, runs the same ensemble;
   the control simply receives no observations.

The diagnostic analyses behind the paper's sampling, error-specification and sensitivity
results are in `analyses/` and run on the filter in `src/mekong_da/`.

## Workflow, from raw data to results

| Step | What happens | Code | You supply |
|---|---|---|---|
| 1 | Set up and calibrate SWAT2012 for your basin | outside this repository | the SWAT project (`TxtInOut`) |
| 2 | Extract SMAP onto your subbasins, one value per subbasin per day | `preprocessing/R/`, `scripts/extract_basin_mean.py` | SMAP granules |
| 3 | Bias-correct retrievals onto the model climatology (CDF matching) | `mekong_da.biascorrect` | the open-loop soil-moisture series |
| 4 | Tune and run the offline scalar EnKF | `mekong-da tune`, `mekong-da enkf` | a `data/enkf/inputs_<product>.csv` per product |
| 5 | Write the coupled model's DA inputs and stage each arm | `mekong-da prepare-run` | the subbasin series and an HRU-to-subbasin map |
| 6 | Compile and run SWAT-DA, then extract the gauge hydrographs | `mekong-da build-fortran`, `run-fortran`, `harvest` | disk space and hours of compute |
| 7 | Score each arm against gauge observations | `pipeline.build_merged_streamflow`, `pipeline.streamflow_table` | gauge records |
| 8 | Run the diagnostic analyses | `analyses/` | the step 4 inputs |

[docs/01_workflow.md](docs/01_workflow.md) walks through each step with the exact commands,
file formats and checks.

## Quick start

```bash
git clone https://github.com/rushmarshall/mekong-swat-smap-enkf.git
cd mekong-swat-smap-enkf
pip install -e ".[dev]"            # or: conda env create -f environment.yml

mekong-da info                      # environment, compiler and input inventory
mekong-da reproduce                 # runs the worked example and checks it
pytest tests                        # 123 tests, Fortran build included
```

The repository ships one worked example: the basin-average inputs of the offline filter for
the three Mekong products, with the expected filter output. `mekong-da reproduce` reruns the
filter on them and confirms the published reductions of 58.9%, 60.5% and 60.6% (9 km, 1 km,
400 m) to machine precision. Only NumPy and pandas are required for the Python half; the
model half needs `gfortran`.

## What to change for your own application

| To change | Where |
|---|---|
| Model geometry and calendar (HRU and subbasin counts, start year, years, warm-up) | `SwatRunSpec` in `src/mekong_da/config.py` |
| Gauges and the SWAT reach that carries each | `STATIONS` in `src/mekong_da/config.py` |
| Satellite products assimilated | `RESOLUTIONS` in `src/mekong_da/config.py`, plus one input series per product |
| Offline filter settings (ensemble size, errors, inflation, persistence) | `EnKFConfig`, or rerun `mekong-da tune` on your series |
| Coupled filter settings (ensemble size, perturbations, stop day, diagnostics) | `InputDA` in `src/mekong_da/inputda.py`, written to `input.da` |
| Observation error and storage bounds inside the model | `fortran/src/init_da.f90` and `fortran/src/EnKF.f90` |
| Evaluation windows | `SM_PERIODS` and `STREAMFLOW_PERIODS` in `src/mekong_da/config.py` |

The filter settings were tuned for the Mekong series and do not transfer between basins.
[docs/08_your_own_basin.md](docs/08_your_own_basin.md) covers what must be re-tuned and the
checks to run before trusting a result, and [docs/07_pitfalls.md](docs/07_pitfalls.md) lists
the mistakes this code base invites.

---

## Repository layout

```
src/mekong_da/        The package. Eleven modules, each one stage of the pipeline.
  config.py           Fixed settings in one place: filter, model geometry, gauges, windows.
  enkf.py             The offline scalar EnKF.
  biascorrect.py      CDF matching of satellite retrievals onto the model climatology.
  tuning.py           The two-stage parameter search for the offline filter.
  metrics.py          The metric set used for every skill table.
  inputda.py          Reader and writer for the model's input.da control file.
  obsfiles.py         Writers for the observation and perturbation files the model reads.
  fortran.py          Compile and run wrappers for the model.
  rch.py              Streaming parser for SWAT's output.rch.
  pipeline.py         Orchestration of both experiments.
  cli.py              The mekong-da command.
fortran/src/          SWAT2012 revision 629 with the EnKF coupling, 324 translation units.
analyses/             Diagnostic analyses of sampling, error specification and sensitivity.
preprocessing/R/      SMAP extraction and bias-correction scripts (GPL-3.0).
scripts/              Basin-mean extraction from raw SMAP NetCDF archives.
data/                 The worked example: offline filter inputs, expected output, tuning grids.
examples/             A short script that runs the filter and explains its output.
tests/                The test suite, including the compiled operator check.
docs/                 Eight documents, listed below.
```

---

## Documentation

| Document | Covers |
|---|---|
| [docs/01_workflow.md](docs/01_workflow.md) | The full pipeline, step by step, with commands and checks |
| [docs/02_install.md](docs/02_install.md) | Python environment and Fortran toolchain |
| [docs/03_data.md](docs/03_data.md) | Every input, its source and the file formats the code expects |
| [docs/04_coupled_model.md](docs/04_coupled_model.md) | The SWAT-DA model: input files, `input.da`, the observation update, outputs |
| [docs/05_enkf_and_tuning.md](docs/05_enkf_and_tuning.md) | The offline filter equations, every parameter, and the tuning |
| [docs/06_analyses.md](docs/06_analyses.md) | What each diagnostic analysis computes and how to run it |
| [docs/07_pitfalls.md](docs/07_pitfalls.md) | Mistakes that produce plausible but wrong results |
| [docs/08_your_own_basin.md](docs/08_your_own_basin.md) | Applying the framework to another basin or product |

## Acknowledgements and licences

The coupled model builds on SWAT2012 (Arnold et al., 1998; Texas A&M AgriLife Research and
USDA Agricultural Research Service), the EnKF coupling of Patil and Ramsankaran (2017,
https://github.com/amolpatil771/SWAT_DA) and its SMAP adaptation by Le et al. (2022,
https://github.com/mhle510/smap_enkf_swat). [NOTICE.md](NOTICE.md) records which parts come
from where and the terms that apply. Code written for this study is MIT licensed
([LICENSE](LICENSE)); `fortran/src/` keeps the terms of SWAT2012 and the upstream coupling, and
`preprocessing/R/` is GPL-3.0.

## Citation

If you use this framework, please cite the paper:

> Marshall, S.R.O., Tran, D., Nguyen, B.Q., Le, M.-H., Fang, B., Elhaddad, H., Eylander, J.,
> Kantoush, S.A., Bolten, J.D., Lakshmi, V. Relative roles of spatial resolution and temporal
> coverage in basin-average SMAP soil moisture assimilation over the Mekong River Basin.
> *Journal of Hydrology*, in revision.

```bibtex
@article{marshall_mekong_smap_da,
  author  = {Marshall, Sebastian R. O. and Tran, Duc and Nguyen, Binh Quang and Le, Manh-Hung
             and Fang, Bin and Elhaddad, Hesham and Eylander, John and Kantoush, Sameh A.
             and Bolten, John D. and Lakshmi, Venkataraman},
  title   = {Relative roles of spatial resolution and temporal coverage in basin-average
             {SMAP} soil moisture assimilation over the {Mekong River Basin}},
  journal = {Journal of Hydrology},
  note    = {In revision}
}
```

The methods this framework builds on are Patil, A., Ramsankaran, R.A.A.J. (2017), *Journal of Hydrology*
555, 683-696, https://doi.org/10.1016/j.jhydrol.2017.10.058, and Le, M.-H. et al. (2022),
*Remote Sensing* 14(7), 1607, https://doi.org/10.3390/rs14071607. [CITATION.cff](CITATION.cff)
carries the same information in machine-readable form.
