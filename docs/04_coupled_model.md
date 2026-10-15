# The coupled SWAT-DA model

`fortran/src/` is SWAT2012 revision 629 with an ensemble Kalman filter (EnKF) built into the
daily time loop, following Patil and Ramsankaran (2017) and its SMAP adaptation by Le et al.
(2022), with the corrections described below. On every day with a screened retrieval, the
filter updates the soil-water storage of each hydrologic response unit (HRU), and the next
day's hydrology starts from the updated soil. That feedback into runoff, percolation and
routed discharge is what the coupled experiment measures, and it cannot be imitated by
post-processing model output. Python stages the run, writes the inputs, starts the binary and
reads the output; it does not compute the hydrology or the in-model filter.

---

## Inputs the model reads

Beyond a standard SWAT2012 `TxtInOut`, the build reads five files from the run directory.
`mekong-da prepare-run` writes all five.

| File | Shape | Content |
|---|---|---|
| `sm_ad_mn_std_monthly_marol.txt` | days x HRUs | surface soil moisture, m3/m3; `0.001` where missing |
| `smdqx_ad_marol.txt` | days x HRUs | retrieval quality flag; `0.01` valid, `1.0` missing |
| `smrfi_ad_marol.txt` | days x HRUs | interference flag; `0.01` valid, `1.0` missing |
| `sample.txt` | (days x members) x 6 | standard normal perturbation draws |
| `input.da` | 22 lines | assimilation settings, below |

The names are opened literally by the Fortran and keep the upstream SMOS names. Rows must
equal the simulated day count and columns the HRU count; the files are read list-directed.
A retrieval is used on an HRU-day only when `soil moisture > 0`, `RFI < 0.4` and `DQX < 0.1`
(`simulate.f`, line 310). The missing soil-moisture sentinel 0.001 passes the first test on
its own; the quality sentinels are what stop the update, so always write all three files
together (`pipeline.prepare_fortran_run` does). Observations are written in volumetric units;
no CDF matching is applied in the coupled experiment, because the operator below converts
them to storage.

### `input.da`

Written by `mekong_da.inputda.InputDA`, one fixed-width value per line in this order:

| Field | Default | Meaning |
|---|---|---|
| `da` | 1 | 1 runs the ensemble and permits updates; 0 runs the deterministic branch (see below) |
| `pcp_std_par` | 0.50 | precipitation perturbation coefficient, fraction of daily depth |
| `fcpercent`, `fcpercent2`, `fcpercent3` | 0.50 | legacy field-capacity controls, kept for the file format |
| `sm_dqx_co`, `sm_rfi_co` | 1.00 | legacy quality-index error multipliers, kept for the file format |
| `sm_std1`, `sm_std2`, `sm_std3` | 0.04 | forecast error standard deviation by soil layer |
| `cn_std` | 0.10 | curve-number perturbation standard deviation |
| `n_en_member` | 200 | ensemble size; must match `sample.txt` |
| `n_ensembles` | 1 | number of ensembles |
| `n_parameters` | 6 | perturbed parameters; the column count of `sample.txt` |
| `n_sampling_method` | 3 | 1 Sobol, 2 Halton, 3 Latin hypercube, 4 random |
| `sol_st_std` | 0.1 | spread of the initial soil-storage ensemble |
| `no_sta_var` | 2 | state length: two soil layers |
| `no_obs_var` | 1 | observation length: one surface value |
| `wrdtf` | 0 | 1 writes per-step filter diagnostics for HRU 1 (`KalmanGain.dat`, `correlation.dat`) |
| `da_stopday` | 99999 | simulation day after which updates stop (forecast experiments) |
| `da_alg` | 1 | 1 EnKF, 2 the alternative update in `SMAR_DA` |
| `smsat` | 1 | 1 the SMOS branch, used here for SMAP; 2 the ASCAT branch |

`prepare-run` sets `da = 1` and `n_en_member` to `--members` for every arm. To change other
fields, call `pipeline.prepare_fortran_run(..., input_da=InputDA(...))` from Python, or edit
the file and read it back with `InputDA.read` to check it.

---

## The observation update

For each HRU that passes screening on a given day, `simulate.f` (lines 295 to 340) calls
`init_da`, `EnKF` and `post_da`:

1. **Operator** (`init_da.f90`). The volumetric retrieval is converted to plant-available
   water in the top soil layer, in millimetres: `theta * sol_z(1) - sol_wpmm(1)`, the layer
   depth times the retrieval minus the water held below wilting point. The
   observation-error standard deviation is `0.04 * sol_z(1)`, the same 0.04 m3/m3 carried
   through the same depth. The quality flags decide eligibility only.
2. **Gain and update** (`st_error`, `obs_error`, `st_cov`, `obs_cov`, `st_obs`,
   `Kalman_Gain_calc.f90`, `EnKF.f90`). The ensemble covariance of the two layer storages
   with the surface storage gives the gain, and each of the 200 members is updated with its
   perturbed observation.
3. **Bounds** (`EnKF.f90`). Each updated member is limited to between zero and `sol_ul`,
   the saturation storage above wilting point of that layer. The raw and bounded mean
   increments and the number of members clipped at each bound are recorded.
4. **Install** (`post_da.f90`). The bounded members replace the ensemble storages, the HRU
   storage becomes their mean, and the profile water content is recomputed.

Two audit files are written in the run directory: `analysis_mass_daily.csv` (per day:
updated HRUs, area-weighted soil water before and after the analysis, its change, raw and
bounded increments, clip counts) and `operator_sample.csv` (the first updated HRU of each day:
forecast, observation, its error, gain, analysis and increments). Together they show how much water each
update added or removed, which the paper uses to explain the discharge response.

`tests/test_corrected_operator.py` pins the five files that carry these changes (`EnKF.f90`,
`init_da.f90`, `modparm.f`, `readfile.f`, `simulate.f`) by SHA-256, and compiles the update
routines against a controlled state in `tests/fixtures/` to check the unit conversion, error
variance, increment, bounds and installed water total. A deliberate change to any of those
files should update the pinned hash in the test and say why.

---

## The matched control, and why `da = 0` is something else

The four arms of the experiment (control, 9 km, 1 km, 400 m) all run with `da = 1`, the same
`sample.txt`, the same initial states and forcing. The control differs only in its
observation files, which contain nothing but missing-value sentinels, so no update ever
passes screening. Any difference between an arm and the control is then due to the
assimilated observations alone.

`da = 0` is not that control. With `da = 0`, eight routines (`etact.f`, `percmain.f`,
`percmicro.f`, `sat_excess.f`, `subbasin.f`, `surface.f`, `surq_daycn.f`, `swu.f`) take their
deterministic branch and `main.f` never calls `readsm` to read observations, so the run is a
single unperturbed simulation rather than the ensemble the arms use. Comparing an
assimilating arm with a `da = 0` run mixes the effect of the ensemble with the effect of the
observations.

---

## Running the four arms

```bash
mekong-da build-fortran
for arm in control 9km 1km 400m; do
  mekong-da prepare-run --run-dir runs/$arm --resolution $arm --template /path/to/TxtInOut
  mekong-da run-fortran --run-dir runs/$arm --timeout-hours 12
  mekong-da harvest --rch runs/$arm/output.rch --out results/$arm.csv
done
```

`prepare-run` copies the project into the run directory, skipping stale model outputs, and
needs `data/smap_subbasin_daily/` and `data/hru_subbasin_mapping.csv` for a product arm
([03_data.md](03_data.md)). Use `--members` to change the ensemble size. Arms are independent
and can run at the same time in separate directories.

`run-fortran` copies the binary into the run directory, launches it by absolute path, and
stops with an error if free disk falls below `--min-free-gb` (default 60) at the start or
below 5 GB during the run. `harvest` reads discharge (`FLOW_OUTcms`) at the reaches named in
`STATIONS` from `output.rch`, one row per simulated day after the warm-up years.

### Outputs that are discarded

To keep runs small, the shipped `readfile.f` opens `/dev/null` in place of every `output*`
file except `output.rch` and `output.std` (for example `output.hru`, `output.sub`,
`output.sed`, `output.rsv`, `output.wtr`). The framework reads only `output.rch`. To keep any
of the others, restore its file name in the corresponding `open` statement in `readfile.f`.
`/dev/null` exists only on POSIX systems; on Windows, change these statements to `NUL` or to
file names before running.

---

## Building by hand

`mekong-da build-fortran` compiles with the flags below. Without the Python wrapper, the
sequence is:

```bash
cd fortran/src
mkdir -p ../bin
FLAGS="-O2 -fno-automatic -fallow-argument-mismatch -std=legacy"
LONG="-ffixed-form -ffixed-line-length-132"
SHORT="-ffixed-form"                  # grow.f and tran.f only: see the note below
gfortran $FLAGS $LONG -c modparm.f -o modparm.o
for f in *.f; do
  [ "$f" = modparm.f ] && continue
  case "$f" in grow.f|tran.f) WIDTH="$SHORT" ;; *) WIDTH="$LONG" ;; esac
  gfortran $FLAGS $WIDTH -c "$f" -o "${f%.f}.o"
done
for f in *.f90; do gfortran $FLAGS -ffree-form -c "$f" -o "${f%.f90}.o"; done
rm -f modparm.o                       # main.f already includes it; linking both duplicates symbols
gfortran $FLAGS -o ../bin/swat_da *.o
```

The `case` line is the part that is easy to drop. `grow.f` and `tran.f` carry code past
column 72 that only parses at the default line length, so compiling them with
`-ffixed-line-length-132` fails with `Unclassifiable statement`; the link then reports
`_tran_` and `_plantmod_` as undefined. `mekong_da.fortran.SHORT_LINE_SOURCES` is the same
exception in code. [07_pitfalls.md](07_pitfalls.md) explains why `-fno-automatic` and the
other flags must stay.
