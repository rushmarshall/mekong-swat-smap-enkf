# Pitfalls

Every entry below is a mistake this code base invites, and most were made at least once
while the study was built. Each says what breaks, why it is easy to miss, and what to do
instead. The entries are ordered roughly by how much damage they do and how quietly.

---

## 1. Do not vectorise the ensemble update loop

```python
for i in range(N):                     # <- this loop
    x[i] = x[i] + K * (observation[t] + np.random.normal(0, obs_error) - x[i])
```

`src/mekong_da/enkf.py` contains a Python loop over 200 ensemble members inside a loop
over 3,563 days. It is the slowest thing in the package and the obvious first target for
anyone optimising it. The protected property is the draw count and order, not the loop
itself: the vectorised form `x += K * (obs + np.random.normal(0, obs_error, N) - x)` was
measured to reproduce `data/enkf/reference_9km.csv` to 9.7e-17, because the legacy
`RandomState` gaussian sampler consumes the stream identically whether it is asked for N
values at once or N times for one (`tests/test_enkf.py` pins that). What does change the
output is adding, dropping or reordering a draw, changing `ddof`, or moving the clip, and
nothing warns you: the output still looks like a soil-moisture series with plausible skill.
Leave the reference loop in place; it is the form the expected output was produced with.

The same applies to the seed, to `np.var` staying at its default `ddof=0`, to the clip
sitting after the update rather than before it, and to the order of the forecast and
analysis steps. Any of these silently rewrites every number.

**Instead:** leave it alone. The whole filter runs in under twenty seconds for all three
resolutions. If it must be faster, keep the current implementation as the reference,
add the fast path beside it, and make `tests/test_enkf.py` prove they agree to 1e-12
before the fast path is used for anything published.

**How you find out you broke it:** `pytest tests/test_enkf.py` fails on
`test_reproduces_archived_analysis`.

---

## 2. Do not conflate the two assimilation systems

There are two filters in this study and they are not the same filter.

|  | Soil-moisture experiment | Streamflow experiment |
|---|---|---|
| Language | Python, `src/mekong_da/enkf.py` | Fortran, `fortran/src/EnKF.f90` |
| State | one basin-average value | two soil layers, per hydrologic response unit |
| Ensemble | perturbs the state only | perturbs precipitation, curve number and soil storage as well |
| Feedback | none: forecast nudges toward the open loop | full: the analysis re-enters the water balance |
| Runtime | seconds | hours |
| Runs from this repository | yes, on the worked example | needs your calibrated SWAT project |

A statement about "the EnKF" in this study is ambiguous until it says which. In
particular the Python filter's Kalman gain (mean 0.533 at 9 km) and the Fortran filter's
per-HRU gain in `KalmanGain.dat` are different quantities and should never be compared or
averaged together.

---

## 3. Do not trust a column because of its name

An innovation is the observation minus the filter's **forecast mean** on an update day. It
is easy to build a column called `innovation` as

```python
df["innovation"] = df["observation_corrected"] - df["baseline"]
```

but `baseline` is the **open-loop series**, not the filter's prior, so that column is a
model-error series. A "gain" derived from it as `update / innovation` is not a gain either:
on the study's series such a ratio ranged over thousands in both directions, with a fifth of
its values outside `[0, 1)`, which no Kalman gain can occupy. Conclusions about filter
behaviour drawn from such columns are wrong.

`enkf.py` records the prior mean before each update, so `innovation` here is
`observation - forecast_mean` and `kalman_gain` is the gain that was actually applied.
`test_gain_is_a_probability` asserts the gain stays inside the unit interval, and
`test_true_innovation_diagnostics` pins the lag-1 autocorrelation of the true innovation on
the example at +0.045 (9 km), +0.067 (1 km) and +0.132 (400 m).

Lag-1 autocorrelation near zero is not a whiteness test. Over twelve lags the same 9 km
series has a lag-8 peak of 0.704, and Ljung-Box rejects whiteness at every resolution,
which is the orbital revisit cycle aliasing into the residuals. Say "near-white at lag 1"
and stop there.

**Standing rule:** read the line that creates a diagnostic column before quoting its
statistic.

---

## 4. Do not use a `da = 0` run as the control for an assimilation experiment

With `da = 1` the model perturbs precipitation, curve number and soil storage every day and
carries 200 members whether or not an observation arrives; with `da = 0` it runs a single
deterministic simulation ([04_coupled_model.md](04_coupled_model.md)). The two differ before
any observation is assimilated. In the Mekong application, over the years before SMAP, when
nothing was assimilated, the `da = 1` arms differed from the deterministic open-loop
simulation by a mean absolute 7.6 per cent of flow at Kratie and 13.8 per cent at Chiang
Saen. Scored against a `da = 0` run, that offset is
added to the assimilation effect and is easily mistaken for it.

**Instead:** use the matched control, `mekong-da prepare-run --resolution control`: `da = 1`,
the same `sample.txt`, initial states and forcing as the assimilating arms, and observation
files with nothing to assimilate. Then the only difference between an arm and the control is
the observations. A quick check that the arms were staged identically: before the first
assimilated retrieval, an arm and the control should produce the same discharge.

---

## 5. Do not write the soil-moisture observation file without its two quality files

The screening test in `fortran/src/simulate.f` is:

```fortran
if (d5 > 0. .and. d2 < 0.4 .and. d3 < 0.1) then   ! soil moisture, RFI, DQX
```

The missing-value sentinel for soil moisture is **0.001, which is greater than zero and
passes the first test on its own.** What actually suppresses an update on a day with no
retrieval is the DQX and RFI sentinels of 1.0 failing the second and third tests.

So a run given a sentinel-filled soil-moisture file but a quality file full of valid flags
assimilates 0.001 m3/m3 into every hydrologic response unit on every simulated day, drives
the soil column to zero, and reports no error. Hours of compute, output that looks like a
drought.

**Instead:** always write all three files together, which is what
`mekong_da.obsfiles.write_observation_files` and the `mekong-da prepare-run` command do.
Never hand-edit one of the three.

---

## 6. Do not read column 5 of `output.rch`

The reach output line is `RCH GIS MON AREAkm2 FLOW_INcms FLOW_OUTcms`, so with the `REACH`
label counted, discharge is whitespace token **6**. Token 5 is inflow. A driver that reads
token 5 still produces something that looks like a hydrograph, with a seasonal cycle and
plausible metrics.

`mekong_da.rch` names both indices (`FLOW_OUT_INDEX`, `FLOW_IN_INDEX`) so the choice is
explicit, and `test_reads_outflow_not_inflow` pins it.

While in that file, two more:

* **Do not count days with `MON`.** It restarts every year and means either month or
  day-of-year depending on the print code. Count the wrap of the reach index instead.
* **Do not date the series from the simulation start year.** Output begins after the
  warm-up years dropped by `NYSKIP` in `file.cio`. In the Mekong application that is
  1998 + 3, so `output.rch` starts in 2001, not 1998.

---

## 7. Do not drop the awkward compiler flags

```
-std=legacy  -fallow-argument-mismatch  -fno-automatic
```

The first two accept pre-Fortran-90 constructs and mismatched dummy arguments that modern
gfortran rejects by default. Removing them turns a working build into several hundred
compile errors, which at least is loud.

`-fno-automatic` is the dangerous one. It gives local variables static storage, which this
code relies on where it carries state between calls without a `save` statement. Drop it and
the build still succeeds, the model still runs, and the numbers are wrong.

Two more build facts, both in `mekong_da.fortran.build`:

* `modparm.f` must compile **first**, because every other unit uses the module it defines.
* `modparm.o` must then be **excluded from the link**, because `main.f` includes
  `modparm.f` directly and linking both raises duplicate-symbol errors.

---

## 8. Do not resize the fixed-width fields in `input.da`

Lines 2 through 11 are read under `format(f5.2)` in `readinput.f90`. A value written at a
different width is silently misread, not rejected, and the run continues with a wrong
observation-error coefficient. `mekong_da.inputda.InputDA` renders those ten fields at
exactly five characters and refuses a value that will not fit.

The other files are read list-directed, so their column alignment is free. What is not
free is their shape:

* the three observation files must have exactly `n_days_sim` rows and `mhru` columns;
* `sample.txt` must have `n_days_sim * n_ensembles * n_en_member` rows. Too few ends the
  run with an end-of-file error hours in; too many are ignored.

`n_days_sim` is computed by `maxndays.f90` from `file.cio`, so changing the simulation
calendar changes the required file sizes. In the Mekong application it is 9,862.

---

## 9. Do not run the Fortran model without watching the disk

SWAT writes large output files. The shipped `readfile.f` already sends every `output*` file
except `output.rch` and `output.std` to `/dev/null`, so a run keeps only what the framework
reads, but the observation files (about 400 MB each at the Mekong geometry), `sample.txt`
(about 165 MB), `output.rch` and the per-step diagnostics when `wrdtf = 1` still need room in
every run directory.

* `mekong_da.fortran.run` refuses to start below 60 GB free and stops a run if free space
  falls below 5 GB while it is going. Do not lower those guards to make a run start.
* If you restore output files such as `output.hru` in `readfile.f`, size the volume for them
  first; on a large basin they reach tens of gigabytes per run.
* The binary is serial, so parallelism means several run directories; cap their number by
  disk, not cores.
* Keep the harvested hydrographs and `output.rch`, then delete the run directory.
* The binary must live on a volume that permits execution. An exFAT external drive cannot
  execute it; APFS or a normal Linux filesystem can.

**Do not run two model instances in the same directory.** Every SWAT path is relative to
the working directory, so two instances overwrite each other's output with no warning.

**`/dev/null` is POSIX only.** On Windows, change those `open` statements in `readfile.f` to
`NUL` or to file names before running.

---

## 10. Do not call this model SWAT+

It is **classic SWAT2012, revision 629**. The binary says so (`VER 2012/Rev 629`, asserted
by `test_binary_is_swat2012_rev629`), every output header says so, and the input and output
layout is SWAT2012 throughout: `output.rch`, `fig.fig`, subbasins, hydrologic response
units, variable-storage routing.

SWAT+ is a different code base with different file formats. Citing Bieger et al. (2017)
for this model is a factual error. Cite Arnold et al. (1998) and Neitsch et al. (2011).

---

## 11. Do not switch the Kling-Gupta variability term

`mekong_da.metrics.metrics` uses the standard-deviation ratio,
`alpha = sd(sim) / sd(obs)`, which is Gupta et al. (2009) as originally stated. A
coefficient-of-variation variant, `gamma = CV(sim) / CV(obs)`, also circulates. The two
give different numbers, and only the first reproduces the paper's tables. `test_kge_uses_the_standard_deviation_ratio` pins it.

Round metrics at presentation time, never in storage, or the rounding compounds.

---

## 12. Do not overwrite the expected output you are checking against

`data/enkf/reference_*.csv` is the expected filter output for the worked example.
`mekong-da reproduce` writes to `outputs/enkf/` and compares. A script that reads and writes
the same file quietly redefines the thing it is verifying, and any drift becomes invisible
after one run. `test_reproduction_does_not_touch_the_archive` asserts the reference file is
byte-identical after a reproduction run. When working on your own series, keep them in a
separate data directory (`--data-dir`) rather than overwriting the example.

---

## 13. Do not read agreement with the assimilated observation as accuracy

Every skill number that compares the analysis with the observation it assimilated (the
RMSE reductions of `mekong-da reproduce`, the tuning scores, most tables in
`outputs/analyses/`) measures how closely the filter follows its input. It says nothing on
its own about how close either is to the true soil moisture, and a filter that copies its
input scores best. A held-out period does not change this when the held-out target was also
assimilated. Independent evidence needs an independent reference: in situ soil moisture, or,
as in the coupled experiment, discharge at gauges the filter never sees.

---

## 14. Do not leave the diagnostic writer on for a long run

`wrdtf = 1` in `input.da` makes the model write `KalmanGain.dat` and `correlation.dat` on
every time step for hydrologic response unit 1. That is useful for a short diagnostic run
and pure overhead for a multi-decade one: on a 27-year Mekong run it added several
gigabytes per arm that nothing reads. `InputDA.wrdtf` defaults to 0, as in the paper's runs.
To turn it on for a short run, pass it in from Python:

```python
from mekong_da.inputda import InputDA
from mekong_da.pipeline import prepare_fortran_run

prepare_fortran_run(run_dir=..., resolution="9km", template_dir=...,
                    input_da=InputDA(wrdtf=1, da_stopday=365))
```

Related: `da_stopday` is how a forecast-skill experiment is set up, by stopping the updates
partway and letting the model run free. The default, 99999, assimilates throughout. If a
run's skill collapses after a particular day, check this value before looking anywhere
else.

---

## 15. Smaller traps, in one list

* **Do not infer the scored day set from the finite observations.** On the example files
  `obs_available` and `isfinite(observation)` coincide at all three resolutions, 3,447,
  3,182 and 3,065 days, and `tests/test_schedule.py` checks that they do. The mask is
  still carried as an explicit column so the scored day set is stated in the data rather
  than inferred from it; pass it, and keep the column in inputs you build.
* **Do not feed the filter an open-loop series with gaps.** The forecast step nudges toward
  it every day, so one NaN poisons the rest of the run. `run_enkf` raises instead.
* **Do not compare the three resolutions on skill alone.** They assimilate different
  numbers of days, so a like-for-like comparison has to say so.
* **Do not open the NetCDF archives without disabling HDF5 file locking.** A
  `PermissionError [Errno 13]` on a NetCDF or HDF5 file is locking, not permissions. Set
  `HDF5_USE_FILE_LOCKING=FALSE`.
* **Do not commit the Fortran object files.** There are 324 of them, they are
  architecture-specific, and they are not a reproducible artifact. `.gitignore` covers
  `*.o`, `*.mod` and `fortran/bin/`, and `test_sources_are_present` asserts none are
  tracked.
* **Do not treat `preprocessing/R/` as tested code.** It documents how the subbasin
  series were extracted, it is GPL-3.0, its paths must be edited before use, and nothing in
  `src/` imports it.
* **Do not edit the worked example in `data/`.** Generated files belong in `outputs/`,
  which is regenerable and ignored by git; your own inputs belong in a separate data
  directory.
