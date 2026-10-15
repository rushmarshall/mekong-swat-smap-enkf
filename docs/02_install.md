# Installation

Two tiers. The first runs the offline filter, its tuning, the scoring and the diagnostic
analyses, and needs almost nothing. The second adds the Fortran toolchain, which is needed
only to build and run the coupled hydrological model.

---

## Tier 1: the Python framework

Requirements: Python 3.10 or newer, NumPy 1.24 or newer, pandas 2.0 or newer. That is the
entire hard dependency list.

### With conda

```bash
conda env create -f environment.yml
conda activate mekong-da
```

### With pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `[dev]` | pytest | the test suite |
| `[analyses]` | joblib, statsmodels | the scripts in `analyses/` |
| `[raw]` | xarray, netCDF4, dask | `scripts/extract_basin_mean.py` on raw SMAP NetCDF archives |

Install editable (`-e`). The package finds `data/` relative to its own source file, and the
example data is not shipped inside the wheel. A non-editable `pip install .` puts the package
in site-packages; the commands then explain the problem rather than fail obscurely, and they
still work with `--data-dir /path/to/repo/data` or `export MEKONG_DA_DATA=/path/to/repo/data`.

### Verify

```bash
mekong-da info
```

The worked-example lines must read `[ok ]`. The coupled-model inputs are listed separately
with a count, because they are yours to supply and start at zero:

```
worked example (shipped with the repository)
  [ok ] filter inputs: 3/3
  [ok ] expected filter output: 3/3
  [ok ] tuning grids: 3/3

coupled-model inputs (yours to supply, see docs/03_data.md)
  [ 0] SMAP subbasin series
  [ 0] HRU-to-subbasin map
  [ 0] merged streamflow
```

Then run the worked example, which takes a few seconds:

```bash
mekong-da reproduce
```

The last line must read `REPRODUCED`. If it does not, the failure lines above it name what
drifted; do not build on a failure here.

---

## Tier 2: the Fortran toolchain

Needed to compile `swat_da` from `fortran/src/`. Any gfortran from version 9 onward works.

```bash
# macOS
brew install gcc

# Debian or Ubuntu
sudo apt-get install gfortran

# inside the conda environment, on any platform
conda install -c conda-forge gfortran
```

### Build and verify

```bash
mekong-da build-fortran
```

```
  compiler   /opt/homebrew/bin/gfortran
  compiled   324 translation units in 55 s
  failed     0
  executable .../fortran/bin/swat_da
```

`failed` must be 0. A non-zero count prints the first twenty failing files; the usual cause
is a compiler flag that was edited out ([07_pitfalls.md](07_pitfalls.md) explains why the
awkward ones are needed). Confirm the binary is the model it claims to be:

```bash
strings fortran/bin/swat_da | grep "VER 20"
# SWAT Aug 25 2014    VER 2012/Rev 629
```

Then run the Fortran tests, which compile the model, check that it accepts an `input.da`
written by `mekong_da.inputda`, and exercise the observation update on a controlled state:

```bash
pytest tests/test_fortran.py tests/test_corrected_operator.py -v
```

Without a compiler these tests skip rather than fail. The whole suite is `pytest tests`
(123 tests).

---

## What you still need to run the model

Building the executable is not the same as running the experiment. A run also needs a
calibrated SWAT2012 project for your basin (its `TxtInOut`), the subbasin SMAP series and the
HRU-to-subbasin map. [03_data.md](03_data.md) gives the formats and
[04_coupled_model.md](04_coupled_model.md) the run procedure.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `mekong-da: command not found` | package not installed, or wrong environment active | `pip install -e .` in the active environment |
| `MISSING` under worked example | incomplete clone | re-clone; `data/` is tracked |
| `cannot reach the data directory` | non-editable install, or a wrong `--data-dir` | `pip install -e .`, or pass `--data-dir` |
| `gfortran is not on PATH` | no Fortran compiler | see Tier 2 above |
| Several hundred compile errors | a required compiler flag was removed | [07_pitfalls.md](07_pitfalls.md) |
| `PermissionError [Errno 13]` on a NetCDF file | HDF5 file locking, not permissions | `export HDF5_USE_FILE_LOCKING=FALSE` |
| `mekong-da reproduce` prints `NOT REPRODUCED` | the filter or the metrics changed | `git diff src/mekong_da/enkf.py`, then [07_pitfalls.md](07_pitfalls.md) |

---

## Platforms

**macOS (arm64) and Linux (Debian, aarch64).** The Python framework, the worked example and
the full test suite, including the model build, have run on both with Python 3.10 to 3.13,
NumPy 2.2 to 2.5 and pandas 2.3 to 3.0, and with Homebrew gfortran, conda-forge gcc and
Debian gfortran 14.2. The example reproduces to about 1e-16 across systems; bit identity holds
within one operating system, because floating-point summation order belongs to the platform
libraries. Every text file the package reads or writes names its encoding, so results do not
depend on the locale.

**Windows.** The Python half is platform neutral: the binary is launched by absolute path and
text I/O is encoding-explicit. The model sources cross-compile with a mingw-w64 gfortran (add
`-static` to the link step, or the executable needs the gfortran runtime DLLs beside it).
However, `fortran/src/readfile.f` opens `/dev/null` for the model output files the framework
does not read, which exists only on POSIX systems. Before running the model on Windows,
change those `open` statements to `NUL` or back to file names
([04_coupled_model.md](04_coupled_model.md)). A complete run on Windows has not been
verified.
