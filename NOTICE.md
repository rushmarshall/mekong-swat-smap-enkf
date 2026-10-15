# Attribution and third-party components

Author of this repository: **Sebastian R.O. Marshall** (smarsh46@jh.edu), Department of
Environmental Health and Engineering, Johns Hopkins University.

This repository packages the data-assimilation framework behind Marshall et al.,
*Relative roles of spatial resolution and temporal coverage in basin-average SMAP soil
moisture assimilation over the Mekong River Basin* (Journal of Hydrology, in revision). It
is assembled from work by several groups. What follows records who wrote what and the terms
that apply.

## Written for this study

Everything under `src/`, `analyses/`, `tests/`, `docs/`, `examples/` and `scripts/`, and the
build and packaging files. MIT licensed, see `LICENSE`.

That includes the Python ensemble Kalman filter, the CDF matching, the parameter search, the
metrics, the diagnostic analyses, the `input.da` reader and writer, the observation-file
writers, the `output.rch` parser, the compile and run wrappers, and the command-line
interface.

## `fortran/src/`: the assimilating hydrological model

Four layers, oldest first.

1. **SWAT2012, revision 629** (Soil and Water Assessment Tool), developed by Texas A&M
   AgriLife Research and the USDA Agricultural Research Service. Arnold et al. (1998),
   *Journal of the American Water Resources Association* 34(1), 73-89,
   doi:10.1111/j.1752-1688.1998.tb05961.x; theoretical documentation Neitsch et al.
   (2011). Distributed by Texas A&M at https://swat.tamu.edu. The binary built from these
   sources reports `SWAT Aug 25 2014  VER 2012/Rev 629`. This is classic SWAT2012, not
   SWAT+; the two are different code bases with different input and output formats.
2. **The data-assimilation coupling.** The ensemble Kalman filter routines added to the time
   loop (`EnKF.f90`, `Kalman_Gain_calc.f90`, `init_da.f90`, `post_da.f90`,
   `ens_reform.f90`, `sm_perturb.f90`, `sol_ens_rest.f90`, `daparm.f90`, `readinput.f90`,
   `readsm.f90`, `rngen.f90`, `maxndays.f90` and companions) originate with Amol Patil and
   R.A.A.J. Ramsankaran, released at https://github.com/amolpatil771/SWAT_DA. Method:
   Patil, A. and Ramsankaran, R.A.A.J. (2017), *Journal of Hydrology* 555, 683-696,
   doi:10.1016/j.jhydrol.2017.10.058.
3. **The SMAP application.** Adapted for SMAP assimilation by Le et al. (2022), *Remote
   Sensing* 14(7), 1607, doi:10.3390/rs14071607, https://github.com/mhle510/smap_enkf_swat.
4. **Corrections made for this study** in five files (`EnKF.f90`, `init_da.f90`,
   `modparm.f`, `readfile.f`, `simulate.f`): the observation operator converts volumetric
   retrievals to plant-available layer storage, the observation-error standard deviation is
   transformed with the same layer depth, each updated member is bounded between zero and
   saturation storage, the daily analysis mass change is written for a water-balance audit,
   and unread model output files are redirected so a run does not fill a disk. Their SHA-256
   values are pinned in `tests/test_corrected_operator.py`.

`fortran/UPSTREAM_README.md` is the upstream README, kept verbatim. The upstream terms of
SWAT2012 and of the coupling continue to apply to these files.

## `preprocessing/R/`: SMAP extraction and bias correction

Four R scripts from the `smap_enkf_swat` line of work above, **GPL-3.0**, licence text at
`preprocessing/R/LICENSE`, upstream README at `preprocessing/R/UPSTREAM_README.md`. They
document how subbasin SMAP series are produced from raw granules. This directory is
licensed separately from the rest of the repository; nothing in `src/` imports or derives
from it.

## Data

No satellite, forcing or gauge data are redistributed. `data/` holds only the worked
example: basin-average inputs of the offline filter derived from public SMAP products and
the study's SWAT simulation, its expected output, and the tuning grids. Mekong River
Commission discharge observations are subject to the Commission's data-sharing terms and are
obtained from https://portal.mrcmekong.org/. `docs/03_data.md` names the source of every
input.
