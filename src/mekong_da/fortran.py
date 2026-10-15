"""Build and run the Fortran SWAT-DA executable from Python.

The streamflow half of the study is a modified SWAT2012 (Rev 629) in which the Ensemble
Kalman Filter updates soil-water storage inside the hydrological time loop, so the
analysis propagates through the water balance into routed discharge. That coupling cannot
be reproduced by post-processing, which is why the Fortran is here rather than replaced.

Python's role is orchestration, not numerics: stage a run directory, write the five DA
input files, invoke the binary, harvest ``output.rch``, and tear the scratch down. This
module is that wrapper.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from .config import FORTRAN_DIR

__all__ = ["BuildResult", "RunResult", "build", "run", "find_compiler", "check_free_space_gb"]

#: Fixed-form files whose comments run past column 72 and must be compiled at the default
#: line length rather than the extended one.
SHORT_LINE_SOURCES = ("grow.f", "tran.f")

#: Compiled first because every other translation unit uses the module it defines, then
#: excluded from the link because ``main.f`` already includes it.
MODULE_SOURCE = "modparm.f"


@dataclass
class BuildResult:
    """Outcome of a compile."""

    executable: Path
    compiler: str
    n_compiled: int
    n_failed: int
    failures: Sequence[str]
    seconds: float

    @property
    def ok(self) -> bool:
        return self.n_failed == 0 and self.executable.exists()


@dataclass
class RunResult:
    """Outcome of a model run."""

    run_dir: Path
    returncode: int
    minutes: float
    output_rch: Optional[Path]
    stdout_tail: str
    stderr_tail: str
    #: Lowest free space seen on the volume during the run, in GB. A run that finishes with
    #: this close to zero finished on luck.
    min_free_gb: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_rch is not None and self.output_rch.exists()


def find_compiler(preferred: str = "gfortran") -> str:
    """Return a usable Fortran compiler command, or raise with a helpful message."""
    found = shutil.which(preferred)
    if found:
        return found
    raise RuntimeError(
        f"'{preferred}' is not on PATH. Install it first: 'brew install gcc' on macOS, "
        "'apt-get install gfortran' on Debian or Ubuntu, 'conda install -c conda-forge "
        "gfortran' inside the environment."
    )


def check_free_space_gb(path: Union[str, Path]) -> float:
    """Free space in gigabytes on the volume holding ``path``."""
    usage = shutil.disk_usage(str(path))
    return usage.free / 1e9


def build(
    src_dir: Union[str, Path] = FORTRAN_DIR / "src",
    output: Union[str, Path] = FORTRAN_DIR / "bin" / "swat_da",
    compiler: str = "gfortran",
    optimize: bool = True,
    keep_objects: bool = False,
) -> BuildResult:
    """Compile the SWAT-DA sources into a single executable.

    The build is a plain three-pass sequence rather than a Makefile because the source
    predates any module system beyond one shared ``parm`` module: compile that module
    first, then every fixed-form file, then every free-form file, then link.

    Parameters
    ----------
    src_dir:
        Directory holding the ``.f`` and ``.f90`` sources.
    output:
        Path of the executable to produce. Parent directories are created.
    compiler:
        Fortran compiler command.
    optimize:
        ``True`` builds at ``-O2`` for production. ``False`` builds at ``-g -fbacktrace``,
        which is what to use when a run aborts and the failing routine must be identified.
    keep_objects:
        Leave the ``.o`` and ``.mod`` files in a scratch directory instead of removing
        them. Useful only when debugging the build itself.

    Returns
    -------
    BuildResult

    Notes
    -----
    Three compiler flags are not optional. ``-std=legacy`` and
    ``-fallow-argument-mismatch`` accept the pre-Fortran-90 constructs and the mismatched
    dummy arguments that the original code relies on; modern gfortran rejects both by
    default. ``-fno-automatic`` gives local variables static storage, which the code
    assumes when it carries state between calls without a ``save`` statement. Dropping it
    produces an executable that builds cleanly and then computes wrong numbers.
    """
    src_dir = Path(src_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fc = find_compiler(compiler)

    common = ["-fno-automatic", "-fallow-argument-mismatch", "-std=legacy"]
    common += ["-O2"] if optimize else ["-g", "-fbacktrace"]
    fixed = common + ["-ffixed-form", "-ffixed-line-length-132"]
    fixed_72 = common + ["-ffixed-form"]
    free = common + ["-ffree-form"]

    obj_dir = output.parent / "_build"
    if obj_dir.exists():
        shutil.rmtree(obj_dir)
    obj_dir.mkdir(parents=True)

    failures: list[str] = []
    n_compiled = 0
    started = time.time()

    def compile_one(source: Path, flags: list) -> Optional[Path]:
        nonlocal n_compiled
        obj = obj_dir / (source.stem + ".o")
        proc = subprocess.run(
            [fc, *flags, "-c", source.name, "-o", str(obj)],
            cwd=str(src_dir), capture_output=True, text=True,
        )
        if proc.returncode != 0:
            failures.append(f"{source.name}: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'unknown error'}")
            return None
        n_compiled += 1
        return obj

    # Pass 1: the shared module, which every later unit needs on the include path.
    module_src = src_dir / MODULE_SOURCE
    if not module_src.exists():
        raise FileNotFoundError(f"{MODULE_SOURCE} not found in {src_dir}")
    compile_one(module_src, fixed)

    # Pass 2: fixed-form sources.
    for source in sorted(src_dir.glob("*.f")):
        if source.name == MODULE_SOURCE:
            continue
        compile_one(source, fixed_72 if source.name in SHORT_LINE_SOURCES else fixed)

    # Pass 3: free-form sources, which include every data-assimilation routine.
    for source in sorted(src_dir.glob("*.f90")):
        compile_one(source, free)

    # The module object is dropped before linking: main.f includes modparm.f directly, so
    # linking both raises duplicate-symbol errors.
    module_obj = obj_dir / Path(MODULE_SOURCE).stem
    for stale in (module_obj.with_suffix(".o"),):
        if stale.exists():
            stale.unlink()

    objects = sorted(str(p) for p in obj_dir.glob("*.o"))
    if not objects:
        raise RuntimeError(f"nothing compiled; first failures: {failures[:5]}")
    link = subprocess.run(
        [fc, *common, "-o", str(output), *objects],
        cwd=str(src_dir), capture_output=True, text=True,
    )
    if link.returncode != 0:
        failures.append(f"link: {link.stderr.strip()[-400:]}")
    else:
        output.chmod(0o755)

    # gfortran writes .mod files into the working directory; clean them out of the source tree.
    for mod in src_dir.glob("*.mod"):
        shutil.move(str(mod), str(obj_dir / mod.name))
    for stray in src_dir.glob("*.o"):
        stray.unlink()
    if not keep_objects:
        shutil.rmtree(obj_dir, ignore_errors=True)

    return BuildResult(
        executable=output,
        compiler=fc,
        n_compiled=n_compiled,
        n_failed=len(failures),
        failures=failures,
        seconds=time.time() - started,
    )


def run(
    run_dir: Union[str, Path],
    executable: Union[str, Path] = FORTRAN_DIR / "bin" / "swat_da",
    timeout_hours: float = 6.0,
    min_free_gb: float = 60.0,
    abort_below_gb: float = 5.0,
    poll_seconds: float = 30.0,
) -> RunResult:
    """Execute the model in a prepared run directory.

    Parameters
    ----------
    run_dir:
        A SWAT project directory (a ``TxtInOut``) already carrying the five DA input files.
    executable:
        Path to the compiled binary. It is copied into ``run_dir`` before execution
        because the model resolves every file relative to its working directory.
    timeout_hours:
        Wall-clock limit. A full Mekong run is hours, not minutes.
    min_free_gb:
        Refuse to start below this much free space. A single run writes tens of gigabytes
        of output before it writes any of the reach output the analysis needs, and running
        the volume dry loses the whole run rather than truncating it.
    abort_below_gb:
        Stop this run if free space falls this low while it is going. Checking only at the
        start cannot protect a job whose consumption grows for hours: a four-arm run
        measured on 2026-08-20 began with 164 GB free and was on course for 209 GB. Ending
        one arm early leaves the volume alive for the others, which a full disk does not.
    poll_seconds:
        How often to look at the volume while the model runs.

    Returns
    -------
    RunResult
    """
    run_dir = Path(run_dir)
    executable = Path(executable)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if not executable.exists():
        raise FileNotFoundError(
            f"executable not found: {executable}. Build it first with "
            "'mekong-da build-fortran'."
        )

    free = check_free_space_gb(run_dir)
    if free < min_free_gb:
        raise RuntimeError(
            f"only {free:.0f} GB free on the volume holding {run_dir}, below the "
            f"{min_free_gb:.0f} GB guard. A Mekong run writes tens of gigabytes of "
            "intermediate output; see docs/07_pitfalls.md."
        )

    local_exe = run_dir / executable.name
    if local_exe.resolve() != executable.resolve():
        shutil.copy2(executable, local_exe)
    if os.name == "posix":                        # a no-op on Windows, where it is meaningless
        local_exe.chmod(0o755)

    started = time.time()
    deadline = started + timeout_hours * 3600
    floor = free
    proc = subprocess.Popen(
        # the absolute path rather than "./name": the model still resolves its own data
        # files relative to cwd, but the invocation no longer assumes a POSIX shell path
        [str(local_exe)],
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while True:
        try:
            out, err = proc.communicate(timeout=poll_seconds)
            break
        except subprocess.TimeoutExpired:
            pass                                    # still running, so look at the volume
        floor = min(floor, check_free_space_gb(run_dir))
        if floor < abort_below_gb:
            proc.kill()
            proc.communicate()
            raise RuntimeError(
                f"stopped the run in {run_dir}: free space fell to {floor:.1f} GB, below "
                f"the {abort_below_gb:.0f} GB floor. The unread outputs (output.hru, "
                "output.sed, output.sub) are what fill a volume; see docs/07_pitfalls.md "
                "section 9."
            )
        if time.time() > deadline:
            proc.kill()
            proc.communicate()
            raise subprocess.TimeoutExpired(local_exe.name, timeout_hours * 3600)

    minutes = (time.time() - started) / 60.0
    rch = run_dir / "output.rch"
    return RunResult(
        run_dir=run_dir,
        returncode=proc.returncode,
        minutes=minutes,
        output_rch=rch if rch.exists() else None,
        stdout_tail="\n".join((out or "").strip().splitlines()[-20:]),
        stderr_tail=(err or "").strip()[-2000:],
        min_free_gb=floor,
    )
