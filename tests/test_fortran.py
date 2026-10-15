"""The Fortran half: it must compile, and it must accept what Python writes for it.

These tests are skipped when no Fortran compiler is present, which is the honest outcome
on a machine that cannot build the model, rather than a silent pass.
"""

import shutil
import subprocess

import pytest

from mekong_da.config import FORTRAN_DIR
from mekong_da.inputda import InputDA

# `slow` marks only what actually compiles the model, so `-m "not slow"` still runs the
# source-inventory and doc-drift checks, which is exactly the lane someone without a
# Fortran compiler uses.

HAS_GFORTRAN = shutil.which("gfortran") is not None


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    if not HAS_GFORTRAN:
        pytest.skip("gfortran not installed")
    from mekong_da.fortran import build

    target = tmp_path_factory.mktemp("bin") / "swat_da"
    result = build(output=target)
    assert result.n_failed == 0, f"{result.n_failed} sources failed: {result.failures[:5]}"
    return result.executable


def test_sources_are_present():
    src = FORTRAN_DIR / "src"
    assert (src / "modparm.f").exists()
    assert (src / "EnKF.f90").exists(), "the assimilation routine is missing"
    assert len(list(src.glob("*.f"))) > 200
    assert not list(src.glob("*.o")), "object files must not be committed"


@pytest.mark.slow
def test_builds_cleanly(executable):
    assert executable.exists() and executable.stat().st_size > 100_000


@pytest.mark.slow
def test_binary_is_swat2012_rev629(executable):
    """The model identity, read out of the binary rather than assumed."""
    blob = executable.read_bytes()
    assert b"VER 2012/Rev 629" in blob


@pytest.mark.slow
def test_binary_accepts_a_python_written_input_da(executable, tmp_path):
    """Cross-language contract: the Fortran must parse what :mod:`inputda` writes.

    With ``input.da`` present but no SWAT project around it, the run must get past
    ``readinput.f90`` and fail later, in ``maxndays.f90``, reading ``file.cio``. Failing
    inside ``readinput.f90`` would mean the control file was written incompatibly.
    """
    InputDA().write(tmp_path / "input.da")
    proc = subprocess.run([str(executable)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert "SWAT2012" in proc.stdout
    assert "readinput.f90" not in proc.stderr, proc.stderr[:400]
    assert "file.cio" in proc.stderr, proc.stderr[:400]


def test_the_documented_hand_build_keeps_the_line_length_exception():
    """docs/04 carries a copy-pasteable build recipe. It must not drift from the code.

    Compiled at the extended line length, grow.f and tran.f fail with `Unclassifiable
    statement`, both are skipped, and the link then reports _tran_ and _plantmod_ as
    undefined three screens later. The recipe shipped without the exception once; running
    it verbatim reproduced exactly that failure.
    """
    import re

    from mekong_da.config import REPO_ROOT
    from mekong_da.fortran import SHORT_LINE_SOURCES

    doc = (REPO_ROOT / "docs" / "04_coupled_model.md").read_text()
    block = re.search(r"```bash\ncd fortran/src\n(.*?)```", doc, re.S)
    assert block, "the hand-build recipe is missing from docs/04_coupled_model.md"
    recipe = block.group(1)
    for source in SHORT_LINE_SOURCES:
        assert source in recipe, f"{source} needs the default line length in the recipe"
    assert "-ffixed-line-length-132" in recipe


# --------------------------------------------------------------------------------------
# The volume watch. These use a stand-in executable rather than SWAT, so they need no
# compiler and finish instantly: what is under test is the guard, not the model.


def _standin(tmp_path, script):
    exe = tmp_path / "swat_da"
    exe.write_text("#!/bin/sh\n" + script)
    exe.chmod(0o755)
    return exe


def test_run_stops_when_free_space_falls_below_the_floor(tmp_path):
    """A start-time check cannot protect a job that grows for hours.

    The four-arm run of 2026-08-20 began with 164 GB free and was on course for 209 GB, so
    the launch guard passed and the volume would still have filled four fifths of the way
    in. Ending one arm early leaves the volume alive for the others.
    """
    from mekong_da.fortran import run

    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    exe = _standin(tmp_path, "sleep 3\n")

    with pytest.raises(RuntimeError, match="free space fell"):
        run(run_dir, executable=exe, min_free_gb=0.0,
            abort_below_gb=1e9,                    # every real volume is below this
            poll_seconds=0.05)


def test_run_reports_the_lowest_free_space_it_saw(tmp_path):
    """A run that finishes with the floor near zero finished on luck, so record it."""
    from mekong_da.fortran import run

    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    exe = _standin(tmp_path, "echo done > output.rch\n")

    result = run(run_dir, executable=exe, min_free_gb=0.0, poll_seconds=0.05)
    assert result.returncode == 0
    assert result.ok
    assert result.min_free_gb > 0
    assert result.min_free_gb == result.min_free_gb        # not NaN


def test_run_still_captures_output(tmp_path):
    """Switching from subprocess.run to Popen must not lose stdout or stderr."""
    from mekong_da.fortran import run

    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    exe = _standin(tmp_path, "echo hello-stdout\necho hello-stderr >&2\ntouch output.rch\n")

    result = run(run_dir, executable=exe, min_free_gb=0.0, poll_seconds=0.05)
    assert "hello-stdout" in result.stdout_tail
    assert "hello-stderr" in result.stderr_tail
