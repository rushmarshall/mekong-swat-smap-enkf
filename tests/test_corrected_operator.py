"""Check the corrected sources and exercise their operator on a controlled state.

The five files below carry the observation-operator and storage-bound corrections used for
the published coupled experiment. Their SHA-256 values pin them byte for byte, line endings
included; a deliberate change to any of them should update the value here and say why.
The controlled-state fixture lives in ``tests/fixtures/``.

Author: Sebastian R.O. Marshall.
"""

import hashlib
import shutil
import subprocess

import pytest

from mekong_da.config import FORTRAN_DIR, REPO_ROOT


FIXTURES = REPO_ROOT / "tests" / "fixtures"
CORRECTED_SHA256 = {
    "EnKF.f90": "6e554516d43e43553d0f4243478cf01cc18d584e9ded73703904de6dbb7829ad",
    "init_da.f90": "3fc772c8a002bbb985c4c5e511670d7330dafa32424a5ba5690b1d57a6a2763e",
    "modparm.f": "9e7670e43730bfc98e5af3a94679cdfcf3bc7e22b0c36c8d7e15b13a861f59f9",
    "readfile.f": "35bf0cfa628e9829a5a60daa3027f76e289bdf22a5fc9616411c6ce77b3d280e",
    "simulate.f": "318f8463673165bdbf469a497a07ac7863bccfcc33ddc88a58a5e89df919e9ad",
}
ROUTINES = (
    "init_da.f90", "EnKF.f90", "st_error.f90", "obs_err.f90", "st_cov.f90",
    "obs_cov.f90", "st_obs.f90", "Kalman_Gain_calc.f90", "inverse.f90",
    "post_da.f90",
)


@pytest.mark.parametrize("filename", sorted(CORRECTED_SHA256))
def test_corrected_source_matches_recorded_hash(filename):
    actual = hashlib.sha256((FORTRAN_DIR / "src" / filename).read_bytes()).hexdigest()
    assert actual == CORRECTED_SHA256[filename], filename


def _controlled_state_sources():
    """The fixture's parameter module and driver program, as Fortran source text."""
    return {
        "MODULE": (FIXTURES / "operator_parm.f90").read_text(),
        "PROGRAM": (FIXTURES / "operator_program.f90").read_text(),
    }


@pytest.mark.slow
@pytest.mark.parametrize("missing_transfer", [False, True], ids=["corrected", "missing-transfer"])
def test_corrected_operator_on_controlled_state(tmp_path, missing_transfer):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran not installed; corrected operator was not compiled")

    constants = _controlled_state_sources()
    module = tmp_path / "parm.f90"
    program = tmp_path / "test.f90"
    module.write_text(constants["MODULE"])
    # Preserve every archived calculation and check the actual state transfer too.
    anchor = "  call post_da(1)\n"
    assert constants["PROGRAM"].count(anchor) == 1
    installed_checks = (
        "  if(maxval(abs(transpose(ens_sol_st(:,:,1))-expected_xaa))>2.e-4) stop 23\n"
        "  if(maxval(abs(sol_st(:,1)-expected))>2.e-4) stop 24\n"
    )
    program.write_text(constants["PROGRAM"].replace(anchor, anchor + installed_checks))

    source_paths = [FORTRAN_DIR / "src" / name for name in ROUTINES]
    if missing_transfer:
        # A defective copy proves the fixture rejects an analysis that was never installed.
        source = (FORTRAN_DIR / "src" / "post_da.f90").read_text()
        for layer in (1, 2):
            statement = f"    ens_sol_st(1:n_en_member,{layer},jj) = xaa({layer},1:n_en_member)\n"
            assert source.count(statement) == 1
            source = source.replace(statement, "")
        defective = tmp_path / "post_da.f90"
        defective.write_text(source)
        source_paths = [defective if path.name == "post_da.f90" else path for path in source_paths]

    executable = tmp_path / "test_operator"
    command = [
        compiler, "-std=legacy", "-fallow-argument-mismatch",
        "-ffree-line-length-none", "-fcheck=all", "-o", str(executable),
        str(module), *[str(path) for path in source_paths],
        str(program),
    ]
    compiled = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    completed = subprocess.run(
        [str(executable)], cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    if missing_transfer:
        assert completed.returncode == 23, completed.stdout + completed.stderr
        assert "STOP 23" in completed.stderr
        return
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "PASS: unit transform, error variance, no-change target, increment, bounds, water total"
        in completed.stdout
    ), completed.stdout + completed.stderr
