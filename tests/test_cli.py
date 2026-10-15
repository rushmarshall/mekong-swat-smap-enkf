"""The command line: option handling, exit codes, and the checks each command performs."""

import pytest

from mekong_da.cli import build_parser, main
from mekong_da.config import DATA_DIR, OUTPUT_DIR


@pytest.mark.parametrize(
    "argv",
    [
        ["enkf", "--resolution", "9km", "--out-dir", "/tmp/x"],
        ["--out-dir", "/tmp/x", "enkf", "--resolution", "9km"],
    ],
    ids=["after the subcommand", "before the subcommand"],
)
def test_path_options_work_in_either_position(argv):
    """Putting --out-dir after the subcommand is what people type, so it has to work."""
    parser = build_parser()
    args = parser.parse_args(argv)
    from mekong_da.cli import _common_parser

    scanned, _ = _common_parser().parse_known_args(argv)
    resolved = scanned.out_dir if scanned.out_dir != str(OUTPUT_DIR) else args.out_dir
    assert resolved == "/tmp/x"


def test_every_subcommand_is_reachable():
    parser = build_parser()
    for command in ("info", "reproduce", "enkf", "tune", "build-fortran",
                    "prepare-run", "run-fortran", "harvest"):
        extra = {"prepare-run": ["--run-dir", "/tmp/r", "--resolution", "9km"],
                 "run-fortran": ["--run-dir", "/tmp/r"],
                 "harvest": ["--rch", "/tmp/a.rch", "--out", "/tmp/b.csv"]}.get(command, [])
        args = parser.parse_args([command, *extra])
        assert callable(args.func)


def test_info_exits_zero_when_the_data_is_present(capsys):
    assert main(["--data-dir", str(DATA_DIR), "info"]) == 0
    assert "MISSING" not in capsys.readouterr().out


def test_info_exits_nonzero_when_the_data_is_absent(tmp_path, capsys):
    """A missing data directory must fail loudly, not report an empty inventory as fine."""
    assert main(["--data-dir", str(tmp_path), "info"]) == 1
    assert "MISSING" in capsys.readouterr().out


def test_reproduce_exits_zero_and_says_so(tmp_path, capsys):
    assert main(["--data-dir", str(DATA_DIR), "--out-dir", str(tmp_path), "reproduce"]) == 0
    out = capsys.readouterr().out
    assert out.strip().endswith("REPRODUCED")
    assert (tmp_path / "tables" / "T1_soil_moisture_metrics.csv").exists()


def test_missing_data_is_diagnosed_rather_than_thrown(tmp_path, capsys):
    """A `pip install .` puts the package in site-packages, where it cannot see data/.

    Before the guard, the first read raised a bare pandas FileNotFoundError naming a path
    the reader had never seen. Every command that reads data/ must diagnose that instead.
    """
    absent = tmp_path / "nowhere"
    for command in ("reproduce", "enkf", "tune", "prepare-run"):
        extra = (["--run-dir", str(tmp_path), "--resolution", "9km"]
                 if command == "prepare-run" else [])
        assert main(["--data-dir", str(absent), command, *extra]) == 2, command
        err = capsys.readouterr().err
        assert "cannot reach the data directory" in err, command
        assert str(absent) in err, command


def test_commands_that_read_no_data_are_not_blocked_by_the_guard():
    """Compiling the model and harvesting a finished run must work without data/."""
    parser = build_parser()
    for command, extra in (("build-fortran", []),
                           ("run-fortran", ["--run-dir", "/tmp/r"]),
                           ("harvest", ["--rch", "/tmp/a.rch", "--out", "/tmp/b.csv"])):
        args = parser.parse_args([command, *extra])
        assert getattr(args, "needs_data", False) is False, command


def test_a_bad_path_is_one_line_not_a_traceback(tmp_path, capsys):
    """A wrong --rch or --template is a user mistake; a traceback reads as a crash."""
    assert main(["harvest", "--rch", str(tmp_path / "absent.rch"),
                 "--out", str(tmp_path / "o.csv")]) == 2
    err = capsys.readouterr().err
    assert "absent.rch" in err
    assert "Traceback" not in err
    assert "--traceback" in err


def test_the_traceback_flag_puts_the_trace_back(tmp_path):
    """Suppressing the trace must not cost debuggability."""
    with pytest.raises(FileNotFoundError):
        main(["harvest", "--rch", str(tmp_path / "absent.rch"),
              "--out", str(tmp_path / "o.csv"), "--traceback"])


def test_a_bad_template_names_the_option_that_is_wrong(tmp_path, capsys):
    assert main(["prepare-run", "--run-dir", str(tmp_path / "r"),
                 "--resolution", "9km", "--template", str(tmp_path / "no_txtinout")]) == 2
    assert "--template" in capsys.readouterr().err
