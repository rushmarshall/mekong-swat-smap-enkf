"""Command-line entry point: ``mekong-da``.

Every stage of both experiments is reachable from here without importing anything. Run
``mekong-da --help`` for the list, or ``mekong-da reproduce`` to run the worked example
and check it against the published soil-moisture result.

Author: Sebastian R.O. Marshall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import (
    DATA_DIR, DEFAULT_CONFIG, EnKFConfig, FORTRAN_DIR, MEKONG_2024, OPTIMAL_CONFIG,
    OUTPUT_DIR, REPO_ROOT, RESOLUTIONS,
)

BANNER = "Mekong SWAT + SMAP data assimilation"


def _rule(title: str = "") -> None:
    print("=" * 78)
    if title:
        print(title)
        print("=" * 78)


# --------------------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    """Report the environment and check that the worked-example data is present.

    Exits non-zero when anything expected is missing, so it works as a precondition check
    in a script. An empty group counts as missing: a directory that has been emptied must
    not read as a clean inventory.
    """
    from .fortran import check_free_space_gb

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    _rule(f"{BANNER} v{__version__}")
    print(f"python        {sys.version.split()[0]}")
    print(f"numpy         {np.__version__}")
    print(f"pandas        {pd.__version__}")
    print(f"data dir      {data_dir}")
    print(f"output dir    {out_dir}")
    print(f"free space    {check_free_space_gb(data_dir if data_dir.exists() else Path.cwd()):.0f} GB")
    try:
        from .fortran import find_compiler

        print(f"fortran       {find_compiler()}")
    except RuntimeError as exc:
        print(f"fortran       NOT AVAILABLE ({exc.args[0].split('.')[0]})")
    binary = FORTRAN_DIR / "bin" / "swat_da"
    print(f"swat_da       {'built at ' + str(binary) if binary.exists() else 'not built'}")

    print("\nworked example (shipped with the repository)")
    expected = [
        ("filter inputs", [data_dir / "enkf" / f"inputs_{r}.csv" for r in RESOLUTIONS], 3),
        ("expected filter output", [data_dir / "enkf" / f"reference_{r}.csv" for r in RESOLUTIONS], 3),
        ("tuning grids", sorted((data_dir / "enkf_tuning").glob("*.csv")), 3),
    ]
    shortfall = 0
    for label, paths, wanted in expected:
        present = sum(1 for p in paths if p.exists())
        shortfall += max(0, wanted - present)
        flag = "ok " if present >= wanted else "MISSING"
        print(f"  [{flag}] {label}: {present}/{wanted}")

    # The coupled model's inputs come from the user's own basin, so their absence is not
    # a fault; report them so a reader can see what the coupled path will look for.
    print("\ncoupled-model inputs (yours to supply, see docs/03_data.md)")
    yours = [
        ("SMAP subbasin series",
         [data_dir / "smap_subbasin_daily" / f"smap_{r}_subbasin_daily.csv" for r in RESOLUTIONS]),
        ("HRU-to-subbasin map", [data_dir / "hru_subbasin_mapping.csv"]),
        ("merged streamflow", sorted((data_dir / "streamflow").glob("*_merged_2024.csv"))),
    ]
    for label, paths in yours:
        print(f"  [{sum(1 for p in paths if p.exists()):>2d}] {label}")
    if shortfall:
        print(f"\n{shortfall} expected file(s) missing. See docs/03_data.md.")
    return 1 if shortfall else 0


def cmd_enkf(args: argparse.Namespace) -> int:
    """Run the soil-moisture filter and write the result series."""
    from .pipeline import reproduce_soil_moisture

    config = EnKFConfig.from_json(args.config) if args.config else OPTIMAL_CONFIG
    resolutions = RESOLUTIONS if args.resolution == "all" else (args.resolution,)
    _rule("Soil-moisture Ensemble Kalman Filter")
    print(f"config: {config}\n")
    result = reproduce_soil_moisture(
        data_dir=args.data_dir, out_dir=args.out_dir, config=config, resolutions=resolutions
    )
    print(f"\nwrote {len(result['frames'])} series to {Path(args.out_dir) / 'enkf'}")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Run the worked example, build the skill tables, and verify them."""
    from .pipeline import (
        reproduce_soil_moisture, soil_moisture_table, streamflow_table,
    )

    out_dir = Path(args.out_dir)
    _rule(f"{BANNER}: full reproduction")

    print("\n[1/3] Soil-moisture EnKF, all resolutions")
    result = reproduce_soil_moisture(data_dir=args.data_dir, out_dir=out_dir, verbose=True)
    summary_path = out_dir / "tables" / "soil_moisture_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(summary_path, index=False)

    print("\n[2/3] Soil-moisture skill table (T1)")
    t1 = soil_moisture_table(result["frames"])
    t1.to_csv(out_dir / "tables" / "T1_soil_moisture_metrics.csv", index=False)
    print(f"  {len(t1)} rows -> outputs/tables/T1_soil_moisture_metrics.csv")

    print("\n[3/3] Streamflow skill table (T2), from merged series under data/streamflow/")
    t2 = None
    if sorted((Path(args.data_dir) / "streamflow").glob("*_merged_2024.csv")):
        t2 = streamflow_table(data_dir=args.data_dir)
        t2.to_csv(out_dir / "tables" / "T2_streamflow_metrics.csv", index=False)
        print(f"  {len(t2)} rows -> outputs/tables/T2_streamflow_metrics.csv")
    else:
        print("  skipped: no merged streamflow series. Build them from your own coupled runs")
        print("  and gauge records with pipeline.build_merged_streamflow (docs/01_workflow.md).")

    _rule("Verification")
    ok = True
    diff = result["max_abs_diff"]
    print(f"  filter versus archived series, max absolute difference: {diff:.2e}")
    if not result["reproduced"]:
        ok = False
        print("  FAIL: the filter no longer reproduces the archived analysis")

    reference_t1 = Path(args.data_dir) / "reference_metrics" / "T1_soil_moisture_metrics.csv"
    if reference_t1.exists():
        expected = pd.read_csv(reference_t1)
        merged = t1.merge(expected, on=["Resolution", "Period", "Model"], suffixes=("", "_ref"))
        worst = 0.0
        for col in ("RMSE", "NSE", "KGE", "R"):
            worst = max(worst, float(np.nanmax(np.abs(merged[col] - merged[f"{col}_ref"]))))
        print(f"  T1 versus published table, max absolute difference: {worst:.2e}")
        if worst > 1e-6:
            ok = False
            print("  FAIL: T1 has drifted from the published table")

    reference_t2 = Path(args.data_dir) / "reference_metrics" / "T2_streamflow_metrics_from_merged2024.csv"
    if t2 is not None and reference_t2.exists():
        expected = pd.read_csv(reference_t2)
        merged = t2.merge(expected, on=["Station", "Period", "Model"], suffixes=("", "_ref"))
        if len(merged):
            worst = 0.0
            for col in ("RMSE", "NSE", "KGE", "R"):
                worst = max(worst, float(np.nanmax(np.abs(merged[col] - merged[f"{col}_ref"]))))
            print(f"  T2 versus published table, max absolute difference: {worst:.2e} "
                  f"over {len(merged)} shared rows")
            if worst > 1e-6:
                ok = False
                print("  FAIL: T2 has drifted from the published table")

    print()
    print("REPRODUCED" if ok else "NOT REPRODUCED, see the failures above")
    return 0 if ok else 1


def cmd_tune(args: argparse.Namespace) -> int:
    """Re-run the two-stage grid search that selected the filter configuration."""
    from .biascorrect import cdf_match_rank
    from .tuning import TRAIN_END, run_tuning

    inputs = Path(args.data_dir) / "enkf_tuning" / "inputs"
    baseline_path = inputs / "swat_baseline_smap_2015_2020.csv"
    if not baseline_path.exists():
        print(f"tuning inputs not found under {inputs}", file=sys.stderr)
        return 1
    open_loop = pd.read_csv(baseline_path, parse_dates=["date"], index_col="date")["sm_baseline"]

    smap = {}
    for res in RESOLUTIONS:
        path = inputs / f"smap_{res}_basin_mean_daily.csv"
        if path.exists():
            smap[res] = pd.read_csv(path, parse_dates=["date"], index_col="date").iloc[:, 0]
        else:
            print(f"  note: {path.name} absent, skipping {res} in stage 2")
    if args.resolution not in smap:
        print(f"tuning resolution {args.resolution} unavailable", file=sys.stderr)
        return 1

    _rule("EnKF parameter tuning")
    out_dir = Path(args.out_dir) / "enkf_tuning"
    result = run_tuning(open_loop, smap, tuning_resolution=args.resolution, out_dir=out_dir)
    print(f"\nselected configuration written to {out_dir/'optimal_config.json'}")
    print(json.dumps(result["optimal"].to_dict(), indent=2, default=str))
    return 0


def cmd_build_fortran(args: argparse.Namespace) -> int:
    """Compile the SWAT-DA executable."""
    from .fortran import build

    _rule("Building SWAT-DA (SWAT2012 Rev 629 with the EnKF coupling)")
    result = build(optimize=not args.debug, compiler=args.compiler)
    print(f"  compiler   {result.compiler}")
    print(f"  compiled   {result.n_compiled} translation units in {result.seconds:.0f} s")
    print(f"  failed     {result.n_failed}")
    for failure in result.failures[:20]:
        print(f"    {failure}")
    print(f"  executable {result.executable}")
    return 0 if result.ok else 1


def cmd_prepare_run(args: argparse.Namespace) -> int:
    """Write the five data-assimilation input files into a SWAT run directory."""
    from .pipeline import prepare_fortran_run

    resolution = None if args.resolution == "control" else args.resolution
    _rule(f"Staging a run directory: {args.resolution}")
    prepare_fortran_run(
        run_dir=args.run_dir,
        resolution=resolution,
        template_dir=args.template,
        data_dir=args.data_dir,
        n_members=args.members,
    )
    print(f"\nready: {args.run_dir}")
    print("run it with:  mekong-da run-fortran --run-dir " + str(args.run_dir))
    return 0


def cmd_run_fortran(args: argparse.Namespace) -> int:
    """Execute the compiled model in a prepared run directory."""
    from .fortran import run

    _rule("Running SWAT-DA")
    result = run(
        args.run_dir,
        executable=args.executable,
        timeout_hours=args.timeout_hours,
        min_free_gb=args.min_free_gb,
    )
    print(f"  exit code {result.returncode} after {result.minutes:.1f} min")
    if result.stdout_tail:
        print("  stdout tail:\n    " + result.stdout_tail.replace("\n", "\n    "))
    if result.returncode != 0:
        print("  stderr:\n    " + result.stderr_tail.replace("\n", "\n    "))
        return 1
    print(f"  output.rch {result.output_rch}")
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    """Pull the five gauge hydrographs out of a completed run."""
    from .pipeline import harvest_fortran_run

    _rule("Harvesting output.rch")
    frame = harvest_fortran_run(args.rch, out_csv=args.out)
    print(frame.describe().T.to_string())
    print(f"\nwrote {args.out}")
    return 0


# --------------------------------------------------------------------------------------


def _common_parser() -> argparse.ArgumentParser:
    """The two path options, shared by the top-level parser and every subparser."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=str(DATA_DIR), help="repository data directory")
    common.add_argument("--out-dir", default=str(OUTPUT_DIR), help="where results are written")
    common.add_argument("--traceback", action="store_true",
                        help="show the full Python traceback instead of a one-line message")
    return common


def _require_data(args: argparse.Namespace) -> int:
    """Return 0 when the data directory is reachable, or 2 after explaining why it is not.

    Without this, the first pandas read raises a bare ``FileNotFoundError`` naming a path
    the reader has never seen. The common way to land there is ``pip install .`` rather
    than ``pip install -e .``: the package then lives in site-packages, so ``REPO_ROOT``
    resolves outside the repository, and the example data is deliberately not
    shipped inside the wheel. The presence of ``pyproject.toml`` beside ``REPO_ROOT``
    separates that case from a data directory that is simply absent or wrongly pointed at.
    """
    data_dir = Path(args.data_dir)
    if (data_dir / "enkf").is_dir():
        return 0

    print(f"{BANNER}: cannot reach the data directory.", file=sys.stderr)
    print(f"  looked in   {data_dir}", file=sys.stderr)
    if not (REPO_ROOT / "pyproject.toml").exists():
        print("  cause       this package is installed non-editable, so it lives in "
              "site-packages and\n              cannot see the repository's data/ directory. "
              "The example data is not shipped\n              inside the wheel, by design, "
              "and outputs/ must stay writable.", file=sys.stderr)
        print("  fix         reinstall from a clone with  pip install -e .\n"
              "              or point at the clone with  --data-dir /path/to/repo/data\n"
              "              or export MEKONG_DA_DATA=/path/to/repo/data", file=sys.stderr)
    else:
        print("  fix         that directory has no enkf/ subdirectory. Run  mekong-da info\n"
              "              for the full inventory, and see docs/03_data.md.", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    # --data-dir and --out-dir are accepted both before and after the subcommand, because
    # putting them after is the natural thing to type and argparse rejects it by default.
    common = _common_parser()

    parser = argparse.ArgumentParser(
        prog="mekong-da",
        parents=[common],
        description=BANNER + ". Run 'mekong-da reproduce' to run the worked example and "
        "check it against the published soil-moisture result.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", parents=[common], help="environment and data inventory check")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("reproduce", parents=[common], help="run the worked example and verify it")
    p.set_defaults(func=cmd_reproduce, needs_data=True)

    p = sub.add_parser("enkf", parents=[common], help="run the soil-moisture filter")
    p.add_argument("--resolution", default="all", choices=[*RESOLUTIONS, "all"])
    p.add_argument("--config", help="JSON file of filter settings, for example data/enkf_tuning/optimal_config.json")
    p.set_defaults(func=cmd_enkf, needs_data=True)

    p = sub.add_parser("tune", parents=[common], help="re-run the two-stage parameter search")
    p.add_argument("--resolution", default="9km", choices=list(RESOLUTIONS))
    p.set_defaults(func=cmd_tune, needs_data=True)

    p = sub.add_parser("build-fortran", parents=[common], help="compile the SWAT-DA executable")
    p.add_argument("--debug", action="store_true", help="build with -g -fbacktrace instead of -O2")
    p.add_argument("--compiler", default="gfortran")
    p.set_defaults(func=cmd_build_fortran)

    p = sub.add_parser("prepare-run", parents=[common], help="write the DA input files into a SWAT run directory")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--resolution", required=True, choices=[*RESOLUTIONS, "control"],
                   help="SMAP product to assimilate, or 'control' for the matched ensemble control")
    p.add_argument("--template", help="calibrated TxtInOut to copy in first")
    p.add_argument("--members", type=int, default=200)
    p.set_defaults(func=cmd_prepare_run, needs_data=True)

    p = sub.add_parser("run-fortran", parents=[common], help="execute the model in a prepared run directory")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--executable", default=str(FORTRAN_DIR / "bin" / "swat_da"))
    p.add_argument("--timeout-hours", type=float, default=6.0)
    p.add_argument("--min-free-gb", type=float, default=60.0)
    p.set_defaults(func=cmd_run_fortran)

    p = sub.add_parser("harvest", parents=[common], help="extract gauge hydrographs from output.rch")
    p.add_argument("--rch", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_harvest)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # The subparser's own default silently overwrites a value given before the subcommand,
    # so rescan the raw arguments with the shared parser alone and let an explicit value win
    # from either position.
    scanned, _ = _common_parser().parse_known_args(argv)
    for name, default in (("data_dir", str(DATA_DIR)), ("out_dir", str(OUTPUT_DIR))):
        if getattr(scanned, name) != default:
            setattr(args, name, getattr(scanned, name))
    if getattr(args, "needs_data", False):
        shortfall = _require_data(args)
        if shortfall:
            return shortfall
    # A bad path or an unfinished model run is a user-facing condition, not a crash. The
    # messages raised downstream are already specific, so the only thing wrong with them is
    # the traceback wrapped around them; --traceback puts it back for debugging.
    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        if getattr(args, "traceback", False):
            raise
        print(f"{BANNER}: {exc}", file=sys.stderr)
        print("  (re-run with --traceback for the full trace)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
