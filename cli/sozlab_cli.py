"""Command line interface for SOZLab."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

from engine._version import __version__
from engine.analysis import SOZAnalysisEngine, load_project_json
from engine.export import PartialResultError, export_results, verify_export
from engine.extract import (
    ExtractionMismatchError,
    extract_from_results,
    select_from_result,
    write_extracted_trajectory,
)
from engine.logging_utils import setup_run_logger
from engine.provenance import InputsChangedError, verify_inputs
from engine.report import generate_report
from engine.validation import validate_project


def _progress(current: int, total: int, message: str) -> None:
    percent = int(100 * current / max(total, 1))
    sys.stdout.write(f"\r[{percent:3d}%] {message[:70]:<70}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _strip_removed_analysis_options(project) -> None:
    """H-bond hydration and water dynamics were disabled in 2.0 and removed in 2.1; drop old entries."""
    project.hbond_hydration.clear()
    project.water_dynamics.clear()


def _command_line() -> str:
    return " ".join(shlex.quote(a) for a in sys.argv)


def _apply_overrides(project, args) -> None:
    if getattr(args, "output", None):
        project.outputs.output_dir = args.output
    if getattr(args, "stride", None) is not None:
        project.analysis.stride = args.stride
    if getattr(args, "start", None) is not None:
        project.analysis.frame_start = args.start
    if getattr(args, "stop", None) is not None:
        project.analysis.frame_stop = args.stop
    if getattr(args, "no_ids", False):
        project.analysis.store_ids = False
    if getattr(args, "no_per_frame", False):
        project.outputs.write_per_frame = False
    if getattr(args, "workers", None) is not None:
        project.analysis.workers = None if args.workers <= 0 else args.workers
    if getattr(args, "dt", None) is not None:
        project.analysis.dt_override = args.dt
    if getattr(args, "gap_time", None) is not None:
        project.analysis.gap_tolerance_time = args.gap_time


def run_command(args: argparse.Namespace) -> int:
    project = load_project_json(args.project)
    _strip_removed_analysis_options(project)
    _apply_overrides(project, args)

    logger, log_path = setup_run_logger(project.outputs.output_dir)
    logger.info("CLI analysis requested: %s", _command_line())
    engine = SOZAnalysisEngine(project)
    progress_state = {"total": None}

    def _progress_cb(current: int, total: int, message: str) -> None:
        progress_state["total"] = total
        _progress(current, total, message)

    result = engine.run(
        progress=_progress_cb if args.progress else None,
        logger=logger,
        command_line=_command_line(),
        use_cache=not args.no_cache,
    )
    paths = export_results(result, project)
    print(f"Results written to {paths['output_dir']}")
    print(f"Log written to {log_path}")
    qc = result.qc_summary or {}
    run = result.run_info
    print(f"Frames analysed: {qc.get('frames_analysed')} | evaluation: {run.get('mode')} "
          f"({run.get('workers_used')} worker(s), {run.get('seconds', 0):.1f} s)")
    for item in qc.get("trust", []):
        if item["level"] in ("warning", "caution"):
            print(f"  [{item['level']}] {item['scope']}: {item['message']}")
    analysis_warnings = qc.get("analysis_warnings", [])
    if analysis_warnings:
        print("Warnings:")
        for warning in analysis_warnings[:10]:
            print(f"  - {warning}")
        if len(analysis_warnings) > 10:
            print(f"  ... {len(analysis_warnings) - 10} more in metadata.json")
    if args.report:
        report_path = generate_report(result, project, command_line=_command_line())
        print(f"Report written to {report_path}")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    project = load_project_json(args.project)
    _strip_removed_analysis_options(project)
    _apply_overrides(project, args)
    results = validate_project(project, max_frames=args.max_frames)
    bad = 0
    for res in results:
        print(f"SOZ {res.soz_name}: mismatches {res.mismatched_frames}/{res.total_frames} "
              f"(vectorised evaluation vs brute-force reference)")
        if res.first_mismatch:
            bad += 1
            print(f"  First mismatch frame: {res.first_mismatch['frame']}")
    return 1 if bad else 0


def extract_command(args: argparse.Namespace) -> int:
    out_dir_log = args.output
    if args.from_results:
        project = load_project_json(args.project)
        out_dir = args.out or project.extraction.output_dir
        rule = args.rule or project.extraction.rule or "n_solvent>=1"
        outputs, selection = extract_from_results(
            args.from_results,
            soz=args.soz,
            rule=rule,
            output_dir=out_dir,
            topology=project.inputs.topology if args.use_project_paths else None,
            trajectory=project.inputs.trajectory if args.use_project_paths else None,
            min_run_length=args.min_run if args.min_run is not None else project.extraction.min_run_length,
            gap_tolerance=args.gap if args.gap is not None else project.extraction.gap_tolerance,
            min_run_time=args.min_run_time,
            gap_tolerance_time=args.gap_time,
            prefix=args.prefix or "extracted",
            fmt=args.format or project.extraction.output_format,
            overwrite=args.overwrite,
        )
        print(f"Selected {len(selection.frame_indices)} frames (verified against the analysed inputs).")
        for key, path in outputs.items():
            print(f"{key}: {path}")
        return 0

    project = load_project_json(args.project)
    _strip_removed_analysis_options(project)
    _apply_overrides(project, args)
    if out_dir_log:
        project.outputs.output_dir = out_dir_log
    logger, log_path = setup_run_logger(project.outputs.output_dir)
    logger.info("CLI extract requested: %s", _command_line())
    project.outputs.write_per_frame = True
    result = SOZAnalysisEngine(project).run(logger=logger, command_line=_command_line(), use_cache=not args.no_cache)
    if not result.soz_results:
        raise ValueError("No SOZ results available for extraction.")
    soz_name = args.soz or next(iter(result.soz_results.keys()))
    if soz_name not in result.soz_results:
        raise ValueError(f"SOZ '{soz_name}' not found.")
    if project.inputs.trajectory is None:
        raise ValueError("Extraction requires a trajectory input.")

    import MDAnalysis as mda

    rule = args.rule or project.extraction.rule or "n_solvent>=1"
    selection = select_from_result(
        result, soz_name, rule,
        min_run_length=args.min_run if args.min_run is not None else project.extraction.min_run_length,
        gap_tolerance=args.gap if args.gap is not None else project.extraction.gap_tolerance,
        min_run_time=args.min_run_time,
        gap_tolerance_time=args.gap_time if args.gap_time is not None else None,
    )
    universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
    out_dir = args.out or project.extraction.output_dir
    outputs = write_extracted_trajectory(
        universe, selection, output_dir=out_dir, prefix=args.prefix or "extracted",
        fmt=args.format or project.extraction.output_format, overwrite=args.overwrite,
        topology_path=project.inputs.topology, trajectory_path=project.inputs.trajectory,
    )
    print(f"Selected {len(selection.frame_indices)} frames for SOZ '{soz_name}' (verified against the analysis).")
    print(f"Log written to {log_path}")
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


def verify_command(args: argparse.Namespace) -> int:
    problems = verify_export(args.results)
    with open(os.path.join(args.results, "provenance.json"), "r", encoding="utf-8") as handle:
        prov = json.load(handle)
    diffs = verify_inputs(prov.get("inputs", {}))
    print(f"SOZLab {prov.get('sozlab_version')} run {prov.get('run_id')} ({prov.get('created_utc')})")
    print(f"Export files: {'intact' if not problems else str(len(problems)) + ' problem(s)'}")
    for p in problems:
        print(f"  - {p}")
    print(f"Input files: {'identical to those analysed' if not diffs else 'CHANGED'}")
    for d in diffs:
        print(f"  - {d}")
    return 1 if (problems or diffs) else 0


def selftest_command(args: argparse.Namespace) -> int:
    from engine.selftest import run_selftest

    return run_selftest(verbose=not args.quiet)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SOZLab CLI", prog="sozlab-cli")
    parser.add_argument("--version", action="version", version=f"SOZLab {__version__}")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sp, window: bool = True) -> None:
        sp.add_argument("--project", required=True, help="Project JSON file")
        sp.add_argument("--workers", type=int, help="CPU worker count (0 or omit for auto)")
        if window:
            sp.add_argument("--dt", type=float, help="ps per trajectory frame (for formats without time stamps)")
            sp.add_argument("--gap-time", type=float, dest="gap_time", help="gap tolerance in ps (stride independent)")

    run_parser = subparsers.add_parser("run", help="Run SOZ analysis")
    common(run_parser)
    run_parser.add_argument("--output", help="Output directory")
    run_parser.add_argument("--stride", type=int, help="Frame stride")
    run_parser.add_argument("--start", type=int, help="Start frame index")
    run_parser.add_argument("--stop", type=int, help="Stop frame index (exclusive)")
    run_parser.add_argument("--no-ids", action="store_true", help="Disable per-frame solvent IDs")
    run_parser.add_argument("--no-per-frame", action="store_true", help="Disable per-frame table storage/export")
    run_parser.add_argument("--no-cache", action="store_true", help="Recompute density maps even if a verified cache exists")
    run_parser.add_argument("--progress", action="store_true", help="Show progress bar")
    run_parser.add_argument("--report", action="store_true", help="Generate report")
    run_parser.set_defaults(func=run_command)

    validate_parser = subparsers.add_parser("validate", help="Validate SOZ selection")
    common(validate_parser, window=False)
    validate_parser.add_argument("--max-frames", type=int, default=200)
    validate_parser.set_defaults(func=validate_command)

    extract_parser = subparsers.add_parser("extract", help="Extract frames by occupancy rule")
    common(extract_parser)
    extract_parser.add_argument("--soz", help="SOZ name to filter")
    extract_parser.add_argument("--rule", help="Rule string, e.g. n_solvent>=1")
    extract_parser.add_argument("--min-run", type=int, help="Minimum consecutive analysed frames")
    extract_parser.add_argument("--min-run-time", type=float, dest="min_run_time", help="Minimum run length in ps")
    extract_parser.add_argument("--gap", type=int, help="Gap tolerance to merge (analysed frames)")
    extract_parser.add_argument("--out", help="Output directory")
    extract_parser.add_argument("--format", default=None, help="Trajectory format: xtc (default), trr (lossless) or dcd")
    extract_parser.add_argument("--prefix", default="extracted", help="Output filename prefix")
    extract_parser.add_argument("--overwrite", action="store_true", help="Overwrite an earlier extraction with the same prefix")
    extract_parser.add_argument("--output", help="Override output dir for logs")
    extract_parser.add_argument("--stride", type=int, help="Frame stride")
    extract_parser.add_argument("--start", type=int, help="Start frame index")
    extract_parser.add_argument("--stop", type=int, help="Stop frame index (exclusive)")
    extract_parser.add_argument("--no-cache", action="store_true")
    extract_parser.add_argument("--from-results", dest="from_results", help="Results directory of an earlier run: "
                                "extract without re-running the analysis (inputs are verified by content)")
    extract_parser.add_argument("--use-project-paths", action="store_true", dest="use_project_paths",
                                help="with --from-results: read topology/trajectory from the project file instead of the recorded paths")
    extract_parser.set_defaults(func=extract_command)

    verify_parser = subparsers.add_parser("verify", help="Check an export's integrity and that its inputs are unchanged")
    verify_parser.add_argument("results", help="Results directory")
    verify_parser.set_defaults(func=verify_command)

    selftest_parser = subparsers.add_parser("selftest", help="Run synthetic known-answer checks on this installation")
    selftest_parser.add_argument("--quiet", action="store_true")
    selftest_parser.set_defaults(func=selftest_command)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (InputsChangedError, ExtractionMismatchError, PartialResultError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        if args.debug:
            raise
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
