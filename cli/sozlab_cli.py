"""Command line interface for SOZLab."""
from __future__ import annotations

import argparse
import sys

from engine.analysis import SOZAnalysisEngine, load_project_json
from engine.export import export_results
from engine.logging_utils import setup_run_logger
from engine.report import generate_report
from engine.validation import validate_project
from engine.extract import select_frames, write_extracted_trajectory


def _progress(current: int, total: int, message: str) -> None:
    percent = int(100 * current / max(total, 1))
    sys.stdout.write(f"\r[{percent:3d}%] {message}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _strip_removed_analysis_options(project) -> None:
    project.hbond_water_bridges.clear()
    project.hbond_hydration.clear()
    project.water_dynamics.clear()


def run_command(args: argparse.Namespace) -> None:
    project = load_project_json(args.project)
    _strip_removed_analysis_options(project)
    if args.output:
        project.outputs.output_dir = args.output
    if args.stride is not None:
        project.analysis.stride = args.stride
    if args.start is not None:
        project.analysis.frame_start = args.start
    if args.stop is not None:
        project.analysis.frame_stop = args.stop
    if args.no_ids:
        project.analysis.store_ids = False
    if args.no_per_frame:
        project.outputs.write_per_frame = False
    if args.workers is not None:
        project.analysis.workers = None if args.workers <= 0 else args.workers

    logger, log_path = setup_run_logger(project.outputs.output_dir)
    logger.info("CLI analysis requested")
    engine = SOZAnalysisEngine(project)
    progress_state = {"total": None}

    def _progress_cb(current: int, total: int, message: str) -> None:
        progress_state["total"] = total
        _progress(current, total, message)

    result = engine.run(progress=_progress_cb if args.progress else None, logger=logger)
    if args.progress and progress_state["total"]:
        total = int(progress_state["total"])
        _progress(max(total - 1, 0), total, "Writing outputs...")
    export_results(result, project)
    if args.progress and progress_state["total"]:
        total = int(progress_state["total"])
        _progress(total, total, "Finalizing outputs...")
    print(f"Log written to {log_path}")
    if result.qc_summary:
        warnings = result.qc_summary.get("preflight", {}).get("warnings", [])
        if warnings:
            print("QC warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        analysis_warnings = result.qc_summary.get("analysis_warnings", [])
        if analysis_warnings:
            print("Analysis warnings:")
            for warning in analysis_warnings[:10]:
                print(f"  - {warning}")
        zero_occupancy = result.qc_summary.get("zero_occupancy_sozs", [])
        if zero_occupancy:
            print("Zero-occupancy SOZs:", ", ".join(zero_occupancy))

    if args.report:
        report_path = generate_report(result, project, command_line=" ".join(sys.argv))
        print(f"Report written to {report_path}")


def validate_command(args: argparse.Namespace) -> None:
    project = load_project_json(args.project)
    _strip_removed_analysis_options(project)
    if args.workers is not None:
        project.analysis.workers = None if args.workers <= 0 else args.workers
    results = validate_project(project, max_frames=args.max_frames)
    for res in results:
        print(
            f"SOZ {res.soz_name}: mismatches {res.mismatched_frames}/{res.total_frames}"
        )
        if res.first_mismatch:
            print(f"  First mismatch frame: {res.first_mismatch['frame']}")


def extract_command(args: argparse.Namespace) -> None:
    project = load_project_json(args.project)
    _strip_removed_analysis_options(project)
    if args.output:
        project.outputs.output_dir = args.output
    if args.workers is not None:
        project.analysis.workers = None if args.workers <= 0 else args.workers

    logger, log_path = setup_run_logger(project.outputs.output_dir)
    logger.info("CLI extract requested")

    if not project.outputs.write_per_frame:
        project.outputs.write_per_frame = True
        logger.info("CLI extract override: write_per_frame forced true for extraction")

    engine = SOZAnalysisEngine(project)
    result = engine.run(logger=logger)
    if not result.soz_results:
        raise ValueError("No SOZ results available for extraction.")

    soz_name = args.soz or next(iter(result.soz_results.keys()))
    if soz_name not in result.soz_results:
        raise ValueError(f"SOZ '{soz_name}' not found.")

    rule = args.rule or project.extraction.rule or "n_solvent>=1"
    per_frame = result.soz_results[soz_name].per_frame
    time_unit = "ps"
    selection = select_frames(
        per_frame,
        rule=rule,
        min_run_length=args.min_run if args.min_run is not None else project.extraction.min_run_length,
        gap_tolerance=args.gap if args.gap is not None else project.extraction.gap_tolerance,
        time_unit=time_unit,
    )

    if project.inputs.trajectory is None:
        raise ValueError("Extraction requires a trajectory input.")

    import MDAnalysis as mda

    universe = mda.Universe(project.inputs.topology, project.inputs.trajectory)
    out_dir = args.out or project.extraction.output_dir
    outputs = write_extracted_trajectory(
        universe,
        selection,
        output_dir=out_dir,
        prefix=args.prefix or "extracted",
        fmt=args.format or project.extraction.output_format,
    )
    print(f"Selected {len(selection.frame_indices)} frames for SOZ '{soz_name}'.")
    print(f"Log written to {log_path}")
    for key, path in outputs.items():
        print(f"{key}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SOZLab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run SOZ analysis")
    run_parser.add_argument("--project", required=True, help="Project JSON file")
    run_parser.add_argument("--output", help="Output directory")
    run_parser.add_argument("--stride", type=int, help="Frame stride")
    run_parser.add_argument("--start", type=int, help="Start frame index")
    run_parser.add_argument("--stop", type=int, help="Stop frame index (exclusive)")
    run_parser.add_argument("--no-ids", action="store_true", help="Disable per-frame solvent IDs")
    run_parser.add_argument(
        "--no-per-frame", action="store_true", help="Disable per-frame table storage/export"
    )
    run_parser.add_argument("--progress", action="store_true", help="Show progress bar")
    run_parser.add_argument("--report", action="store_true", help="Generate report")
    run_parser.add_argument(
        "--workers",
        type=int,
        help="CPU worker count (0 or omit for auto)",
    )
    run_parser.set_defaults(func=run_command)

    validate_parser = subparsers.add_parser("validate", help="Validate SOZ selection")
    validate_parser.add_argument("--project", required=True, help="Project JSON file")
    validate_parser.add_argument("--max-frames", type=int, default=200)
    validate_parser.add_argument(
        "--workers",
        type=int,
        help="CPU worker count (0 or omit for auto)",
    )
    validate_parser.set_defaults(func=validate_command)

    extract_parser = subparsers.add_parser("extract", help="Extract frames by occupancy rule")
    extract_parser.add_argument("--project", required=True, help="Project JSON file")
    extract_parser.add_argument("--soz", help="SOZ name to filter")
    extract_parser.add_argument("--rule", help="Rule string, e.g. n_solvent>=1")
    extract_parser.add_argument("--min-run", type=int, help="Minimum consecutive frames")
    extract_parser.add_argument("--gap", type=int, help="Gap tolerance to merge")
    extract_parser.add_argument("--out", help="Output directory")
    extract_parser.add_argument("--format", default="xtc", help="Trajectory format (xtc)")
    extract_parser.add_argument("--prefix", default="extracted", help="Output filename prefix")
    extract_parser.add_argument("--output", help="Override output dir for logs")
    extract_parser.add_argument(
        "--workers",
        type=int,
        help="CPU worker count (0 or omit for auto)",
    )
    extract_parser.set_defaults(func=extract_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
