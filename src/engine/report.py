"""Report generation for SOZLab."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.analysis import AnalysisResult
from engine.serialization import to_jsonable
from engine.models import ProjectConfig


def generate_report(result: AnalysisResult, project: ProjectConfig, command_line: str | None = None) -> str:
    output_dir = project.outputs.output_dir
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    report_lines: List[str] = []
    report_lines.append("# SOZLab Report")
    report_lines.append("")
    report_lines.append("## Reproducibility")
    report_lines.append(f"- Command line: {command_line or 'N/A'}")
    report_lines.append("- Project file: metadata.json")
    report_lines.append("- Output directory: {}".format(output_dir))
    report_lines.append("")
    report_lines.append("### Package Versions")
    for name, version in _package_versions().items():
        report_lines.append(f"- {name}: {version}")
    report_lines.append("")
    report_lines.append("### Config Snapshot")
    report_lines.append("```")
    report_lines.append(str(project.to_dict()))
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("### QC Summary")
    report_lines.append("```")
    report_lines.append(str(to_jsonable(result.qc_summary)))
    report_lines.append("```")
    report_lines.append("")

    for name, soz in result.soz_results.items():
        report_lines.append(f"## SOZ: {name}")
        report_lines.append("")
        report_lines.append("### Summary")
        report_lines.append("```")
        report_lines.append(str(soz.summary))
        report_lines.append("```")
        report_lines.append("")

        fig_paths = _plot_soz_summary(soz, report_dir, name)
        for fig in fig_paths:
            report_lines.append(f"![{name}]({os.path.basename(fig)})")
        report_lines.append("")

    for name, bridge in result.distance_bridge_results.items():
        report_lines.append(f"## Distance Bridge: {name}")
        report_lines.append("")
        fig_paths = _plot_bridge_summary(
            bridge,
            report_dir,
            name,
            title_prefix="Distance Bridge (distance-cutoff intersection)",
        )
        for fig in fig_paths:
            report_lines.append(f"![{name}]({os.path.basename(fig)})")
        report_lines.append("")

    for name, bridge in result.hbond_bridge_results.items():
        report_lines.append(f"## H-bond Water Bridge: {name}")
        report_lines.append("")
        fig_paths = _plot_bridge_summary(
            bridge,
            report_dir,
            name,
            title_prefix="H-bond Water Bridge",
        )
        for fig in fig_paths:
            report_lines.append(f"![{name}]({os.path.basename(fig)})")
        report_lines.append("")


    for name, hydration in result.hbond_hydration_results.items():
        report_lines.append(f"## H-bond Hydration: {name}")
        report_lines.append("")
        fig_paths = _plot_hydration_summary(
            hydration,
            report_dir,
            name,
            title_prefix="H-bond Hydration",
        )
        for fig in fig_paths:
            report_lines.append(f"![{name}]({os.path.basename(fig)})")
        report_lines.append("")

    for name, density in result.density_results.items():
        report_lines.append(f"## Density Map: {name}")
        report_lines.append("")
        fig_paths = _plot_density_summary(density, report_dir, name)
        for fig in fig_paths:
            report_lines.append(f"![{name}]({os.path.basename(fig)})")
        report_lines.append("")

    for name, dynamics in result.water_dynamics_results.items():
        report_lines.append(f"## Water Dynamics: {name}")
        report_lines.append("")
        fig_paths = _plot_water_dynamics_summary(dynamics, report_dir, name)
        for fig in fig_paths:
            report_lines.append(f"![{name}]({os.path.basename(fig)})")
        report_lines.append("")

    report_md = "\n".join(report_lines)
    report_path = os.path.join(report_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_md)

    if project.outputs.report_format.lower() == "html":
        _write_html_report(report_lines, report_dir)

    return report_path


def _package_versions() -> Dict[str, str]:
    versions = {}
    modules = {
        "MDAnalysis": "MDAnalysis",
        "numpy": "numpy",
        "pandas": "pandas",
        "pyqtgraph": "pyqtgraph",
        "PyQt6": "PyQt6",
    }
    for name, module_name in modules.items():
        try:
            module = __import__(module_name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not available"
    return versions


def _write_html_report(lines: List[str], report_dir: str) -> None:
    html_lines = ["<html><body>"]
    in_code_block = False
    for line in lines:
        if line.startswith("```"):
            in_code_block = not in_code_block
            html_lines.append("<pre>" if in_code_block else "</pre>")
            continue
        if in_code_block:
            html_lines.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("![") and "](" in line:
            path = line.split("](")[1].rstrip(")")
            html_lines.append(f"<img src=\"{path}\" style=\"max-width:100%;\"/>")
        elif line.startswith("- "):
            html_lines.append(f"<p>{line}</p>")
        elif not line:
            html_lines.append("<br/>")
        else:
            html_lines.append(f"<p>{line}</p>")
    html_lines.append("</body></html>")

    html_path = os.path.join(report_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_lines))


def _plot_soz_summary(soz, report_dir: str, name: str) -> List[str]:
    fig_paths: List[str] = []
    per_frame = soz.per_frame

    if not per_frame.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(per_frame["time"], per_frame["n_solvent"], color="#2b6cb0")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("N solvent")
        ax.set_title(f"{name}: n_solvent vs time")
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_nsolvent.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(per_frame["n_solvent"], bins=20, color="#38a169", alpha=0.8)
        ax.set_xlabel("N solvent")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}: occupancy histogram")
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_hist.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    per_solvent = soz.per_solvent
    if not per_solvent.empty:
        top = per_solvent.head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(top["solvent_id"], top["occupancy_pct"], color="#dd6b20")
        ax.set_xlabel("Occupancy %")
        ax.set_title(f"{name}: top solvents")
        ax.invert_yaxis()
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_top_solvents.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    durations = []
    dt = float(soz.summary.get("dt", 1.0))
    for lengths in soz.residence_cont.values():
        durations.extend([length * dt for length in lengths])
    if durations:
        durations = np.array(durations, dtype=float)
        durations.sort()
        survival = 1.0 - np.arange(1, len(durations) + 1) / len(durations)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.step(durations, survival, where="post", color="#805ad5")
        ax.set_xlabel("Residence (time)")
        ax.set_ylabel("Survival")
        ax.set_title(f"{name}: residence survival")
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_residence_ccdf.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    return fig_paths


def _plot_bridge_summary(bridge, report_dir: str, name: str, title_prefix: str) -> List[str]:
    fig_paths: List[str] = []
    per_frame = bridge.per_frame
    if not per_frame.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(per_frame["time"], per_frame["n_solvent"], color="#2b6cb0")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("Bridging solvent count")
        ax.set_title(f"{title_prefix}: # bridging solvents")
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_bridge_timeseries.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    per_solvent = bridge.per_solvent
    if not per_solvent.empty:
        top = per_solvent.head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(top["solvent_id"], top["occupancy_pct"], color="#dd6b20")
        ax.set_xlabel("Occupancy %")
        ax.set_title(f"{title_prefix}: top bridges")
        ax.invert_yaxis()
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_bridge_top.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    durations = []
    dt = float(bridge.summary.get("dt", 1.0))
    for lengths in bridge.residence_cont.values():
        durations.extend([length * dt for length in lengths])
    if durations:
        durations = np.array(durations, dtype=float)
        durations.sort()
        survival = 1.0 - np.arange(1, len(durations) + 1) / len(durations)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.step(durations, survival, where="post", color="#805ad5")
        ax.set_xlabel("Residence (time)")
        ax.set_ylabel("Survival")
        ax.set_title(f"{title_prefix}: residence survival")
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_bridge_residence.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)
    return fig_paths


def _plot_hydration_summary(hydration, report_dir: str, name: str, title_prefix: str) -> List[str]:
    fig_paths: List[str] = []
    table = hydration.table
    if table is None or table.empty:
        return fig_paths

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(table["resid"], table["freq_total"], color="#2b6cb0", linewidth=1.5)
    ax.set_xlabel("Residue ID")
    ax.set_ylabel("Freq (total)")
    ax.set_title(f"{title_prefix}: frequency profile")
    fig.tight_layout()
    path = os.path.join(report_dir, f"{name}_hydration_profile.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    fig_paths.append(path)

    top = table.sort_values(by="freq_total", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(
        [f"{row.resname}{row.resid}" for row in top.itertuples()],
        top["freq_total"],
        color="#38a169",
    )
    ax.set_xlabel("Freq (total)")
    ax.set_title(f"{title_prefix}: top residues")
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(report_dir, f"{name}_hydration_top.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    fig_paths.append(path)

    return fig_paths


def _plot_density_summary(density, report_dir: str, name: str) -> List[str]:
    fig_paths: List[str] = []
    slices = density.slices or {}
    if not slices:
        return fig_paths

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()
    titles = ["XY", "XZ", "YZ", "Max Projection"]
    keys = ["xy", "xz", "yz", "max_projection"]
    for ax, title, key in zip(axes, titles, keys):
        data = slices.get(key)
        if data is None:
            ax.axis("off")
            continue
        im = ax.imshow(data.T, origin="lower", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Density Map: {name}")
    fig.tight_layout()
    path = os.path.join(report_dir, f"{name}_density_panel.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    fig_paths.append(path)
    return fig_paths


def _plot_water_dynamics_summary(dynamics, report_dir: str, name: str) -> List[str]:
    fig_paths: List[str] = []
    rows = 1 + int(dynamics.hbl is not None) + int(dynamics.wor is not None)
    fig, axes = plt.subplots(rows, 1, figsize=(6, 2.5 * rows))
    if rows == 1:
        axes = [axes]
    ax = axes[0]
    sp = dynamics.sp_tau
    if not sp.empty:
        ax.plot(sp["tau"], sp["survival"], color="#2b6cb0")
    ax.set_xlabel("Tau")
    ax.set_ylabel("Survival")
    ax.set_title("SP(τ)")
    idx = 1
    if dynamics.hbl is not None:
        ax = axes[idx]
        ax.plot(dynamics.hbl["tau"], dynamics.hbl["correlation"], color="#805ad5")
        ax.set_xlabel("Tau")
        ax.set_ylabel("Correlation")
        ax.set_title("H-bond lifetime")
        idx += 1
    if dynamics.wor is not None:
        ax = axes[idx]
        ax.plot(dynamics.wor["tau"], dynamics.wor["correlation"], color="#38a169")
        ax.set_xlabel("Tau")
        ax.set_ylabel("Correlation")
        ax.set_title("Water orientational relaxation")
    if rows > 0:
        fig.suptitle(f"Water Dynamics: {name}")
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_water_dynamics.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)
    else:
        plt.close(fig)
    return fig_paths
