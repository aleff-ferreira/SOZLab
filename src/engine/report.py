"""Report generation for SOZLab (Markdown + HTML with figures).

The report is written to be *self-explaining*: every number comes with its uncertainty or the reason it has none,
every quantity can be traced to a definition (appendix), and everything needed to reproduce the run (input
fingerprints, versions, configuration hash, command line) is on the first page.
"""
from __future__ import annotations

import html
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine._version import SCHEMA_VERSION, __version__
from engine.analysis import AnalysisResult
from engine.dictionary import DEFINITIONS
from engine.models import ProjectConfig

# Okabe-Ito colour-blind safe palette
BLUE, ORANGE, GREEN, PURPLE, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#6b7280"


def _fmt(x, digits: int = 4) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{digits}g}"


def _pm(block: Dict[str, object], key: str = "value", unit: str = "") -> str:
    if not block:
        return "n/a"
    v, se = block.get(key), block.get("se")
    if se is None or not np.isfinite(float(se)):
        return f"{_fmt(v)}{unit} (no error estimate)"
    return f"{_fmt(v)} ± {_fmt(se, 2)}{unit}"


def generate_report(result: AnalysisResult, project: ProjectConfig, command_line: str | None = None) -> str:
    output_dir = project.outputs.output_dir
    if not getattr(result, "complete", True):
        output_dir = os.path.join(output_dir, "partial")
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    prov = result.provenance or {}
    qc = result.qc_summary or {}
    L: List[str] = []

    L += ["# SOZLab report", ""]
    if not getattr(result, "complete", True):
        L += ["> **PARTIAL RESULT** - the run was cancelled; numbers cover only the frames analysed.", ""]
    L += ["## Reproducibility", ""]
    L.append(f"- SOZLab {__version__} (schema {SCHEMA_VERSION}); run `{prov.get('run_id', 'n/a')}` at {prov.get('created_utc', 'n/a')}")
    L.append(f"- Command line: `{command_line or prov.get('command_line') or 'GUI'}`")
    for label, key in (("Topology", "topology"), ("Trajectory", "trajectory")):
        fp = (prov.get("inputs") or {}).get(key)
        if fp:
            L.append(f"- {label}: `{fp['name']}` - {fp['size']:,} bytes, SHA-256 `{fp['sha256'][:16]}...` ({fp['hash_mode']})")
    tr = prov.get("trajectory") or {}
    t = (qc.get("timing") or {})
    L.append(f"- Frames analysed: {qc.get('frames_analysed')} of window {qc.get('total_frames')} "
             f"(trajectory {tr.get('n_frames')} frames, {tr.get('n_atoms')} atoms)")
    L.append(f"- Time axis: source *{t.get('source')}*, median interval {_fmt(t.get('dt_median'))} ps, "
             f"{'uniform' if t.get('uniform') else 'irregular'}, trusted: {t.get('trusted')}")
    L.append(f"- Configuration SHA-256: `{str(prov.get('config_sha256', ''))[:16]}...`; algorithms {qc.get('algorithm_versions')}")
    env = prov.get("environment") or {}
    L.append(f"- Environment: Python {env.get('python')}, {env.get('platform')}; packages {env.get('packages')}")
    L.append(f"- Evaluation: {qc.get('parallel', {}).get('mode')} ({qc.get('workers')} worker(s))")
    L.append(f"- Output directory: `{project.outputs.output_dir}`")
    L.append("")

    trust = qc.get("trust") or []
    L += ["## Can these numbers be trusted?", ""]
    if not trust:
        L.append("No caveats were raised.")
    for item in trust:
        icon = {"warning": "**WARNING**", "caution": "caution", "info": "note"}.get(item["level"], item["level"])
        L.append(f"- {icon} ({item['scope']}): {item['message']}")
    L.append("")

    warnings = [w for w in result.warnings if w]
    if warnings:
        L += ["### Warnings", ""] + [f"- {w}" for w in warnings] + [""]

    for name, soz in result.soz_results.items():
        sm = soz.summary
        unc = sm.get("uncertainty", {})
        res = sm.get("residence", {}).get("continuous", {})
        resi = sm.get("residence", {}).get("intermittent", {})
        L += [f"## SOZ: {name}", ""]
        L += ["### Key results", "", "| Quantity | Value | Note |", "|---|---|---|"]
        L.append(f"| Mean n_solvent | {_pm(unc.get('mean_n_solvent', {}))} molecules | autocorrelation-aware SE; "
                 f"{_fmt((unc.get('mean_n_solvent') or {}).get('n_eff'), 3)} effective samples |")
        L.append(f"| Occupancy fraction | {_pm(unc.get('occupancy_fraction', {}))} | fraction of frames with n_solvent >= 1 |")
        L.append(f"| Entry rate | {_pm(unc.get('entry_rate_per_time', {}), unit=' /ps')} | sampling-interval dependent |")
        L.append(f"| Residence time (survival integral) | {_pm(unc.get('tau_survival_integral', {}), unit=' ps')} | "
                 f"{'converged' if res.get('tau_survival_integral_converged') else 'LOWER BOUND (not decayed)'} |")
        L.append(f"| Residence time (exponential fit) | {_fmt(res.get('tau_survival_fit'))} ps | fit r2 = {_fmt(res.get('tau_survival_fit_r2'), 3)} |")
        L.append(f"| Mean complete visit | {_fmt(res.get('mean_visit_duration_complete'))} ps | excludes visits cut by the window |")
        L.append(f"| Kaplan-Meier restricted mean | {_fmt(res.get('km_restricted_mean'))} ps | accounts for right censoring |")
        L.append(f"| Visits (complete / censored) | {res.get('n_visits_complete')} / "
                 f"{(res.get('n_visits') or 0) - (res.get('n_visits_complete') or 0)} | |")
        L.append(f"| Intermittent visits (gap {_fmt(sm.get('gap_tolerance_time'))} ps) | {resi.get('n_visits')} | "
                 f"survival time {_fmt(resi.get('tau_survival_integral'))} ps |")
        ex = sm.get("tau_extrapolation") or {}
        if ex.get("valid"):
            L.append(f"| Continuous-time residence (experimental) | {_fmt(ex['tau_dt0'])} ps | sqrt(dt) extrapolation; {ex['caveat']} |")
        L.append("")
        sens = soz.stride_sensitivity or []
        if len(sens) > 1:
            L += ["### Stride sensitivity", "", "| stride | dt (ps) | mean n | entry rate (/ps) | tau survival (ps) |", "|---|---|---|---|---|"]
            for r in sens:
                L.append(f"| {r['stride_multiplier']}x | {_fmt(r['dt'])} | {_fmt(r['mean_n_solvent'])} | "
                         f"{_fmt(r['entry_rate_per_time'])} | {_fmt(r['tau_survival_integral'])} |")
            L += ["", "Occupancy should not change with stride; residence times and entry rates of diffusing solvent do.", ""]
        L += ["### Summary (raw)", "```"]
        L += [f"{k}: {v}" for k, v in sm.items() if k not in ("uncertainty", "residence", "stride_sensitivity", "n_solvent_hist")]
        L += ["```", ""]
        for fig in _plot_soz_summary(soz, report_dir, name):
            L.append(f"![{name}]({os.path.basename(fig)})")
        L.append("")

    for name, density in result.density_results.items():
        m = density.metadata
        b = m.get("bulk", {}) or {}
        L += [f"## Density map: {name}", ""]
        L.append(f"- Engine `{m.get('engine')}`, grid {m.get('shape')} at {_fmt(m.get('grid_spacing'))} A, {m.get('n_frames')} frames "
                 f"(stride {m.get('stride')}), {m.get('n_blocks', 'n/a')} blocks for the standard error")
        L.append(f"- Bulk reference: **{_fmt(m.get('rho_bulk'))} A^-3** via `{b.get('method')}` (box average {_fmt(b.get('box_mean'))}, "
                 f"far field {_fmt(b.get('far_field'))})")
        L.append(f"- Coverage: fully covered radius {_fmt(m.get('fully_covered_radius'))} A; reliable voxels {_fmt(100 * (m.get('reliable_fraction') or 0), 3)}%; "
                 f"species inside the grid {_fmt(100 * (m.get('fraction_of_species_inside_grid') or 0), 3)}%")
        L.append(f"- Solute motion removed: max drift {_fmt(m.get('solute_com_drift_max'))} A, max rotation {_fmt(m.get('solute_rotation_max_deg'))} deg")
        L.append(f"- Cache: {m.get('cache')}")
        L.append("")
        for fig in _plot_density_summary(density, report_dir, name):
            L.append(f"![{name}]({os.path.basename(fig)})")
        L.append("")

    L += ["## Definitions", ""]
    for d in DEFINITIONS.values():
        unit = f" ({d['unit']})" if d["unit"] else ""
        L += [f"**{d['title']}{unit}.** {d['text']} *Trust:* {d['trust']}", ""]

    report_md = "\n".join(L)
    report_path = os.path.join(report_dir, "report.md")
    with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report_md)
    if project.outputs.report_format.lower() == "html":
        _write_html_report(L, report_dir)
    return report_path


def _package_versions() -> Dict[str, str]:
    versions = {}
    for name in ("MDAnalysis", "numpy", "pandas", "pyqtgraph", "PyQt6"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not available"
    return versions


_CSS = ("body{font:15px/1.55 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#1f2937}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #d1d5db;padding:.3rem .6rem;text-align:left}"
        "th{background:#f3f4f6}code,pre{background:#f3f4f6;border-radius:4px}pre{padding:.7rem;overflow:auto}"
        "code{padding:.1rem .3rem}img{max-width:100%}blockquote{border-left:4px solid #D55E00;margin:1rem 0;padding:.2rem 1rem;background:#fff7ed}")


def _inline(text: str) -> str:
    import re

    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def _write_html_report(lines: List[str], report_dir: str) -> None:
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>SOZLab report</title><style>{_CSS}</style></head><body>"]
    in_code, in_table, table_header_done = False, False, False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            out.append("<pre>" if in_code else "</pre>")
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            if not in_table:
                out.append("<table>")
                in_table, table_header_done = True, False
            tag = "th" if not table_header_done else "td"
            table_header_done = True
            out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("# "):
            out.append(f"<h1>{_inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{_inline(line[4:])}</h3>")
        elif line.startswith("!["):
            path = line.split("](")[1].rstrip(")")
            out.append(f'<img src="{html.escape(path)}" alt=""/>')
        elif line.startswith("> "):
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.startswith("- "):
            out.append(f"<p>&bull; {_inline(line[2:])}</p>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    if in_table:
        out.append("</table>")
    out.append("</body></html>")
    with open(os.path.join(report_dir, "report.html"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(out))


def _time_scale_to_ns(time_unit: str) -> float:
    """Factor converting the trajectory's native time unit to ns (MDAnalysis stores ps; Michaud-Agrawal et al. 2011)."""
    unit = time_unit.lower() if time_unit else "ps"
    if unit == "ps":
        return 1e-3
    if unit == "fs":
        return 1e-6
    if unit == "ns":
        return 1.0
    return 1e-3


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def _plot_soz_summary(soz, report_dir: str, name: str) -> List[str]:
    fig_paths: List[str] = []
    per_frame = soz.per_frame
    dt = float(soz.summary.get("dt", 1.0))
    scale_to_ns = _time_scale_to_ns(soz.summary.get("time_unit", "ps"))

    if not per_frame.empty:
        time_ns = per_frame["time"].to_numpy(dtype=float) * scale_to_ns
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(time_ns, per_frame["n_solvent"], color=BLUE, linewidth=1.2)
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("N solvent in zone")
        ax.set_title(f"{name}: occupancy over time")
        _style(ax)
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_nsolvent.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(per_frame["n_solvent"], bins=20, color=GREEN, alpha=0.85)
        ax.set_xlabel("N solvent in zone")
        ax.set_ylabel("Frames")
        ax.set_title(f"{name}: occupancy histogram")
        _style(ax)
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_hist.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    per_solvent = soz.per_solvent
    if not per_solvent.empty:
        top = per_solvent.head(10)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(top["solvent_id"], top["occupancy_pct"], color=ORANGE)
        ax.set_xlabel("Occupancy (% of frames)")
        ax.set_title(f"{name}: most persistent solvent")
        ax.invert_yaxis()
        _style(ax)
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_top_solvents.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    surv = getattr(soz, "survival", None)
    if surv is not None and len(surv) > 1 and np.isfinite(surv["tau"]).any():
        ok = (surv["origins_continuous"] >= 20) & np.isfinite(surv["S_continuous"])
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(surv["tau"][ok] * scale_to_ns, surv["S_continuous"][ok], color=PURPLE, label="continuous")
        ok2 = (surv["origins_intermittent"] >= 20) & np.isfinite(surv["S_intermittent"])
        if ok2.any() and not np.allclose(surv["S_intermittent"][ok2], surv["S_continuous"][ok2], equal_nan=True):
            ax.plot(surv["tau"][ok2] * scale_to_ns, surv["S_intermittent"][ok2], color=ORANGE, linestyle="--", label="intermittent")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.05)
        ax.set_xlabel("Lag τ (ns)")
        ax.set_ylabel("S(τ)")
        ax.set_title(f"{name}: survival correlation function")
        ax.legend(frameon=False)
        _style(ax)
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_survival.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)

    # Complementary CDF of visit durations (a different quantity from S(tau): it counts visits, not origins)
    durations = []
    for lengths in soz.residence_cont.values():
        durations.extend([length * dt * scale_to_ns for length in lengths])
    if durations:
        durations = np.array(durations, dtype=float)
        durations = durations[np.isfinite(durations)]
        durations.sort()
        if durations.size:
            ccdf = 1.0 - np.arange(1, len(durations) + 1) / len(durations)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.step(durations, ccdf, where="post", color=PURPLE)
            ax.set_xlabel("Visit duration (ns)")
            ax.set_ylabel("Fraction of visits longer")
            ax.set_title(f"{name}: visit-duration CCDF")
            _style(ax)
            fig.tight_layout()
            path = os.path.join(report_dir, f"{name}_residence_ccdf.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            fig_paths.append(path)
    return fig_paths


def _plot_density_summary(density, report_dir: str, name: str) -> List[str]:
    fig_paths: List[str] = []
    slices = density.slices or {}
    if not slices:
        return fig_paths
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
    axes = axes.ravel()
    for ax, title, key in zip(axes, ["XY (centre)", "XZ (centre)", "YZ (centre)", "Max projection"],
                              ["xy", "xz", "yz", "max_projection"]):
        data = slices.get(key)
        if data is None:
            ax.axis("off")
            continue
        im = ax.imshow(np.asarray(data).T, origin="lower", cmap="viridis")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ρ (Å⁻³)")
    fig.suptitle(f"Density map: {name}")
    fig.tight_layout()
    path = os.path.join(report_dir, f"{name}_density_panel.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    fig_paths.append(path)

    g, mask = getattr(density, "g", None), getattr(density, "reliable", None)
    if g is not None and np.isfinite(g).any():
        k = g.shape[2] // 2
        fig, ax = plt.subplots(figsize=(4.6, 4))
        view = np.where(mask[:, :, k], g[:, :, k], np.nan) if mask is not None else g[:, :, k]
        im = ax.imshow(view.T, origin="lower", cmap="RdBu_r", vmin=0, vmax=max(2.0, float(np.nanmax(view)) if np.isfinite(view).any() else 2.0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="g(r) = ρ/ρ_bulk")
        ax.set_title(f"{name}: g(r), reliable voxels only")
        fig.tight_layout()
        path = os.path.join(report_dir, f"{name}_g_slice.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        fig_paths.append(path)
    return fig_paths
