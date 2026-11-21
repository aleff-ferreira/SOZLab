"""The GUI and the CLI are two front ends of one engine: given the same project they must produce identical results."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.gui.conftest import run

pytestmark = [pytest.mark.gui, pytest.mark.integration]
ROOT = Path(__file__).parents[2]


def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src"), env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, str(ROOT / "cli" / "sozlab_cli.py"), *args], capture_output=True, text=True, env=env, timeout=300)


def test_gui_and_cli_export_identical_numbers(qtbot, loaded, tmp_path):
    from engine.analysis import write_project_json

    s = loaded.session
    s.run.density = False
    s.zone.cutoff = 3.8
    s.set_run(stride=1)
    run(qtbot, loaded)
    gui_out = str(tmp_path / "gui")
    with qtbot.waitSignal(s.taskDone, timeout=60000):
        s.export_data(gui_out)
    proj = tmp_path / "project.json"
    write_project_json(s.run_project, str(proj))
    cli_out = str(tmp_path / "cli")
    r = _cli("run", "--project", str(proj), "--output", cli_out)
    assert r.returncode == 0, r.stderr
    name = next(iter(s.result.soz_results))
    a = pd.read_csv(os.path.join(gui_out, f"soz_{name}", "per_frame.csv"))
    b = pd.read_csv(os.path.join(cli_out, f"soz_{name}", "per_frame.csv"))
    cols = ["frame", "time", "n_solvent", "entries", "exits"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert (a["solvent_ids"] == b["solvent_ids"]).all()  # not just the counts: the same molecules in every frame
    sa = json.load(open(os.path.join(gui_out, f"soz_{name}", "summary.json"), encoding="utf-8"))
    sb = json.load(open(os.path.join(cli_out, f"soz_{name}", "summary.json"), encoding="utf-8"))
    for key in ("mean_n_solvent", "occupancy_fraction", "entry_rate_per_time", "n_entry_events", "n_exit_events"):
        assert sa[key] == pytest.approx(sb[key], rel=1e-12), key
    ra, rb = sa["residence"]["continuous"], sb["residence"]["continuous"]
    for key in ("tau_survival_integral", "n_visits", "km_restricted_mean"):
        assert ra[key] == pytest.approx(rb[key], rel=1e-12), key
    # identical configuration identity as well
    pa = json.load(open(os.path.join(gui_out, "provenance.json"), encoding="utf-8"))
    pb = json.load(open(os.path.join(cli_out, "provenance.json"), encoding="utf-8"))
    assert pa["config_sha256"] == pb["config_sha256"]
    assert pa["inputs"]["trajectory"]["sha256"] == pb["inputs"]["trajectory"]["sha256"]
