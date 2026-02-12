from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.main import MainWindow
import app.main as main_module

pytestmark = pytest.mark.gui


def test_refresh_project_doctor_if_initialized_uses_silent_mode() -> None:
    window = SimpleNamespace(
        _preflight_report=object(),
        _analysis_running=False,
        _run_project_doctor=MagicMock(),
    )

    MainWindow._refresh_project_doctor_if_initialized(window)

    window._run_project_doctor.assert_called_once_with(silent=True)


def test_refresh_project_doctor_if_initialized_skips_without_report() -> None:
    window = SimpleNamespace(
        _preflight_report=None,
        _analysis_running=False,
        _run_project_doctor=MagicMock(),
    )

    MainWindow._refresh_project_doctor_if_initialized(window)

    window._run_project_doctor.assert_not_called()


def test_run_project_doctor_silent_skips_status_message(monkeypatch) -> None:
    report = SimpleNamespace(ok=True, errors=[], warnings=[])
    monkeypatch.setattr(main_module, "run_preflight", lambda *_args, **_kwargs: report)

    window = SimpleNamespace(
        _ensure_project=MagicMock(return_value=True),
        state=SimpleNamespace(project=object()),
        _ensure_preflight_universe=MagicMock(return_value=object()),
        _update_project_doctor_ui=MagicMock(),
        doctor_status_label=MagicMock(),
        doctor_text=MagicMock(),
        doctor_seed_table=MagicMock(),
        status_bar=MagicMock(),
        run_logger=None,
        _preflight_report=None,
    )

    ok = MainWindow._run_project_doctor(window, silent=True)

    assert ok is True
    assert window._preflight_report is report
    window._update_project_doctor_ui.assert_called_once_with(report)
    window.status_bar.showMessage.assert_not_called()


def test_analysis_finished_refreshes_project_doctor() -> None:
    window = SimpleNamespace(
        analysis_thread=MagicMock(),
        _update_results_view=MagicMock(),
        _run_progress_total=None,
        run_project=None,
        state=SimpleNamespace(project=None),
        status_bar=MagicMock(),
        report_text=MagicMock(),
        _set_run_ui_state=MagicMock(),
        _refresh_project_doctor_if_initialized=MagicMock(),
        _refresh_log_view=MagicMock(),
        _set_active_step=MagicMock(),
        run_logger=None,
        _timeline_stats_cache={},
        _timeline_event_cache={},
    )
    result = SimpleNamespace(soz_results={})

    MainWindow._on_analysis_finished(window, result)

    window._refresh_project_doctor_if_initialized.assert_called_once()


def test_update_project_doctor_ui_solvent_badge_uses_residue_counts() -> None:
    window = SimpleNamespace(
        doctor_status_label=MagicMock(),
        _set_status_badge=MagicMock(),
        doctor_errors_badge=object(),
        doctor_warnings_badge=object(),
        doctor_solvent_badge=object(),
        doctor_probe_badge=object(),
        doctor_text=MagicMock(),
        doctor_seed_table=MagicMock(),
        qc_inspector_text=MagicMock(),
    )
    report = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        solvent_summary={
            "solvent_matches": ["WAT"],
            "solvent_residue_counts": {"WAT": 41516},
            "probe_summary": {"probe_atom_count": 41516},
        },
        pbc_summary={},
        gmx_summary={},
        trajectory_summary={},
        selection_checks={},
        seed_checks={},
    )

    MainWindow._update_project_doctor_ui(window, report)

    assert any(
        call.args[0] is window.doctor_solvent_badge and call.args[1] == "Solvent atoms 41516"
        for call in window._set_status_badge.call_args_list
    )


def test_sync_wizard_selection_defs_to_project_updates_selection_count() -> None:
    project = SimpleNamespace(selections={})
    window = SimpleNamespace(
        state=SimpleNamespace(project=project),
        wizard_soz_name=SimpleNamespace(text=lambda: "SOZ_1"),
        wizard_seed_a=SimpleNamespace(text=lambda: "protein and resid 10 and name CA"),
        wizard_seed_b=SimpleNamespace(text=lambda: "protein and resid 20 and name CA"),
        wizard_seed_a_unique=SimpleNamespace(isChecked=lambda: True),
        wizard_seed_b_unique=SimpleNamespace(isChecked=lambda: False),
        _wizard_selection_label=lambda which: f"SOZ_1_selection_{'a' if which.upper() == 'A' else 'b'}",
        _wizard_synced_selection_a=None,
        _wizard_synced_selection_b=None,
        _refresh_selection_state_ui=MagicMock(),
    )

    MainWindow._sync_wizard_selection_defs_to_project(window)

    assert set(project.selections.keys()) == {"SOZ_1_selection_a", "SOZ_1_selection_b"}
    window._refresh_selection_state_ui.assert_called_once()


def test_refresh_selection_state_ui_updates_badge_and_refreshes() -> None:
    project = SimpleNamespace(selections={"A": object(), "B": object()})
    window = SimpleNamespace(
        state=SimpleNamespace(project=project),
        _refresh_selection_combos=MagicMock(),
        _refresh_project_doctor_if_initialized=MagicMock(),
    )

    MainWindow._refresh_selection_state_ui(window)

    window._refresh_selection_combos.assert_called_once()
    window._refresh_project_doctor_if_initialized.assert_called_once()


def test_add_soz_from_builder_refreshes_project_doctor_when_applied() -> None:
    window = SimpleNamespace(
        _ensure_project=MagicMock(return_value=True),
        _apply_wizard_to_project=MagicMock(return_value=True),
        _refresh_project_ui=MagicMock(),
        _refresh_project_doctor_if_initialized=MagicMock(),
    )

    MainWindow._add_soz_from_builder(window)

    window._apply_wizard_to_project.assert_called_once_with(update_existing=True)
    window._refresh_project_ui.assert_called_once()
    window._refresh_project_doctor_if_initialized.assert_called_once()


def test_add_soz_from_builder_skips_doctor_refresh_when_not_applied() -> None:
    window = SimpleNamespace(
        _ensure_project=MagicMock(return_value=True),
        _apply_wizard_to_project=MagicMock(return_value=False),
        _refresh_project_ui=MagicMock(),
        _refresh_project_doctor_if_initialized=MagicMock(),
    )

    MainWindow._add_soz_from_builder(window)

    window._apply_wizard_to_project.assert_called_once_with(update_existing=True)
    window._refresh_project_ui.assert_called_once()
    window._refresh_project_doctor_if_initialized.assert_not_called()


def test_distance_bridge_form_change_refreshes_doctor() -> None:
    bridge = SimpleNamespace(
        name="distance_bridge",
        selection_a="",
        selection_b="",
        cutoff_a=3.5,
        cutoff_b=3.5,
        unit="A",
        atom_mode="probe",
    )
    item = MagicMock()
    window = SimpleNamespace(
        _distance_bridge_form_updating=False,
        state=SimpleNamespace(project=SimpleNamespace(distance_bridges=[bridge])),
        distance_bridge_list=SimpleNamespace(currentRow=lambda: 0, item=lambda _idx: item),
        distance_bridge_name_edit=SimpleNamespace(text=lambda: "distance_bridge"),
        distance_bridge_cutoff_a_spin=SimpleNamespace(value=lambda: 3.5),
        distance_bridge_cutoff_b_spin=SimpleNamespace(value=lambda: 3.5),
        distance_bridge_unit_combo=SimpleNamespace(currentText=lambda: "A"),
        distance_bridge_probe_combo=SimpleNamespace(currentText=lambda: "probe"),
        _selection_combo_value=lambda combo: "SOZ_1_selection_a" if combo == "A" else "SOZ_1_selection_b",
        distance_bridge_sel_a_combo="A",
        distance_bridge_sel_b_combo="B",
        _distance_bridge_item_text=lambda _bridge: "distance_bridge item",
        _validate_distance_bridge_form=MagicMock(),
        _refresh_project_doctor_if_initialized=MagicMock(),
    )

    MainWindow._on_distance_bridge_form_changed(window)

    assert bridge.selection_a == "SOZ_1_selection_a"
    assert bridge.selection_b == "SOZ_1_selection_b"
    window._refresh_project_doctor_if_initialized.assert_called_once()


def test_apply_wizard_selection_to_combo_forces_form_sync_and_doctor_refresh() -> None:
    signal = SimpleNamespace(emit=MagicMock())
    combo = SimpleNamespace(
        currentTextChanged=signal,
        currentText=lambda: "SOZ_1_selection_a",
    )
    window = SimpleNamespace(
        _ensure_wizard_selection=MagicMock(return_value="SOZ_1_selection_a"),
        _set_selection_combo_value=MagicMock(),
        _refresh_project_doctor_if_initialized=MagicMock(),
    )

    MainWindow._apply_wizard_selection_to_combo(window, combo, "A")

    window._ensure_wizard_selection.assert_called_once_with("A")
    window._set_selection_combo_value.assert_called_once_with(combo, "SOZ_1_selection_a")
    signal.emit.assert_called_once_with("SOZ_1_selection_a")
    window._refresh_project_doctor_if_initialized.assert_called_once()


def test_apply_wizard_selection_to_combo_skips_when_selection_empty() -> None:
    signal = SimpleNamespace(emit=MagicMock())
    combo = SimpleNamespace(
        currentTextChanged=signal,
        currentText=lambda: "",
    )
    window = SimpleNamespace(
        _ensure_wizard_selection=MagicMock(return_value=None),
        _set_selection_combo_value=MagicMock(),
        _refresh_project_doctor_if_initialized=MagicMock(),
    )

    MainWindow._apply_wizard_selection_to_combo(window, combo, "B")

    window._ensure_wizard_selection.assert_called_once_with("B")
    window._set_selection_combo_value.assert_not_called()
    signal.emit.assert_not_called()
    window._refresh_project_doctor_if_initialized.assert_not_called()
