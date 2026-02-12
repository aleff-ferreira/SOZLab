from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.viz_3d import Density3DWidget

pytestmark = pytest.mark.gui


class _ToggleWidget:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


def test_refresh_pick_controls_state_ready_requires_pick_for_add() -> None:
    measure_mode = _ToggleWidget()
    clear_btn = _ToggleWidget()
    auto_label = _ToggleWidget()
    label_edit = _ToggleWidget()
    add_btn = _ToggleWidget()
    widget = SimpleNamespace(
        _is_ngl_runtime_ready=lambda: True,
        _ngl_runtime_ready=True,
        _ngl_had_error=False,
        _last_pick_event=None,
        measure_mode_label=measure_mode,
        clear_measure_btn=clear_btn,
        auto_label_check=auto_label,
        custom_label_edit=label_edit,
        add_label_btn=add_btn,
        _set_pick_status=MagicMock(),
    )
    widget._has_valid_label_pick_target = lambda: Density3DWidget._has_valid_label_pick_target(widget)

    Density3DWidget._refresh_pick_controls_state(widget)

    assert measure_mode.enabled is True
    assert clear_btn.enabled is True
    assert auto_label.enabled is True
    assert label_edit.enabled is True
    assert add_btn.enabled is False
    widget._set_pick_status.assert_not_called()


def test_refresh_pick_controls_state_not_ready_disables_controls() -> None:
    measure_mode = _ToggleWidget()
    clear_btn = _ToggleWidget()
    auto_label = _ToggleWidget()
    label_edit = _ToggleWidget()
    add_btn = _ToggleWidget()
    widget = SimpleNamespace(
        _is_ngl_runtime_ready=lambda: False,
        _ngl_runtime_ready=False,
        _ngl_had_error=False,
        _last_pick_event={"kind": "atom"},
        measure_mode_label=measure_mode,
        clear_measure_btn=clear_btn,
        auto_label_check=auto_label,
        custom_label_edit=label_edit,
        add_label_btn=add_btn,
        _set_pick_status=MagicMock(),
    )
    widget._has_valid_label_pick_target = lambda: Density3DWidget._has_valid_label_pick_target(widget)

    Density3DWidget._refresh_pick_controls_state(widget)

    assert measure_mode.enabled is False
    assert clear_btn.enabled is False
    assert auto_label.enabled is False
    assert label_edit.enabled is False
    assert add_btn.enabled is False
    widget._set_pick_status.assert_called_once()


def test_on_add_custom_label_requires_pick_feedback() -> None:
    widget = SimpleNamespace(
        custom_label_edit=SimpleNamespace(text=lambda: "MyLabel"),
        _last_pick_event={},
        _set_pick_status=MagicMock(),
        _set_notice=MagicMock(),
        _js_invoke=MagicMock(),
    )
    widget._has_valid_label_pick_target = lambda: Density3DWidget._has_valid_label_pick_target(widget)

    Density3DWidget._on_add_custom_label(widget)

    widget._js_invoke.assert_not_called()
    widget._set_pick_status.assert_called_once()
    widget._set_notice.assert_called_once()


def test_on_add_custom_label_rejects_invalid_pick_coordinates() -> None:
    widget = SimpleNamespace(
        custom_label_edit=SimpleNamespace(text=lambda: "Hotspot"),
        _last_pick_event={"kind": "density", "x": None, "y": 2.0, "z": 3.0},
        _set_pick_status=MagicMock(),
        _set_notice=MagicMock(),
        _js_invoke=MagicMock(),
    )
    widget._has_valid_label_pick_target = lambda: Density3DWidget._has_valid_label_pick_target(widget)

    Density3DWidget._on_add_custom_label(widget)

    widget._js_invoke.assert_not_called()
    widget._set_pick_status.assert_called_once()
    widget._set_notice.assert_called_once()


def test_handle_pick_event_switches_to_pick_section() -> None:
    widget = SimpleNamespace(
        _set_notice=MagicMock(),
        _set_pick_status=MagicMock(),
        _js_invoke=MagicMock(),
        _set_context_panel_visible=MagicMock(),
        _set_active_context_section=MagicMock(),
        _focus_selection=None,
    )
    event = {
        "kind": "atom",
        "atomName": "CA",
        "resname": "ALA",
        "resno": 10,
        "chain": "A",
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "atomIndex": 8,
    }

    Density3DWidget._handle_pick_event(widget, event)

    widget._set_active_context_section.assert_called_once_with("pick")
    widget._set_pick_status.assert_called_once()
    widget._js_invoke.assert_called_once_with("highlightSelection", "@8")
