"""Reusable Workbench widgets: cards, parameter fields (with modified-dot + reset), stat tiles with trust badges,
stepper, chips, drop zone, toasts. Everything is themed through :class:`app.workbench.theme.Theme`."""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from app.workbench import icons
from app.workbench.theme import Theme, polish, set_prop

Qt = QtCore.Qt


def _theme() -> Theme:
    return Theme.instance()


def make_label(text: str = "", role: str = "body", word_wrap: bool = False, selectable: bool = False) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setProperty("role", role)
    lab.setWordWrap(word_wrap)
    if selectable:
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return lab


def dispose(widget: QtWidgets.QWidget) -> None:
    """Remove a widget now (hidden and unparented, so it can never be painted again) and free it later."""
    widget.hide()
    widget.setParent(None)
    widget.deleteLater()


class ElideLabel(QtWidgets.QLabel):
    """Single-line label that elides with an ellipsis instead of widening its container (full text in the tooltip)."""

    def __init__(self, text: str = "", role: str = "caption", parent=None):
        super().__init__("", parent)
        self.setProperty("role", role)
        self._full = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.set_full_text(text)

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self.update()

    def sizeHint(self):  # noqa: N802
        fm = self.fontMetrics()
        return QtCore.QSize(min(fm.horizontalAdvance(self._full) + 2, 400), fm.height() + 2)

    def minimumSizeHint(self):  # noqa: N802
        return QtCore.QSize(0, self.fontMetrics().height() + 2)

    def paintEvent(self, _):  # noqa: N802
        p = QtGui.QPainter(self)
        p.setFont(self.font())
        p.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        txt = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, self.width())
        p.drawText(self.rect(), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), txt)


class FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps (for chips)."""

    def __init__(self, parent=None, margin: int = 0, hspacing: int = 6, vspacing: int = 6):
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self._h, self._v = hspacing, vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._layout(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect, test_only):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > r.right() + 1 and line_h > 0:
                x = r.x()
                y += line_h + self._v
                line_h = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()

    def clear(self):
        while self._items:
            it = self._items.pop()
            w = it.widget()
            if w is not None:
                w.setParent(None)
                dispose(w)


# ------------------------------------------------------------------------------------------------ chips
class Chip(QtWidgets.QLabel):
    def __init__(self, text: str = "", kind: str = "neutral", parent=None):
        super().__init__(text, parent)
        self.setObjectName("Chip")
        self.set_kind(kind)

    def set_kind(self, kind: str) -> None:
        set_prop(self, "kind", kind)

    def set(self, text: str, kind: str = "neutral") -> None:
        self.setText(text)
        self.set_kind(kind)


# ------------------------------------------------------------------------------------------------ help popup
class DefinitionPopup(QtWidgets.QFrame):
    """In-context scientific definition (what it measures, units, when to trust it)."""

    def __init__(self, key: str, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        from engine.dictionary import DEFINITIONS

        d = DEFINITIONS.get(key, {"title": key, "unit": "", "text": "No definition available.", "trust": ""})
        self.setObjectName("Card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(make_label(d["title"], "h1"))
        head.addStretch(1)
        if d["unit"]:
            head.addWidget(Chip(d["unit"], "accent"))
        lay.addLayout(head)
        body = make_label(d["text"], "body", word_wrap=True)
        lay.addWidget(body)
        if d["trust"]:
            plate = QtWidgets.QFrame()
            plate.setObjectName("Plate")
            pl = QtWidgets.QVBoxLayout(plate)
            pl.setContentsMargins(12, 10, 12, 10)
            pl.addWidget(make_label("WHEN TO TRUST IT", "caption"))
            pl.addWidget(make_label(d["trust"], "muted", word_wrap=True))
            lay.addWidget(plate)
        self.setFixedWidth(_theme().px(380))


class HelpButton(QtWidgets.QToolButton):
    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self._key = key
        set_prop(self, "variant", "ghost")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("What does this measure? (click)")
        self.setFixedSize(_theme().px(24), _theme().px(24))
        self.clicked.connect(self._open)
        self.refresh()

    def refresh(self) -> None:
        self.setIcon(icons.icon("help", _theme().t["ink3"], 16, 1.8))

    def _open(self) -> None:
        pop = DefinitionPopup(self._key, self)
        pop.move(self.mapToGlobal(QtCore.QPoint(-pop.width() + self.width(), self.height() + 6)))
        pop.show()


# ------------------------------------------------------------------------------------------------ parameter field
class ParamField(QtWidgets.QWidget):
    """A labelled control that knows its default: shows a ↺ button and an accent dot when modified."""

    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, label: str, control: QtWidgets.QWidget, default, getter: Callable, setter: Callable,
                 signal, help_key: Optional[str] = None, tooltip: str = "", parent=None):
        super().__init__(parent)
        self._get, self._set, self._default = getter, setter, default
        self.control = control
        self.label = make_label(label, "muted")
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(3)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(4)
        top.addWidget(self.label)
        self.dot = QtWidgets.QLabel("●")
        self.dot.setStyleSheet("color: %s; font-size: 9px;" % _theme().t["warn"])
        self.dot.setToolTip("Changed from the default")
        self.dot.hide()
        top.addWidget(self.dot)
        if help_key:
            top.addWidget(HelpButton(help_key))
        top.addStretch(1)
        lay.addLayout(top, 0, 0, 1, 2)
        lay.addWidget(control, 1, 0)
        self.reset_btn = QtWidgets.QToolButton()
        set_prop(self.reset_btn, "variant", "reset")
        self.reset_btn.setIcon(icons.icon("reset", _theme().t["warn"], 16, 2.0))
        self.reset_btn.setToolTip("Reset to default")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.hide()
        self.reset_btn.clicked.connect(self.reset)
        lay.addWidget(self.reset_btn, 1, 1)
        lay.setColumnStretch(0, 1)
        if tooltip:
            control.setToolTip(tooltip)
            self.label.setToolTip(tooltip)
        signal.connect(self._changed)
        self._refresh_modified()

    # factories ---------------------------------------------------------------------------------
    @classmethod
    def spin(cls, label, value, lo, hi, step=1, unit="", default=None, decimals=0, help_key=None, tooltip="", special=None):
        w = QtWidgets.QDoubleSpinBox() if decimals else QtWidgets.QSpinBox()
        if decimals:
            w.setDecimals(decimals)
        w.setRange(lo, hi)
        w.setSingleStep(step)
        if unit:
            w.setSuffix(f" {unit}")
        if special:
            w.setSpecialValueText(special)
        w.setKeyboardTracking(False)
        w.setValue(value)
        w.setAlignment(Qt.AlignmentFlag.AlignLeft)
        return cls(label, w, value if default is None else default, w.value, w.setValue, w.valueChanged, help_key, tooltip)

    @classmethod
    def combo(cls, label, items: Sequence[Tuple[str, object]], current=None, default=None, help_key=None, tooltip=""):
        w = QtWidgets.QComboBox()
        for text, data in items:
            w.addItem(text, data)
        if current is not None:
            idx = w.findData(current)
            if idx >= 0:
                w.setCurrentIndex(idx)
        d = default if default is not None else w.currentData()

        def setter(v):
            i = w.findData(v)
            if i >= 0:
                w.setCurrentIndex(i)
        return cls(label, w, d, w.currentData, setter, w.currentIndexChanged, help_key, tooltip)

    @classmethod
    def check(cls, label, checked=False, default=None, help_key=None, tooltip=""):
        w = QtWidgets.QCheckBox(label)
        w.setChecked(checked)
        f = cls("", w, checked if default is None else default, w.isChecked, w.setChecked, w.toggled, help_key, tooltip)
        f.label.hide()
        return f

    @classmethod
    def text(cls, label, text="", default=None, placeholder="", mono=True, help_key=None, tooltip=""):
        w = QtWidgets.QLineEdit(text)
        w.setPlaceholderText(placeholder)
        if mono:
            w.setProperty("role", "mono")
        return cls(label, w, text if default is None else default, w.text, w.setText, w.textChanged, help_key, tooltip)

    # API ---------------------------------------------------------------------------------------
    def value(self):
        return self._get()

    def set_value(self, v, emit: bool = False) -> None:
        self.control.blockSignals(not emit)
        self._set(v)
        self.control.blockSignals(False)
        self._refresh_modified()

    def set_default(self, v) -> None:
        self._default = v
        self._refresh_modified()

    def default(self):
        return self._default

    def is_modified(self) -> bool:
        a, b = self._get(), self._default
        if isinstance(a, float) or isinstance(b, float):
            try:
                return abs(float(a) - float(b)) > 1e-9
            except (TypeError, ValueError):
                return a != b
        return a != b

    def reset(self) -> None:
        self._set(self._default)

    def _changed(self, *_):
        self._refresh_modified()
        self.valueChanged.emit(self._get())

    def _refresh_modified(self) -> None:
        mod = self.is_modified()
        self.dot.setVisible(mod)
        self.reset_btn.setVisible(mod)


# ------------------------------------------------------------------------------------------------ card
class Card(QtWidgets.QFrame):
    """Titled panel with an optional status chip, help button and a collapsible 'Advanced' region."""

    def __init__(self, title: str, icon_name: Optional[str] = None, subtitle: str = "", help_key: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._icon_name = icon_name
        self._advanced: Optional[QtWidgets.QWidget] = None
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        self.icon_label = QtWidgets.QLabel()
        self.icon_label.setFixedSize(20, 20)
        head.addWidget(self.icon_label)
        self.title_label = make_label(title, "h2")
        head.addWidget(self.title_label)
        if help_key:
            self.help_btn = HelpButton(help_key)
            head.addWidget(self.help_btn)
        head.addStretch(1)
        self.status = Chip("", "neutral")
        self.status.hide()
        head.addWidget(self.status)
        root.addLayout(head)
        self.subtitle = make_label(subtitle, "caption", word_wrap=True)
        self.subtitle.setVisible(bool(subtitle))
        root.addWidget(self.subtitle)
        self.body = QtWidgets.QVBoxLayout()
        self.body.setSpacing(10)
        root.addLayout(self.body)
        self._root = root
        self.refresh_theme()

    def refresh_theme(self) -> None:
        if self._icon_name:
            self.icon_label.setPixmap(icons.pixmap(self._icon_name, _theme().t["solvent"], 20))
        if hasattr(self, "help_btn"):
            self.help_btn.refresh()

    def add(self, widget_or_layout) -> None:
        if isinstance(widget_or_layout, QtWidgets.QLayout):
            self.body.addLayout(widget_or_layout)
        else:
            self.body.addWidget(widget_or_layout)

    def set_status(self, text: str, kind: str = "neutral") -> None:
        if not text:
            self.status.hide()
            return
        self.status.set(text, kind)
        self.status.show()

    def set_active(self, active: bool) -> None:
        set_prop(self, "active", "true" if active else "false")

    # advanced region ---------------------------------------------------------------------------
    def add_advanced(self, widgets: Sequence[QtWidgets.QWidget], label: str = "Advanced settings") -> None:
        self._adv_fields = [w for w in widgets if isinstance(w, ParamField)]
        toggle = QtWidgets.QToolButton()
        toggle.setText(label)
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        set_prop(toggle, "variant", "ghost")
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        # Ignored (not Expanding): the toggle may shrink, so the chip and Reset button never push the card wider than its rail
        toggle.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        toggle.setToolTip("Optional settings: every one can be adjusted, compared with the default and reset")
        self._adv_toggle = toggle
        self._adv_count = Chip("", "warn")
        self._adv_count.hide()
        reset_all = QtWidgets.QToolButton()
        set_prop(reset_all, "variant", "ghost")
        reset_all.setText("Reset")
        reset_all.setToolTip("Reset every advanced setting of this card to its default")
        reset_all.hide()
        self._adv_reset_all = reset_all
        reset_all.clicked.connect(lambda: [f.reset() for f in self._adv_fields])
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(toggle, 1)
        row.addWidget(self._adv_count)
        row.addWidget(reset_all)
        self.add(row)
        box = QtWidgets.QFrame()
        box.setObjectName("Plate")
        bl = QtWidgets.QVBoxLayout(box)
        bl.setContentsMargins(12, 12, 12, 12)
        bl.setSpacing(12)
        for w in widgets:
            bl.addWidget(w)
        box.setVisible(False)
        self._advanced = box
        self.add(box)

        def update_toggle(checked: bool) -> None:
            toggle.setIcon(icons.icon("chevron_down" if checked else "chevron_right", _theme().t["ink2"], 16))
            box.setVisible(checked)

        toggle.toggled.connect(update_toggle)
        update_toggle(False)
        for f in self._adv_fields:
            f.valueChanged.connect(lambda *_: self._refresh_adv_count())
        self._refresh_adv_count()

    def open_advanced(self, open_: bool = True) -> None:
        if hasattr(self, "_adv_toggle"):
            self._adv_toggle.setChecked(open_)

    def _refresh_adv_count(self) -> None:
        n = sum(1 for f in self._adv_fields if f.is_modified())
        self._adv_count.setText(f"{n} changed")
        self._adv_count.setVisible(n > 0)
        self._adv_reset_all.setVisible(n > 0)


# ------------------------------------------------------------------------------------------------ stat tile
class StatTile(QtWidgets.QFrame):
    """One headline number: value ± error, unit, trust badge, definition on demand, count-up motion."""

    def __init__(self, title: str, unit: str = "", help_key: Optional[str] = None, decimals: int = 3, parent=None):
        super().__init__(parent)
        self.setObjectName("Tile")
        self._decimals = decimals
        self._unit = unit
        self._value: Optional[float] = None
        self._anim: Optional[QtCore.QVariantAnimation] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 10, 10)
        lay.setSpacing(1)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(5)
        self.badge = QtWidgets.QLabel()
        self.badge.setFixedSize(16, 16)
        top.addWidget(self.badge)
        self.title_label = make_label(title, "caption")
        top.addWidget(self.title_label)
        top.addStretch(1)
        if help_key:
            top.addWidget(HelpButton(help_key))
        lay.addLayout(top)
        self.value_label = make_label("–", "kpi")
        lay.addWidget(self.value_label)
        self.unit_label = make_label(unit, "unit")
        self.unit_label.setVisible(bool(unit))
        lay.addWidget(self.unit_label)
        self.sub = ElideLabel("", "caption")
        self.note = ElideLabel("", "caption")
        lay.addWidget(self.sub)
        lay.addWidget(self.note)
        self.setFixedHeight(_theme().px(122))

    def _fmt(self, v: float) -> str:
        if v is None or not math.isfinite(v):
            return "n/a"
        if abs(v) >= 1000 or (v != 0 and abs(v) < 0.001):
            return f"{v:.3g}"
        return f"{v:.{self._decimals}g}"

    def set_value(self, value: Optional[float], error: Optional[float] = None, unit: Optional[str] = None,
                  trust: Optional[str] = None, note: str = "", animate: bool = True) -> None:
        if unit is not None:
            self._unit = unit
            self.unit_label.setText(unit)
            self.unit_label.setVisible(bool(unit))
        t = _theme()
        if error is not None and math.isfinite(error):
            self.sub.set_full_text(f"± {self._fmt(error)} (SE)")
        else:
            self.sub.set_full_text("")
        self.note.set_full_text(note)
        set_prop(self, "trust", trust or "ok")
        name = {"warning": "alert", "caution": "alert", "ok": "check", "preview": "eye", None: "check"}.get(trust, "check")
        color = {"warning": t.t["err"], "caution": t.t["warn"], "preview": t.t["ink3"]}.get(trust, t.t["ok"])
        self.badge.setPixmap(icons.pixmap(name, color, 16, 2.2))
        self.badge.setToolTip({"warning": "Do not use without fixing the cause (see Trust).",
                               "caution": "Use with the stated caveat (see Trust).", "preview": "Live preview from sampled frames: run the analysis for the exact value.", None: "No caveats raised.",
                               "ok": "No caveats raised."}.get(trust, ""))
        if value is None or not math.isfinite(value):
            self.value_label.setText("n/a")
            self._value = None
            return
        if self._anim is not None:
            self._anim.stop()
        start = self._value if (self._value is not None and animate and not t.reduce_motion) else None
        self._value = value
        if start is None or start == value:
            self.value_label.setText(self._fmt(value))
            return
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(float(start))
        anim.setEndValue(float(value))
        anim.setDuration(380)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(lambda v: self.value_label.setText(self._fmt(float(v))))
        anim.finished.connect(lambda: self.value_label.setText(self._fmt(value)))
        self._anim = anim
        anim.start()

    def clear(self) -> None:
        self._value = None
        self.value_label.setText("–")
        self.sub.set_full_text("")
        self.note.set_full_text("")
        self.badge.clear()
        set_prop(self, "trust", "ok")


# ------------------------------------------------------------------------------------------------ stepper
class Stepper(QtWidgets.QWidget):
    """Guided flow indicator: Import → Define → Run → Inspect → Export (clickable, animated progress)."""

    stepClicked = QtCore.pyqtSignal(str)

    def __init__(self, steps: Sequence[Tuple[str, str]], parent=None):
        super().__init__(parent)
        self._steps = list(steps)
        self._state = {k: "todo" for k, _ in self._steps}
        self._progress = 0.0
        self._hover = -1
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(450)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.setMouseTracking(True)
        self.setMinimumHeight(_theme().px(46))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _on_anim(self, v) -> None:
        self._progress = float(v)
        self.update()

    def set_state(self, key: str, state: str) -> None:
        self._state[key] = state
        done = 0
        for k, _ in self._steps:
            if self._state[k] == "done":
                done += 1
            else:
                break
        target = float(done)
        if abs(target - self._progress) > 1e-3:
            if _theme().reduce_motion:
                self._progress = target
            else:
                self._anim.stop()
                self._anim.setStartValue(self._progress)
                self._anim.setEndValue(target)
                self._anim.start()
        self.update()

    def state(self, key: str) -> str:
        return self._state[key]

    def _geom(self):
        """Per step: circle centre, label origin and the horizontal extent of the whole (circle + label) group."""
        t = _theme()
        n = len(self._steps)
        r = t.px(11)
        fm = QtGui.QFontMetrics(t.font("ui", 9.5, QtGui.QFont.Weight.DemiBold))
        widths = [2 * r + 8 + fm.horizontalAdvance(lbl) for _k, lbl in self._steps]
        h = self.height() / 2.0
        gap = max((self.width() - 8 - sum(widths)) / max(n - 1, 1), 18.0)
        x = 4.0
        out = []
        for w in widths:
            out.append({"cx": x + r, "cy": h, "lx": x + 2 * r + 8, "x0": x, "x1": x + w})
            x += w + gap
        return out, r, fm

    def paintEvent(self, _):  # noqa: N802
        t = _theme()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        geo, r, fm = self._geom()
        line = QtGui.QColor(t.t["line2"])
        accent = QtGui.QColor(t.t["accent"])
        solvent = QtGui.QColor(t.t["solvent"])
        # connectors run only through the gaps between groups (never through a label), gradient shows progress
        for i in range(len(geo) - 1):
            x0, x1, y = geo[i]["x1"] + 8, geo[i + 1]["x0"] - 8, geo[i]["cy"]
            if x1 <= x0:
                continue
            p.setPen(QtGui.QPen(line, 2.0, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x1, y))
            frac = max(0.0, min(1.0, self._progress - i))
            if frac > 0:
                g = QtGui.QLinearGradient(x0, y, x1, y)
                g.setColorAt(0, solvent)
                g.setColorAt(1, accent)
                p.setPen(QtGui.QPen(QtGui.QBrush(g), 2.8, cap=Qt.PenCapStyle.RoundCap))
                p.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x0 + (x1 - x0) * frac, y))
        fnt = t.font("ui", 9.5, QtGui.QFont.Weight.DemiBold)
        for i, (g, (key, label)) in enumerate(zip(geo, self._steps)):
            cx, cy = g["cx"], g["cy"]
            st = self._state[key]
            hovered = i == self._hover
            if st == "done":
                p.setBrush(solvent)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
                p.drawPixmap(QtCore.QRectF(cx - 7, cy - 7, 14, 14).toRect(), icons.pixmap("check", t.t["bg0"], 14, 3.0))
            elif st == "active":
                halo = QtGui.QColor(accent)
                halo.setAlpha(60)
                p.setBrush(halo)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(QtCore.QPointF(cx, cy), r + 4, r + 4)
                p.setBrush(accent)
                p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
                p.setPen(QtGui.QColor(t.t["accent_ink"]))
                p.setFont(fnt)
                p.drawText(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, str(i + 1))
            else:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QtGui.QPen(QtGui.QColor(t.t["accent"] if hovered else t.t["line2"]), 1.6))
                p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
                p.setPen(QtGui.QColor(t.t["ink3"] if st != "blocked" else t.t["line2"]))
                p.setFont(fnt)
                p.drawText(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, str(i + 1))
            p.setPen(QtGui.QColor(t.t["ink"] if st in ("active", "done") else t.t["ink3"]))
            p.setFont(fnt)
            p.drawText(QtCore.QPointF(g["lx"], cy + fm.ascent() / 2 - 1), label)

    def _index_at(self, pos) -> int:
        geo, _r, _fm = self._geom()
        for i, g in enumerate(geo):
            if g["x0"] - 8 <= pos.x() <= g["x1"] + 8:
                return i
        return -1

    def mouseMoveEvent(self, e):  # noqa: N802
        i = self._index_at(e.position())
        if i != self._hover:
            self._hover = i
            self.update()

    def leaveEvent(self, _):  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, e):  # noqa: N802
        i = self._index_at(e.position())
        if i >= 0:
            self.stepClicked.emit(self._steps[i][0])


# ------------------------------------------------------------------------------------------------ drop zone
class DropZone(QtWidgets.QFrame):
    filesDropped = QtCore.pyqtSignal(list)
    browseClicked = QtCore.pyqtSignal()
    demoClicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 20, 18, 18)
        lay.setSpacing(8)
        self.glyph = QtWidgets.QLabel()
        self.glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.glyph)
        title = make_label("Drop a trajectory here", "h1")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        sub = make_label("Topology (.gro .pdb .psf .tpr …) and trajectory (.xtc .trr .dcd …). "
                         "Drop both at once, or just a single multi-frame file.", "muted", word_wrap=True)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(sub)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.browse = QtWidgets.QPushButton("Choose files…")
        set_prop(self.browse, "variant", "primary")
        self.browse.setToolTip("Select the topology and the trajectory (Ctrl+O)")
        self.browse.clicked.connect(self.browseClicked)
        self.demo = QtWidgets.QPushButton("Try the demo")
        self.demo.setToolTip("Generates a small synthetic solvated system (no files needed) so you can try every step")
        self.demo.clicked.connect(self.demoClicked)
        row.addStretch(1)
        row.addWidget(self.browse)
        row.addWidget(self.demo)
        row.addStretch(1)
        lay.addLayout(row)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        t = _theme().t
        self.glyph.setPixmap(icons.pixmap("drop", t["solvent"], 40, 1.6))

    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            set_prop(self, "hover", "true")

    def dragLeaveEvent(self, e):  # noqa: N802
        set_prop(self, "hover", "false")

    def dropEvent(self, e):  # noqa: N802
        set_prop(self, "hover", "false")
        files = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if files:
            self.filesDropped.emit(files)


# ------------------------------------------------------------------------------------------------ segmented
class Segmented(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal(object)

    def __init__(self, options: Sequence[Tuple[str, object]], current=None, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._buttons: List[Tuple[QtWidgets.QPushButton, object]] = []
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for i, (text, data) in enumerate(options):
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            set_prop(b, "variant", "seg")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            group.addButton(b)
            lay.addWidget(b)
            b.toggled.connect(lambda on, bb=b, d=data: self._on(on, bb, d))
            self._buttons.append((b, data))
        self._current = None
        self.set_value(current if current is not None else options[0][1])

    def _on(self, on, button, data) -> None:
        set_prop(button, "checked", "true" if on else "false")
        if on and data != self._current:
            self._current = data
            self.changed.emit(data)

    def set_value(self, data) -> None:
        for b, d in self._buttons:
            if d == data:
                b.setChecked(True)

    def value(self):
        return self._current


# ------------------------------------------------------------------------------------------------ toasts
class ToastHost(QtCore.QObject):
    """Transient, non-blocking messages anchored to the bottom centre of a parent widget."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self._parent = parent
        self._toasts: List[QtWidgets.QFrame] = []

    def show_message(self, text: str, kind: str = "info", ms: int = 4200) -> None:
        t = _theme()
        frame = QtWidgets.QFrame(self._parent)
        frame.setObjectName("Plate")
        accent = {"ok": t.t["ok"], "warn": t.t["warn"], "err": t.t["err"]}.get(kind, t.t["accent"])
        frame.setStyleSheet(f"QFrame#Plate {{ border: 1px solid {accent}; background: {t.t['bg3']}; border-radius: 12px; }} "
                            f"QLabel {{ color: {t.t['ink']}; background: transparent; }}")
        lay = QtWidgets.QHBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        ic = QtWidgets.QLabel()
        ic.setPixmap(icons.pixmap({"ok": "check", "warn": "alert", "err": "alert"}.get(kind, "info"), accent, 18, 2.2))
        lay.addWidget(ic)
        lab = make_label(text, "body", word_wrap=True)
        lab.setMaximumWidth(t.px(520))
        lay.addWidget(lab)
        frame.adjustSize()
        effect = QtWidgets.QGraphicsOpacityEffect(frame)
        frame.setGraphicsEffect(effect)
        self._toasts.append(frame)
        self._layout()
        frame.show()
        frame.raise_()
        fade_in = QtCore.QPropertyAnimation(effect, b"opacity", frame)
        fade_in.setDuration(0 if t.reduce_motion else 200)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.start()

        def close():
            try:
                fade = QtCore.QPropertyAnimation(effect, b"opacity", frame)
            except RuntimeError:  # the window (and with it this toast) is already gone
                return
            fade.setDuration(0 if t.reduce_motion else 260)
            fade.setStartValue(1.0)
            fade.setEndValue(0.0)

            def done():
                if frame in self._toasts:
                    self._toasts.remove(frame)
                frame.deleteLater()
                self._layout()
            fade.finished.connect(done)
            fade.start()

        timer = QtCore.QTimer(frame)  # owned by the toast: cancelled automatically if the window closes first
        timer.setSingleShot(True)
        timer.timeout.connect(close)
        timer.start(ms)

    def _layout(self) -> None:
        pw, ph = self._parent.width(), self._parent.height()
        y = ph - 24
        for fr in reversed(self._toasts):
            fr.adjustSize()
            y -= fr.height()
            fr.move(int((pw - fr.width()) / 2), y)
            y -= 8
            fr.raise_()

    def reposition(self) -> None:
        self._layout()


class Divider(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
