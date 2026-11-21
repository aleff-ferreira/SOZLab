"""SOZLab Workbench design system: colour tokens, typography, spacing, QSS.

Identity: a lab bench for water. Deep-sea instrument panel (dark, default) or warm lab-notebook paper (light);
two signature hues carry the science - **aquamarine = solvent / occupied** and **coral = solute / zone anchor** -
on a cool neutral scale, with one periwinkle accent for interaction. Every colour in the app comes from these
tokens, so charts, the 3D scene and widgets always agree.
"""
from __future__ import annotations

import os
from typing import Dict

from PyQt6 import QtCore, QtGui, QtWidgets

DARK: Dict[str, str] = {
    "bg0": "#080F1D", "bg1": "#0D1729", "bg2": "#132038", "bg3": "#1A2A47", "line": "#233556", "line2": "#2E4470",
    "ink": "#E9F1FB", "ink2": "#A8B9D1", "ink3": "#7489A5",
    "solvent": "#35E0C8", "solvent_dim": "#1B7F76", "solute": "#FF8A5B", "solute_dim": "#9B4F31",
    "accent": "#7AA7FF", "accent_hi": "#A9C4FF", "accent_ink": "#07122B",
    "ok": "#4ADE80", "warn": "#FBBF24", "err": "#FB7185", "shadow": "#000000",
}
LIGHT: Dict[str, str] = {
    "bg0": "#F3F0E8", "bg1": "#FAF8F3", "bg2": "#FFFFFF", "bg3": "#EEE9DD", "line": "#DDD5C4", "line2": "#C9BFA9",
    "ink": "#13233B", "ink2": "#44566F", "ink3": "#626E81",
    "solvent": "#0B7A6D", "solvent_dim": "#7FCBC2", "solute": "#BB471A", "solute_dim": "#EBA98D",
    "accent": "#2B62EB", "accent_hi": "#1C4FD6", "accent_ink": "#FFFFFF",
    "ok": "#157D3C", "warn": "#B05109", "err": "#BE123C", "shadow": "#1B2A44",
}

UI_STACK = ["Segoe UI Variable Text", "Inter", "SF Pro Text", "Helvetica Neue", "Noto Sans", "DejaVu Sans"]
DISPLAY_STACK = ["Bahnschrift", "Segoe UI Variable Display", "Inter", "SF Pro Display", "Noto Sans", "DejaVu Sans"]
MONO_STACK = ["Cascadia Mono", "JetBrains Mono", "SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono"]


def _first_available(stack, families) -> str:
    for name in stack:
        if name in families:
            return name
    return stack[-1]


class Theme(QtCore.QObject):
    """Application-wide theme singleton (``Theme.instance()``); emits ``changed`` after every switch."""

    changed = QtCore.pyqtSignal()
    _inst = None

    @classmethod
    def instance(cls) -> "Theme":
        if cls._inst is None:
            cls._inst = Theme()
        return cls._inst

    def __init__(self) -> None:
        super().__init__()
        self.mode = "dark"
        self.reduce_motion = False
        self.scale = 1.0
        self.ui_family = UI_STACK[-1]
        self.display_family = DISPLAY_STACK[-1]
        self.mono_family = MONO_STACK[-1]
        self._fonts_ready = False

    # -------------------------------------------------------------------------- tokens
    @property
    def t(self) -> Dict[str, str]:
        return DARK if self.mode == "dark" else LIGHT

    def color(self, key: str) -> QtGui.QColor:
        return QtGui.QColor(self.t[key])

    def rgba(self, key: str, alpha: float) -> str:
        c = QtGui.QColor(self.t[key])
        return f"rgba({c.red()},{c.green()},{c.blue()},{alpha:.3f})"

    def px(self, value: float) -> int:
        return max(1, int(round(value * self.scale)))

    @property
    def is_dark(self) -> bool:
        return self.mode == "dark"

    # -------------------------------------------------------------------------- fonts
    def register_fonts(self) -> None:
        if self._fonts_ready:
            return
        try:  # DejaVu ships with matplotlib (a hard dependency): guarantees a good fallback everywhere
            import matplotlib

            base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
            for fname in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"):
                QtGui.QFontDatabase.addApplicationFont(os.path.join(base, fname))
        except Exception:
            pass
        fams = set(QtGui.QFontDatabase.families())
        self.ui_family = _first_available(UI_STACK, fams)
        self.display_family = _first_available(DISPLAY_STACK, fams)
        self.mono_family = _first_available(MONO_STACK, fams)
        self._fonts_ready = True

    def font(self, role: str = "ui", size: float = 10.0, weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal) -> QtGui.QFont:
        fam = {"ui": self.ui_family, "display": self.display_family, "mono": self.mono_family}[role]
        f = QtGui.QFont(fam)
        f.setPointSizeF(size * self.scale)
        f.setWeight(weight)
        f.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
        if role == "mono":
            f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        return f

    # -------------------------------------------------------------------------- apply
    def apply(self, app: QtWidgets.QApplication) -> None:
        self.register_fonts()
        app.setStyle("Fusion")
        app.setFont(self.font("ui", 9.5))
        pal = QtGui.QPalette()
        t = self.t
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(t["bg0"]))
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(t["ink"]))
        pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(t["bg2"]))
        pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(t["bg1"]))
        pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(t["ink"]))
        pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(t["bg2"]))
        pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(t["ink"]))
        pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(t["accent"]))
        pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(t["accent_ink"]))
        pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(t["bg3"]))
        pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(t["ink"]))
        pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor(t["ink3"]))
        app.setPalette(pal)
        app.setStyleSheet(self.qss())
        self.changed.emit()

    def set_mode(self, mode: str, app: QtWidgets.QApplication | None = None) -> None:
        self.mode = "light" if mode == "light" else "dark"
        app = app or QtWidgets.QApplication.instance()
        if app is not None:
            self.apply(app)

    # -------------------------------------------------------------------------- QSS
    def qss(self) -> str:
        t = self.t
        p = self.px
        ui, disp, mono = self.ui_family, self.display_family, self.mono_family
        glow = self.rgba("accent", 0.35)
        return f"""
* {{ outline: none; }}
QWidget {{ color: {t['ink']}; font-family: "{ui}"; }}
QMainWindow, QDialog {{ background: {t['bg0']}; }}
QWidget#Root {{ background: {t['bg0']}; }}
QWidget#Transparent, QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollArea {{ border: none; }}
QToolTip {{ background: {t['bg3']}; color: {t['ink']}; border: 1px solid {t['line2']}; padding: {p(6)}px {p(8)}px;
           border-radius: {p(6)}px; font-size: {p(12)}px; }}

/* ---- typography roles ---- */
QLabel {{ background: transparent; }}
QLabel[role="brand"] {{ font-family: "{disp}"; font-weight: 700; font-size: {p(19)}px; letter-spacing: 0.5px; }}
QLabel[role="h1"] {{ font-family: "{disp}"; font-weight: 700; font-size: {p(17)}px; }}
QLabel[role="h2"] {{ font-family: "{disp}"; font-weight: 600; font-size: {p(13)}px; letter-spacing: 0.6px;
                    text-transform: uppercase; color: {t['ink2']}; }}
QLabel[role="caption"] {{ color: {t['ink3']}; font-size: {p(11)}px; }}
QLabel[role="muted"] {{ color: {t['ink2']}; font-size: {p(12)}px; }}
QLabel[role="body"] {{ font-size: {p(12.5)}px; }}
QLabel[role="mono"] {{ font-family: "{mono}"; font-size: {p(12)}px; }}
QLabel[role="kpi"] {{ font-family: "{mono}"; font-weight: 700; font-size: {p(24)}px; }}
QLabel[role="unit"] {{ font-family: "{mono}"; color: {t['ink3']}; font-size: {p(12)}px; }}
QLabel[role="error"] {{ color: {t['err']}; font-size: {p(12)}px; }}

/* ---- cards ---- */
QFrame#Card {{ background: {t['bg1']}; border: 1px solid {t['line']}; border-radius: {p(14)}px; }}
QFrame#Card[active="true"] {{ border: 1px solid {t['accent']}; }}
QFrame#CardHeader {{ background: transparent; border: none; }}
QFrame#Divider {{ background: {t['line']}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#Plate {{ background: {t['bg2']}; border: 1px solid {t['line']}; border-radius: {p(10)}px; }}
QFrame#Tile {{ background: {t['bg2']}; border: 1px solid {t['line']}; border-radius: {p(12)}px; }}
QFrame#Tile[trust="warning"] {{ border: 1px solid {t['err']}; }}
QFrame#Tile[trust="caution"] {{ border: 1px solid {t['warn']}; }}
QFrame#Tile[trust="preview"] {{ border: 1px dashed {t['line2']}; }}
QFrame#DropZone {{ background: {self.rgba('accent', 0.06)}; border: 1.5px dashed {t['line2']}; border-radius: {p(14)}px; }}
QFrame#DropZone[hover="true"] {{ border: 1.5px dashed {t['accent']}; background: {self.rgba('accent', 0.14)}; }}
QFrame#TopBar {{ background: {t['bg1']}; border-bottom: 1px solid {t['line']}; }}
QFrame#StatusBar {{ background: {t['bg1']}; border-top: 1px solid {t['line']}; }}

/* ---- chips ---- */
QLabel#Chip {{ border-radius: {p(9)}px; padding: {p(2)}px {p(9)}px; font-size: {p(11)}px; font-weight: 600;
              background: {t['bg3']}; color: {t['ink2']}; border: 1px solid {t['line']}; }}
QLabel#Chip[kind="ok"] {{ background: {self.rgba('ok', 0.14)}; color: {t['ok']}; border: 1px solid {self.rgba('ok', 0.45)}; }}
QLabel#Chip[kind="warn"] {{ background: {self.rgba('warn', 0.14)}; color: {t['warn']}; border: 1px solid {self.rgba('warn', 0.45)}; }}
QLabel#Chip[kind="err"] {{ background: {self.rgba('err', 0.14)}; color: {t['err']}; border: 1px solid {self.rgba('err', 0.45)}; }}
QLabel#Chip[kind="solvent"] {{ background: {self.rgba('solvent', 0.14)}; color: {t['solvent']}; border: 1px solid {self.rgba('solvent', 0.45)}; }}
QLabel#Chip[kind="solute"] {{ background: {self.rgba('solute', 0.14)}; color: {t['solute']}; border: 1px solid {self.rgba('solute', 0.45)}; }}
QLabel#Chip[kind="accent"] {{ background: {self.rgba('accent', 0.14)}; color: {t['accent']}; border: 1px solid {self.rgba('accent', 0.45)}; }}

/* ---- buttons ---- */
QPushButton, QToolButton {{ background: {t['bg2']}; color: {t['ink']}; border: 1px solid {t['line2']}; border-radius: {p(9)}px;
                           padding: {p(7)}px {p(14)}px; font-size: {p(12.5)}px; font-weight: 600; }}
QPushButton:hover, QToolButton:hover {{ background: {t['bg3']}; border-color: {t['accent']}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {t['bg1']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {t['ink3']}; background: {t['bg1']}; border-color: {t['line']}; }}
QPushButton:focus, QToolButton:focus {{ border: 1px solid {t['accent']}; }}
QPushButton[variant="primary"] {{ background: {t['accent']}; color: {t['accent_ink']}; border: 1px solid {t['accent']};
                                 font-size: {p(14)}px; padding: {p(10)}px {p(18)}px; }}
QPushButton[variant="primary"]:hover {{ background: {t['accent_hi']}; border-color: {t['accent_hi']}; }}
QPushButton[variant="primary"]:disabled {{ background: {t['bg3']}; color: {t['ink3']}; border-color: {t['line']}; }}
QPushButton[variant="go"] {{ background: {t['solvent']}; color: {t['accent_ink'] if self.is_dark else '#FFFFFF'}; border: 1px solid {t['solvent']};
                            font-size: {p(14)}px; padding: {p(10)}px {p(18)}px; }}
QPushButton[variant="go"]:hover {{ background: {t['solvent_dim'] if not self.is_dark else '#6DF0DC'}; }}
QPushButton[variant="go"]:disabled {{ background: {t['bg3']}; color: {t['ink3']}; border-color: {t['line']}; }}
QPushButton[variant="danger"] {{ color: {t['err']}; border-color: {self.rgba('err', 0.5)}; }}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {{ background: transparent; border: 1px solid transparent; color: {t['ink2']}; padding: {p(5)}px {p(8)}px; }}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {{ background: {t['bg3']}; color: {t['ink']}; border-color: {t['line']}; }}
QToolButton[variant="reset"] {{ background: transparent; border: none; color: {t['warn']}; padding: {p(2)}px; }}
QPushButton[variant="seg"] {{ border-radius: 0; border-right-width: 0; padding: {p(6)}px {p(12)}px; font-weight: 600; }}
QPushButton[variant="seg"][checked="true"] {{ background: {self.rgba('accent', 0.22)}; border-color: {t['accent']}; color: {t['accent_hi'] if self.is_dark else t['accent']}; }}

/* ---- inputs ---- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: {p(8)}px; padding: {p(6)}px {p(9)}px;
    selection-background-color: {t['accent']}; selection-color: {t['accent_ink']}; font-size: {p(12.5)}px; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {t['accent']}; }}
QLineEdit[state="err"], QSpinBox[state="err"], QDoubleSpinBox[state="err"] {{ border: 1px solid {t['err']}; }}
QLineEdit[state="ok"] {{ border: 1px solid {self.rgba('ok', 0.6)}; }}
QLineEdit[role="mono"], QPlainTextEdit[role="mono"] {{ font-family: "{mono}"; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: {p(16)}px; border: none; background: transparent; }}
QComboBox::drop-down {{ border: none; width: {p(22)}px; }}
QComboBox QAbstractItemView {{ background: {t['bg2']}; border: 1px solid {t['line2']}; selection-background-color: {t['accent']};
                              selection-color: {t['accent_ink']}; padding: {p(4)}px; outline: none; }}
QCheckBox, QRadioButton {{ spacing: {p(8)}px; font-size: {p(12.5)}px; }}
QCheckBox::indicator {{ width: {p(17)}px; height: {p(17)}px; border: 1px solid {t['line2']}; border-radius: {p(5)}px; background: {t['bg2']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}
QCheckBox::indicator:hover {{ border-color: {t['accent']}; }}
QRadioButton::indicator {{ width: {p(16)}px; height: {p(16)}px; border: 1px solid {t['line2']}; border-radius: {p(9)}px; background: {t['bg2']}; }}
QRadioButton::indicator:checked {{ background: {t['accent']}; border: 4px solid {t['bg2']}; outline: 1px solid {t['accent']}; }}

QSlider::groove:horizontal {{ height: {p(5)}px; background: {t['bg3']}; border-radius: {p(3)}px; }}
QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: {p(3)}px; }}
QSlider::handle:horizontal {{ background: {t['ink']}; border: 2px solid {t['accent']}; width: {p(14)}px; height: {p(14)}px;
                             margin: -{p(6)}px 0; border-radius: {p(8)}px; }}
QSlider::handle:horizontal:hover {{ background: {t['accent_hi']}; }}

QProgressBar {{ background: {t['bg3']}; border: none; border-radius: {p(5)}px; text-align: center; color: {t['ink']};
               font-size: {p(11)}px; min-height: {p(10)}px; max-height: {p(10)}px; }}
QProgressBar::chunk {{ border-radius: {p(5)}px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {t['accent']}, stop:1 {t['solvent']}); }}

QScrollBar:vertical {{ background: transparent; width: {p(10)}px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['line2']}; border-radius: {p(4)}px; min-height: {p(30)}px; }}
QScrollBar::handle:vertical:hover {{ background: {t['ink3']}; }}
QScrollBar:horizontal {{ background: transparent; height: {p(10)}px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['line2']}; border-radius: {p(4)}px; min-width: {p(30)}px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTabWidget::pane {{ border: none; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {t['ink2']}; padding: {p(8)}px {p(14)}px; margin-right: {p(2)}px; border: none;
               border-bottom: 2px solid transparent; font-weight: 600; font-size: {p(12.5)}px; }}
QTabBar::tab:hover {{ color: {t['ink']}; }}
QTabBar::tab:selected {{ color: {t['ink']}; border-bottom: 2px solid {t['solvent']}; }}

QTableView, QTableWidget, QTreeView {{ background: {t['bg1']}; alternate-background-color: {t['bg2']}; border: 1px solid {t['line']};
    border-radius: {p(8)}px; gridline-color: transparent; selection-background-color: {self.rgba('accent', 0.28)};
    selection-color: {t['ink']}; font-size: {p(12)}px; }}
QHeaderView::section {{ background: {t['bg2']}; color: {t['ink2']}; border: none; border-bottom: 1px solid {t['line']};
                       padding: {p(6)}px {p(8)}px; font-weight: 600; font-size: {p(11)}px; }}
QMenu {{ background: {t['bg2']}; border: 1px solid {t['line2']}; padding: {p(6)}px; border-radius: {p(8)}px; }}
QMenu::item {{ padding: {p(7)}px {p(18)}px; border-radius: {p(6)}px; }}
QMenu::item:selected {{ background: {t['accent']}; color: {t['accent_ink']}; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: {p(7)}px; }}
QSplitter::handle:vertical {{ height: {p(7)}px; }}
QSplitter::handle:hover {{ background: {self.rgba('accent', 0.35)}; }}
QGroupBox {{ border: 1px solid {t['line']}; border-radius: {p(10)}px; margin-top: {p(14)}px; padding-top: {p(10)}px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: {p(12)}px; color: {t['ink2']}; font-weight: 600; }}
QPushButton[variant="chip"] {{ background: {t['bg2']}; border: 1px solid {t['line2']}; border-radius: {p(12)}px; padding: {p(4)}px {p(11)}px;
                              font-size: {p(11.5)}px; font-weight: 600; color: {t['ink2']}; }}
QPushButton[variant="chip"]:hover {{ border-color: {t['accent']}; color: {t['ink']}; }}
QPushButton[variant="chip"][checked="true"] {{ background: {self.rgba('solute', 0.18)}; border-color: {t['solute']}; color: {t['solute']}; }}
QFrame#Plate[selected="true"] {{ border: 1px solid {t['accent']}; }}
QFrame#RunBar {{ background: {t['bg1']}; border-top: 1px solid {t['line']}; }}
QFrame#Stage {{ background: {t['bg1']}; border: 1px solid {t['line']}; border-radius: {p(14)}px; }}
QToolButton[variant="bar"] {{ background: transparent; border: 1px solid transparent; border-radius: {p(9)}px; padding: {p(6)}px; }}
QToolButton[variant="bar"]:hover {{ background: {t['bg3']}; border-color: {t['line']}; }}
QToolButton[variant="bar"]:checked {{ background: {self.rgba('accent', 0.18)}; border-color: {t['accent']}; }}
QFrame#Issue {{ background: {t['bg2']}; border: 1px solid {t['line']}; border-radius: {p(10)}px; }}
QFrame#Issue[level="error"] {{ border: 1px solid {self.rgba('err', 0.6)}; background: {self.rgba('err', 0.08)}; }}
QFrame#Issue[level="warn"] {{ border: 1px solid {self.rgba('warn', 0.55)}; background: {self.rgba('warn', 0.07)}; }}
"""



def polish(widget: QtWidgets.QWidget) -> None:
    """Re-evaluate dynamic-property styles after a property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_prop(widget: QtWidgets.QWidget, name: str, value) -> None:
    widget.setProperty(name, value)
    polish(widget)
