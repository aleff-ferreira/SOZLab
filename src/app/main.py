"""SOZLab Workbench entry point (``sozlab-gui``).

    sozlab-gui                       open the Workbench
    sozlab-gui system.gro run.xtc    open a topology and trajectory directly
    sozlab-gui --demo                load the built-in synthetic demo system
    sozlab-gui --theme light         start in the light theme

The released v2.0 GUI (a single 11 000-line module) was replaced by the Workbench in v2.1; its code remains in the
git history (tag ``v2.0.0``). Project files written by it still open: use *File → Open a project file*.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sozlab-gui", description="SOZLab Workbench: solvent occupancy zone analysis")
    p.add_argument("files", nargs="*", help="a topology and (optionally) a trajectory or a SOZLab project (.json)")
    p.add_argument("--demo", action="store_true", help="load the synthetic demo system (no files needed)")
    p.add_argument("--theme", choices=("dark", "light"), help="colour theme (remembered between sessions)")
    p.add_argument("--reduce-motion", action="store_true", help="turn off animations")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    from engine._version import __version__

    if args.version:
        print(f"SOZLab {__version__}")
        return 0
    from PyQt6 import QtCore, QtWidgets

    from app.workbench.theme import Theme
    from app.workbench.window import WorkbenchWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("SOZLab")
    app.setOrganizationName("SOZLab")
    app.setApplicationVersion(__version__)
    settings = QtCore.QSettings("SOZLab", "Workbench")
    theme = Theme.instance()
    theme.mode = args.theme or str(settings.value("theme", "dark"))
    theme.reduce_motion = bool(args.reduce_motion or str(settings.value("reduce_motion", "false")).lower() == "true")
    theme.apply(app)
    win = WorkbenchWindow()
    win.btn_motion.setChecked(theme.reduce_motion)
    geom = settings.value("geometry")
    if geom is not None:
        win.restoreGeometry(geom)
    win.show()

    def remember() -> None:
        settings.setValue("theme", theme.mode)
        settings.setValue("reduce_motion", "true" if theme.reduce_motion else "false")
        settings.setValue("geometry", win.saveGeometry())

    app.aboutToQuit.connect(remember)
    if args.files:
        QtCore.QTimer.singleShot(0, lambda: win.session.open_paths(args.files) if not args.files[0].lower().endswith(".json")
                                 else win._open_project_path(args.files[0]))
    elif args.demo:
        QtCore.QTimer.singleShot(0, win.load_demo)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
