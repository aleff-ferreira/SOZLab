"""GUI test fixtures: offscreen Qt, a small synthetic hydrated solute, and a ready Workbench window."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session")
def demo_files(tmp_path_factory):
    from engine.synthetic import make_demo_system

    d = tmp_path_factory.mktemp("gui_demo")
    make_demo_system(n_waters=420, n_frames=70, box=26.0, seed=11).write(str(d), "demo", "xtc")
    return str(d / "demo.gro"), str(d / "demo.xtc")


@pytest.fixture
def window(qtbot, tmp_path):
    from PyQt6 import QtWidgets

    from app.workbench.theme import Theme
    from app.workbench.window import WorkbenchWindow

    th = Theme.instance()
    th.mode, th.reduce_motion = "dark", True
    th.apply(QtWidgets.QApplication.instance())
    win = WorkbenchWindow(demo_dir=str(tmp_path / "demo_cache"))
    qtbot.addWidget(win)
    win.resize(1366, 768)
    win.show()
    yield win
    win.close()


def load(qtbot, win, files):
    with qtbot.waitSignal(win.session.summaryReady, timeout=30000):
        win.session.open_inputs(files[0], files[1])
    qtbot.waitUntil(lambda: win.session.selection_ok is not None, timeout=10000)
    return win


def run(qtbot, win, timeout=90000):
    with qtbot.waitSignal(win.session.resultReady, timeout=timeout):
        win.session.start_run()
    return win.session.result


@pytest.fixture
def loaded(qtbot, window, demo_files):
    return load(qtbot, window, demo_files)
