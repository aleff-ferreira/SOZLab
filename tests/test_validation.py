from types import SimpleNamespace

from engine.models import (
    AnalysisOptions,
    InputConfig,
    OutputConfig,
    ProbeConfig,
    ProjectConfig,
    SOZDefinition,
    SOZNode,
    SelectionSpec,
    SolventConfig,
)
from engine.preflight import PreflightReport
from engine.validation import validate_project
import engine.validation as validation_module


class _FakeTrajectory:
    def __init__(self, n_frames: int) -> None:
        self._n_frames = n_frames
        self.frame = 0

    def __len__(self) -> int:
        return self._n_frames

    def __getitem__(self, frame_index: int):
        self.frame = int(frame_index)
        return self


class _FakeUniverse:
    def __init__(self, n_frames: int) -> None:
        self.trajectory = _FakeTrajectory(n_frames)


def _make_project() -> ProjectConfig:
    return ProjectConfig(
        inputs=InputConfig(topology="dummy", trajectory="dummy.xtc"),
        solvent=SolventConfig(
            water_resnames=["WAT"],
            water_oxygen_names=["O"],
            probe=ProbeConfig(selection="name O", position="atom"),
        ),
        selections={
            "selection_a": SelectionSpec(
                label="selection_a",
                selection="name CA",
                require_unique=True,
            )
        },
        sozs=[
            SOZDefinition(
                name="SOZ_1",
                description="validation window regression",
                root=SOZNode(
                    type="distance",
                    params={"selection_label": "selection_a", "cutoff": 3.5, "unit": "A"},
                ),
            )
        ],
        analysis=AnalysisOptions(frame_start=10, frame_stop=30, stride=5),
        outputs=OutputConfig(output_dir="results"),
    )


def test_validate_project_respects_analysis_window(monkeypatch) -> None:
    visited_frames: list[tuple[str, int]] = []
    universe = _FakeUniverse(n_frames=50)
    project = _make_project()

    monkeypatch.setattr(validation_module.mda, "Universe", lambda *args, **kwargs: universe)
    monkeypatch.setattr(
        validation_module,
        "run_preflight",
        lambda *_args, **_kwargs: PreflightReport(
            ok=True,
            errors=[],
            warnings=[],
            selection_checks={},
            solvent_summary={},
            trajectory_summary={},
            pbc_summary={},
            gmx_summary={},
        ),
    )
    monkeypatch.setattr(validation_module, "build_solvent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        validation_module,
        "resolve_selection",
        lambda *_args, **_kwargs: SimpleNamespace(group=object()),
    )

    def _record(kind: str):
        def _inner(node, context):
            visited_frames.append((kind, int(context.universe.trajectory.frame)))
            return set()

        return _inner

    monkeypatch.setattr(validation_module, "evaluate_node_fast", _record("fast"))
    monkeypatch.setattr(validation_module, "evaluate_node_slow", _record("slow"))

    results = validate_project(project, max_frames=4)

    assert [frame for kind, frame in visited_frames if kind == "fast"] == [10, 15, 20, 25]
    assert [frame for kind, frame in visited_frames if kind == "slow"] == [10, 15, 20, 25]
    assert results[0].total_frames == 4
