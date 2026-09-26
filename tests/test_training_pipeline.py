import sys
import types

# CI installs only core dependencies; training_pipeline imports YOLO at module level.
sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=object))

from training_pipeline import WarmStartTrainer, _is_cell_box  # noqa: E402


def test_full_frame_boxes_are_not_cell_boxes():
    assert _is_cell_box(["1", "0.5", "0.5", "0.3", "0.4"])
    assert not _is_cell_box(["1", "0.5", "0.5", "1.0", "0.95"])
    assert not _is_cell_box(["1", "0.5", "0.5"])


def test_remap_drops_full_frame_boxes(tmp_path):
    src = tmp_path / "src.txt"
    dest = tmp_path / "dest.txt"
    src.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "1 0.5 0.5 1.0 1.0\n"
        "2 0.4 0.4 0.3 0.3\n",
        encoding="utf-8",
    )
    WarmStartTrainer()._remap_and_copy_label(src, dest, {0: 7, 1: 5, 2: 7})
    assert dest.read_text(encoding="utf-8").splitlines() == [
        "7 0.5 0.5 0.2 0.2",
        "7 0.4 0.4 0.3 0.3",
    ]
