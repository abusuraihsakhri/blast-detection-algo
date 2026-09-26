import sys
import types
import zlib

import pytest

# CI installs only core dependencies; training_pipeline imports YOLO at module level.
sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=object))

from training_pipeline import (  # noqa: E402
    WarmStartTrainer,
    _is_cell_box,
    _source_image_id,
)


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
        "7 0.500000 0.500000 0.200000 0.200000",
        "7 0.400000 0.400000 0.300000 0.300000",
    ]


def test_polygon_rows_become_boxes_in_mixed_files(tmp_path):
    src = tmp_path / "src.txt"
    dest = tmp_path / "dest.txt"
    # One plain box row and one 4-corner polygon, as in Leukemia-NFXZN.
    src.write_text(
        "2 0.5 0.5 0.1 0.2\n"
        "1 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n",
        encoding="utf-8",
    )
    WarmStartTrainer()._remap_and_copy_label(src, dest, {1: 1, 2: 2})
    assert dest.read_text(encoding="utf-8").splitlines() == [
        "2 0.500000 0.500000 0.100000 0.200000",
        "1 0.200000 0.400000 0.200000 0.400000",
    ]


def test_large_polygon_is_not_mistaken_for_full_frame():
    from training_pipeline import _to_box

    # Read naively, fields 3-4 (0.9, 0.95) look like a full-frame box.
    box = _to_box(["0", "0.8", "0.85", "0.9", "0.85", "0.9", "0.95", "0.8", "0.95"])
    assert box == ["0", "0.850000", "0.900000", "0.100000", "0.100000"]
    assert _is_cell_box(box)


def _make_dataset(root, split, stems):
    (root / split / "images").mkdir(parents=True)
    (root / split / "labels").mkdir(parents=True)
    for stem in stems:
        path = root / split / "images" / f"{stem}.jpg"
        _noise_image(path, seed=zlib.crc32(str(path).encode()))
        (root / split / "labels" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )


def test_augmented_copies_share_a_source_id():
    ids = {
        _source_image_id(f"ALL1_100{suffix}_jpg.rf.{i}abc.jpg")
        for i, suffix in enumerate(["", "_fH", "_fV", "_r090", "_r180", "_r270"])
    }
    assert ids == {"ALL1_100"}
    assert _source_image_id("ALL1_101_jpg.rf.x.jpg") == "ALL1_101"


def test_train_only_dataset_lends_holdout_to_validation(tmp_path):
    trainer = WarmStartTrainer()
    trainer.datasets_dir = tmp_path / "sources"
    trainer.unified_train_dir = tmp_path / "unified"
    trainer.remapping_rules = {"only-train": {0: 8}, "has-valid": {0: 0}}
    stems = [
        f"IMG{i:03d}{suffix}_jpg.rf.h{i}{suffix}"
        for i in range(100)
        for suffix in ("", "_fH", "_r090")
    ]
    _make_dataset(trainer.datasets_dir / "only-train", "train", stems)
    _make_dataset(trainer.datasets_dir / "has-valid", "train", ["a", "b"])
    _make_dataset(trainer.datasets_dir / "has-valid", "valid", ["c"])

    trainer.prepare_unified_dataset()

    def source_ids(split):
        return {
            _source_image_id(p.name.removeprefix("only-train_"))
            for p in (trainer.unified_train_dir / split / "images").iterdir()
            if p.name.startswith("only-train_")
        }

    valid, train = source_ids("valid"), source_ids("train")
    assert valid and train
    assert not valid & train
    assert len(valid | train) == 100
    assert len(list((trainer.unified_train_dir / "train" / "images").glob("has-valid_*"))) == 2
    assert len(list((trainer.unified_train_dir / "valid" / "images").glob("has-valid_*"))) == 1


def test_resume_does_not_rebuild_the_merged_dataset(tmp_path, monkeypatch):
    trainer = WarmStartTrainer()
    trainer.unified_train_dir = tmp_path / "unified"

    def fail_rebuild(**_):
        raise AssertionError("resume must not rebuild the dataset")

    monkeypatch.setattr(trainer, "prepare_unified_dataset", fail_rebuild)
    with pytest.raises(FileNotFoundError, match="Cannot resume"):
        trainer.train(resume=True)


def _noise_image(path, seed):
    import random

    from PIL import Image

    rng = random.Random(seed)
    image = Image.new("L", (64, 64))
    image.putdata([rng.randrange(256) for _ in range(64 * 64)])
    image.convert("RGB").save(path)
    return image


def test_train_copies_of_eval_images_are_removed(tmp_path):
    from PIL import Image

    trainer = WarmStartTrainer()
    trainer.datasets_dir = tmp_path / "sources"
    trainer.unified_train_dir = tmp_path / "unified"
    trainer.remapping_rules = {"a": {0: 0}, "b": {0: 0}}
    for ds in ("a", "b"):
        for split in ("train", "valid", "test"):
            (trainer.datasets_dir / ds / split / "images").mkdir(parents=True)
            (trainer.datasets_dir / ds / split / "labels").mkdir(parents=True)

    def add(ds, split, name, seed, transform=None):
        path = trainer.datasets_dir / ds / split / "images" / f"{name}.jpg"
        image = _noise_image(path, seed)
        if transform is not None:
            image.transpose(transform).convert("RGB").save(path, quality=95)
        (path.parent.parent / "labels" / f"{name}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )

    add("a", "valid", "shared", seed=1)
    add("b", "train", "rotated_copy", seed=1, transform=Image.Transpose.ROTATE_90)
    add("a", "test", "held_out", seed=2)
    add("b", "train", "copy_of_test", seed=2)
    add("b", "train", "unrelated", seed=3)

    trainer.prepare_unified_dataset()

    kept = sorted(
        p.stem for p in (trainer.unified_train_dir / "train" / "images").iterdir()
    )
    labels = sorted(
        p.stem for p in (trainer.unified_train_dir / "train" / "labels").iterdir()
    )
    assert kept == labels == ["b_unrelated"]


def test_images_whose_only_labels_are_artifacts_are_skipped(tmp_path):
    trainer = WarmStartTrainer()
    trainer.datasets_dir = tmp_path / "sources"
    trainer.unified_train_dir = tmp_path / "unified"
    trainer.remapping_rules = {"ds": {0: 4}}
    _make_dataset(trainer.datasets_dir / "ds", "train", ["tag_only", "cell", "empty"])
    _make_dataset(trainer.datasets_dir / "ds", "valid", ["v"])
    labels = trainer.datasets_dir / "ds" / "train" / "labels"
    (labels / "tag_only.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    (labels / "empty.txt").write_text("", encoding="utf-8")

    trainer.prepare_unified_dataset()

    images = sorted(
        p.stem for p in (trainer.unified_train_dir / "train" / "images").iterdir()
    )
    assert images == ["ds_cell", "ds_empty"]
