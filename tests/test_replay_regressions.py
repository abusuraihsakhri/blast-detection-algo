from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import replay_buffer as replay_module
from models import Base, Tile
from replay_buffer import ExperienceReplayBuffer, ReplayBufferError


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


def _tile(db, path: str) -> Tile:
    tile = Tile(
        file_path=path,
        source_wsi="test.svs",
        x_coord=0,
        y_coord=0,
        level=0,
        tissue_pct=1.0,
        is_annotated=True,
    )
    db.add(tile)
    db.flush()
    return tile


def test_new_tiles_are_never_silently_dropped_by_dataset_cap(
    session,
    tmp_path,
):
    for index in range(2):
        image = tmp_path / f"tile_{index}.jpg"
        image.write_bytes(b"x")
        _tile(session, str(image))

    buffer = ExperienceReplayBuffer(
        session,
        max_dataset_size=1,
        seed=1,
    )
    with pytest.raises(
        ReplayBufferError,
        match="exceed max dataset size",
    ):
        buffer.assemble_dataset()


def test_legacy_raw_filename_resolves_to_raw_tiles(
    session,
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw_tiles"
    raw_dir.mkdir(parents=True)
    image = raw_dir / "legacy.jpg"
    image.write_bytes(b"x")

    monkeypatch.setattr(
        replay_module,
        "DATA_DIR",
        data_dir,
    )
    monkeypatch.setattr(
        replay_module,
        "RAW_TILES_DIR",
        raw_dir,
    )
    tile = _tile(
        session,
        "legacy.jpg",
    )

    buffer = ExperienceReplayBuffer(
        session,
        seed=1,
    )
    assert (
        buffer._resolve_image_path(tile)
        == image.resolve()
    )


def test_negative_tiles_get_explicit_empty_label_file(
    session,
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    temp_dir = data_dir / "temp_train"
    image = data_dir / "negative.jpg"
    data_dir.mkdir(parents=True)
    image.write_bytes(b"x")

    monkeypatch.setattr(
        replay_module,
        "DATA_DIR",
        data_dir,
    )
    monkeypatch.setattr(
        replay_module,
        "TEMP_TRAIN_DIR",
        temp_dir,
    )
    tile = _tile(
        session,
        str(image),
    )

    buffer = ExperienceReplayBuffer(
        session,
        seed=1,
    )
    buffer._prepare_temp_directory()
    buffer._copy_tiles_and_labels(
        [tile],
        "train",
    )

    label = (
        temp_dir
        / "labels"
        / "train"
        / "negative.txt"
    )
    assert label.exists()
    assert (
        label.read_text(
            encoding="utf-8"
        )
        == ""
    )
