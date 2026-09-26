from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import inference_pipeline as ip  # noqa: E402
from PIL import Image  # noqa: E402


class _Box:
    def __init__(self, xywhn, cls):
        import torch

        self.xywhn = torch.tensor([xywhn])
        self.conf = torch.tensor([0.9])
        self.cls = torch.tensor([float(cls)])


class _FakeModel:
    """Returns the given boxes per image; class defaults to batch position."""

    def __init__(self, boxes=None):
        self.calls = []
        self.boxes = boxes

    def predict(self, images, **_):
        self.calls.append(len(images))
        return [
            SimpleNamespace(
                boxes=[_Box(b, i) for b in (self.boxes or [[0.5, 0.5, 0.25, 0.25]])]
            )
            for i in range(len(images))
        ]


def _wsi(downsample=1.0, width=10_000, height=10_000):
    return SimpleNamespace(downsample=downsample, width=width, height=height)


def test_batched_results_map_to_their_own_tile(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "RAW_TILES_DIR", tmp_path)
    pipe = ip.InferencePipeline.__new__(ip.InferencePipeline)
    pipe.model = _FakeModel()
    pipe.run_id = "t"
    tile = Image.new("RGB", (ip.settings.tile_size, ip.settings.tile_size))
    batch = [(0, 0, tile, 1.0), (1000, 2000, tile, 1.0)]
    detections, tiles = [], []

    pipe._process_batch(
        batch, _wsi(downsample=2.0), "s.svs", detections, tiles
    )

    assert pipe.model.calls == [2]
    span = ip.settings.tile_size * 2.0
    assert [t["filename"] for t in tiles] == ["s_t_0_0.jpg", "s_t_1000_2000.jpg"]
    second = detections[1]
    assert second["tile_filename"] == "s_t_1000_2000.jpg"
    assert second["box"] == pytest.approx(
        [1000 + 0.375 * span, 2000 + 0.375 * span,
         1000 + 0.625 * span, 2000 + 0.625 * span]
    )


def test_boxes_cut_by_interior_tile_edges_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "RAW_TILES_DIR", tmp_path)
    size = ip.settings.tile_size
    pipe = ip.InferencePipeline.__new__(ip.InferencePipeline)
    # Whole cell, cell cut by the left edge, cell cut by the right edge.
    pipe.model = _FakeModel(
        [[0.5, 0.5, 0.2, 0.2], [0.05, 0.5, 0.1, 0.2], [0.95, 0.5, 0.1, 0.2]]
    )
    pipe.run_id = "t"
    tile = Image.new("RGB", (size, size))
    wsi = _wsi(width=3 * size, height=size)

    def centers(x0):
        detections = []
        pipe._process_batch([(x0, 0, tile, 1.0)], wsi, "s.svs", detections, [])
        return sorted(round((d["box"][0] + d["box"][2]) / 2 - x0) for d in detections)

    whole, left, right = round(0.5 * size), round(0.05 * size), round(0.95 * size)
    # First tile: its left edge is the slide border, so a box there is kept.
    assert centers(0) == [left, whole]
    # Middle tile: both edges are interior, only the whole cell survives.
    assert centers(size) == [whole]
    # Last tile: its right edge is the slide border.
    assert centers(2 * size) == [whole, right]
