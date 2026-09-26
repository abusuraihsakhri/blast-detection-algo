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
    """Returns one centered box per image, class = position in the batch."""

    def __init__(self):
        self.calls = []

    def predict(self, images, **_):
        self.calls.append(len(images))
        return [
            SimpleNamespace(boxes=[_Box([0.5, 0.5, 0.25, 0.25], i)])
            for i in range(len(images))
        ]


def test_batched_results_map_to_their_own_tile(tmp_path, monkeypatch):
    monkeypatch.setattr(ip, "RAW_TILES_DIR", tmp_path)
    pipe = ip.InferencePipeline.__new__(ip.InferencePipeline)
    pipe.model = _FakeModel()
    pipe.run_id = "t"
    tile = Image.new("RGB", (ip.settings.tile_size, ip.settings.tile_size))
    batch = [(0, 0, tile, 1.0), (1000, 2000, tile, 1.0)]
    detections, tiles = [], []

    pipe._process_batch(
        batch, SimpleNamespace(downsample=2.0), "s.svs", detections, tiles
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
