from __future__ import annotations

import cv2
import numpy as np

from app.person_detection_cache import frame_content_sha256
from app.pitch_line_mask_cache import (
    cached_pitch_line_mask_loader,
    lookup_pitch_line_mask,
    store_pitch_line_mask,
)


def _mask() -> np.ndarray:
    mask = np.zeros((60, 100), dtype=np.uint8)
    mask[30, 10:90] = 255
    return mask


def test_mask_round_trip_and_corrupt_artifacts_are_misses(tmp_path):
    digest = "a" * 64
    store_pitch_line_mask(tmp_path, frame_sha256=digest, mask=_mask())

    hit = lookup_pitch_line_mask(tmp_path, frame_sha256=digest)
    assert hit.status == "hit"
    assert np.array_equal(hit.mask, _mask())

    assert lookup_pitch_line_mask(tmp_path, frame_sha256="b" * 64).status == "absent"

    # Tampered bytes fail the checksum and become an ordinary miss.
    envelope_path = next(tmp_path.rglob("*.json"))
    envelope_path.write_text(envelope_path.read_text().replace('"a"', '"x"')[:-40])
    assert lookup_pitch_line_mask(tmp_path, frame_sha256=digest).status == "corrupt"


def test_warm_loader_never_decodes_the_frame_again(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame_00001.jpg"
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    # Paint a white line on green so the mask is non-trivial.
    image[:, :] = (40, 160, 60)
    image[30, 10:90] = (240, 240, 240)
    assert cv2.imwrite(str(frame_path), image)

    cache_dir = tmp_path / "pitch-line-masks"
    loader = cached_pitch_line_mask_loader(cache_dir, enabled=True)
    first = loader(frame_path)
    assert first is not None
    assert (
        lookup_pitch_line_mask(
            cache_dir, frame_sha256=frame_content_sha256(frame_path)
        ).status
        == "hit"
    )

    import app.pitch_line_mask_cache as cache_module

    monkeypatch.setattr(
        cache_module.cv2,
        "imread",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("a warm mask cache must not decode the frame")
        ),
    )
    warm_loader = cache_module.cached_pitch_line_mask_loader(
        cache_dir, enabled=True
    )
    second = warm_loader(frame_path)
    assert np.array_equal(first, second)


def test_missing_frame_path_disables_the_cache_transparently(tmp_path):
    loader = cached_pitch_line_mask_loader(
        tmp_path / "pitch-line-masks", enabled=True
    )
    # Synthetic test paths cannot be hashed: the loader falls back to the
    # plain decode path, which reports an unreadable frame as None.
    assert loader(tmp_path / "does-not-exist.jpg") is None
    assert not (tmp_path / "pitch-line-masks").exists()
