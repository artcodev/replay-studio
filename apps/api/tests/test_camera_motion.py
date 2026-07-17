import cv2
import numpy as np

from app.reconstruction import _camera_motion_estimate


def _textured_pitch(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((360, 640, 3), (45, 125, 45), dtype=np.uint8)
    for x, y in rng.integers([20, 20], [620, 340], size=(300, 2)):
        cv2.circle(image, (int(x), int(y)), 2, (230, 230, 230), -1)
    for x in (100, 320, 540):
        cv2.line(image, (x, 15), (x, 345), (245, 245, 245), 2)
    return image


def test_static_camera_is_a_valid_estimate_not_an_unestimated_identity() -> None:
    image = _textured_pitch()

    estimate = _camera_motion_estimate(image, image.copy())

    assert estimate.status == "estimated"
    assert estimate.reliable
    assert estimate.confidence > 0.9
    np.testing.assert_allclose(estimate.matrix, np.eye(3), atol=1e-5)


def test_camera_motion_maps_current_frame_back_to_previous_frame() -> None:
    previous = _textured_pitch()
    current = cv2.warpAffine(
        previous,
        np.float32([[1.0, 0.0, 9.0], [0.0, 1.0, 4.0]]),
        (previous.shape[1], previous.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

    estimate = _camera_motion_estimate(previous, current)

    assert estimate.status == "estimated"
    projected = estimate.matrix @ np.asarray([200.0 + 9.0, 140.0 + 4.0, 1.0])
    projected /= projected[2]
    np.testing.assert_allclose(projected[:2], [200.0, 140.0], atol=0.15)


def test_unrelated_shot_is_a_hard_camera_cut() -> None:
    previous = _textured_pitch()
    current = np.full_like(previous, (140, 45, 35))
    cv2.putText(
        current,
        "REPLAY",
        (120, 210),
        cv2.FONT_HERSHEY_DUPLEX,
        3.0,
        (255, 255, 255),
        6,
    )

    estimate = _camera_motion_estimate(previous, current)

    assert estimate.status == "cut"
    assert not estimate.reliable
    assert estimate.reason
