from __future__ import annotations

"""Pure scene-document rules for automatic and manually edited ball paths."""

from copy import deepcopy
from datetime import UTC, datetime
from math import isfinite

from .reconstruction_errors import ReconstructionError


def ball_keyframe_documents(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [deepcopy(item) for item in value if isinstance(item, dict)]


def manual_ball_diagnostics(keyframes: list[dict]) -> dict:
    return {
        "trajectoryMode": "manual",
        "source": "manual-keypoints",
        "confidenceKind": "manual-authoritative",
        "manualKeyframeCount": len(keyframes),
        "interpolationSegmentCount": max(0, len(keyframes) - 1),
        "interpolation": {
            "method": "linear-between-keypoints",
            "scope": "between-first-and-last-keypoint",
        },
        # Detector coverage describes a different evidence source and must not
        # be copied onto a user-authored trajectory.
        "observedCoverage": None,
        "publishedCoverage": None,
        "worldProjectionStatus": "published" if keyframes else "no-manual-keypoints",
        "provenance": {
            "source": "manual",
            "method": "user-pitch-keypoint",
        },
    }


def normalize_ball_payload(value: object) -> dict:
    """Normalize the current reversible automatic/manual trajectory contract."""

    ball = deepcopy(value) if isinstance(value, dict) else {}
    mode = (
        ball.get("mode")
        if ball.get("mode") in {"automatic", "manual"}
        else "automatic"
    )
    automatic_keyframes = ball_keyframe_documents(ball.get("automaticKeyframes"))
    manual_keyframes = ball_keyframe_documents(ball.get("manualKeyframes"))
    automatic_diagnostics = (
        deepcopy(ball.get("automaticDiagnostics"))
        if isinstance(ball.get("automaticDiagnostics"), dict)
        else {}
    )
    automatic_diagnostics = {
        **automatic_diagnostics,
        "trajectoryMode": "automatic",
        "source": automatic_diagnostics.get("source")
        or "automatic-ball-resolver",
    }
    manual_diagnostics = manual_ball_diagnostics(manual_keyframes)

    ball.update(
        {
            "mode": mode,
            "automaticKeyframes": automatic_keyframes,
            "manualKeyframes": manual_keyframes,
            "automaticDiagnostics": automatic_diagnostics,
            "manualDiagnostics": manual_diagnostics,
            "keyframes": (
                manual_keyframes if mode == "manual" else automatic_keyframes
            ),
            "diagnostics": (
                manual_diagnostics if mode == "manual" else automatic_diagnostics
            ),
        }
    )
    return ball


def manual_ball_keyframes(scene: dict, values: list[dict]) -> list[dict]:
    duration = float(scene.get("duration") or 0.0)
    pitch = scene.get("payload", {}).get("pitch") or {}
    try:
        pitch_length = float(pitch["length"])
        pitch_width = float(pitch["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReconstructionError("Scene pitch dimensions are unavailable") from exc
    if not all(
        isfinite(item) and item > 0.0
        for item in (duration, pitch_length, pitch_width)
    ):
        raise ReconstructionError(
            "Scene duration and pitch dimensions must be finite and positive"
        )

    def number(item: dict, key: str, *, required: bool = True) -> float | None:
        raw = item.get(key)
        if raw is None and not required:
            return None
        if isinstance(raw, bool):
            raise ReconstructionError(
                f"Manual ball keyframe {key} must be a finite number"
            )
        try:
            result = float(raw)
        except (TypeError, ValueError) as exc:
            raise ReconstructionError(
                f"Manual ball keyframe {key} must be a finite number"
            ) from exc
        if not isfinite(result):
            raise ReconstructionError(
                f"Manual ball keyframe {key} must be a finite number"
            )
        return result

    by_time: dict[float, dict] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ReconstructionError("Manual ball keyframes must be objects")
        time = number(item, "t")
        x = number(item, "x")
        z = number(item, "z")
        y = number(item, "y", required=False)
        assert time is not None and x is not None and z is not None
        if time < 0.0 or time > duration:
            raise ReconstructionError(
                f"Manual ball keyframe time must be between 0 and {duration:g} seconds"
            )
        if abs(x) > pitch_length / 2.0:
            raise ReconstructionError(
                "Manual ball x must be within the pitch bounds "
                f"[-{pitch_length / 2:g}, {pitch_length / 2:g}]"
            )
        if abs(z) > pitch_width / 2.0:
            raise ReconstructionError(
                "Manual ball z must be within the pitch bounds "
                f"[-{pitch_width / 2:g}, {pitch_width / 2:g}]"
            )

        normalized_time = round(time, 3)
        resolved_y = 0.22 if y is None else round(y, 3)
        by_time[normalized_time] = {
            "id": f"manual-ball-{int(round(normalized_time * 1000)):06d}",
            "t": normalized_time,
            "x": round(x, 3),
            "y": resolved_y,
            "z": round(z, 3),
            "confidence": 1.0,
            "confidenceKind": "manual-authoritative",
            "detectionConfidence": None,
            "trajectoryConfidence": 1.0,
            "state": "observed",
            "observed": True,
            "positionSource": "manual-keypoint",
            "heightSource": "manual" if y is not None else "rendering-placeholder",
            "projectionSource": "manual-pitch-coordinate",
            "positionUncertaintyMetres": 0.0,
            "projection": {
                "source": "manual-pitch-coordinate",
                "calibrationFrameIndex": None,
                "uncertaintyMetres": 0.0,
            },
            "provenance": {
                "source": "manual",
                "method": "user-pitch-keypoint",
            },
        }
    return [by_time[time] for time in sorted(by_time)]


def edit_scene_ball_trajectory(
    scene: dict,
    mode: str,
    keyframes: list[dict] | None = None,
) -> dict:
    """Apply one validated editor change to an in-memory scene document."""

    if mode not in {"automatic", "manual"}:
        raise ReconstructionError("Ball trajectory mode must be automatic or manual")
    if mode == "automatic" and keyframes is not None:
        raise ReconstructionError(
            "Keyframes can only be supplied for manual ball trajectory mode"
        )

    payload = scene.setdefault("payload", {})
    ball = normalize_ball_payload(payload.get("ball"))
    if keyframes is not None:
        ball["manualKeyframes"] = manual_ball_keyframes(scene, keyframes)
        ball["manualUpdatedAt"] = datetime.now(UTC).isoformat()
    ball["mode"] = mode
    ball["manualDiagnostics"] = manual_ball_diagnostics(ball["manualKeyframes"])
    if mode == "manual":
        ball["keyframes"] = deepcopy(ball["manualKeyframes"])
        ball["diagnostics"] = deepcopy(ball["manualDiagnostics"])
    else:
        ball["keyframes"] = deepcopy(ball["automaticKeyframes"])
        ball["diagnostics"] = deepcopy(ball["automaticDiagnostics"])
    payload["ball"] = ball
    return scene


def publish_automatic_ball_trajectory(
    scene: dict,
    keyframes: list[dict],
    diagnostics: dict,
) -> dict:
    payload = scene.setdefault("payload", {})
    ball = normalize_ball_payload(payload.get("ball"))
    automatic_keyframes = deepcopy(keyframes)
    automatic_diagnostics = {
        **deepcopy(diagnostics),
        "trajectoryMode": "automatic",
        "source": "automatic-ball-resolver",
    }
    ball.update(
        {
            "automaticKeyframes": automatic_keyframes,
            "automaticDiagnostics": automatic_diagnostics,
            "automaticUpdatedAt": datetime.now(UTC).isoformat(),
        }
    )
    if ball["mode"] == "manual":
        ball["keyframes"] = deepcopy(ball["manualKeyframes"])
        ball["diagnostics"] = deepcopy(ball["manualDiagnostics"])
    else:
        ball["keyframes"] = deepcopy(automatic_keyframes)
        ball["diagnostics"] = deepcopy(automatic_diagnostics)
    payload["ball"] = ball
    return ball


__all__ = (
    "ball_keyframe_documents",
    "edit_scene_ball_trajectory",
    "manual_ball_diagnostics",
    "manual_ball_keyframes",
    "normalize_ball_payload",
    "publish_automatic_ball_trajectory",
)
