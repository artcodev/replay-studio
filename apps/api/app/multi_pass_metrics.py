from __future__ import annotations

"""Evidence scoring for one camera pass and the resulting consensus."""


def pass_quality(scene: dict) -> float:
    payload = scene.get("payload") or {}
    reconstruction = (payload.get("videoAsset") or {}).get("reconstruction") or {}
    quality = reconstruction.get("quality") or {}
    verdict = str(reconstruction.get("qualityVerdict") or quality.get("verdict") or "")
    if not quality or not verdict:
        # A pass without the current QA contract is unknown evidence. Counts of
        # detections are never treated as a proxy for accuracy.
        return 0.0

    verdict_score = {
        "pass": 1.0,
        "review": 0.55,
        "reject": 0.0,
        "unknown": 0.15,
    }.get(verdict, 0.1)
    calibration_summary = (reconstruction.get("calibration") or {}).get("summary") or {}
    coverage = float(
        calibration_summary.get("usableCoverage")
        or calibration_summary.get("directCoverage")
        or 0.0
    )
    gates = quality.get("gates") or []
    if isinstance(gates, dict):
        gates = list(gates.values())
    required = [gate for gate in gates if gate.get("required", True)]
    gate_score = (
        sum(
            1.0
            if gate.get("status") == "pass"
            else 0.45
            if gate.get("status") == "review"
            else 0.0
            for gate in required
        )
        / len(required)
        if required
        else verdict_score
    )
    tracks = len(payload.get("tracks") or [])
    ball_samples = len((payload.get("ball") or {}).get("keyframes") or [])
    frame_count = int(reconstruction.get("frameCount") or 0)
    availability = (
        min(1.0, tracks / 14.0) * 0.55
        + min(1.0, ball_samples / 18.0) * 0.30
        + min(1.0, frame_count / 30.0) * 0.15
    )
    score = (
        verdict_score * 0.50
        + coverage * 0.25
        + gate_score * 0.20
        + availability * 0.05
    )
    return round(min(score, 0.24) if verdict == "reject" else score, 3)


def consensus_summary(passes: list[dict]) -> dict:
    ready = [item for item in passes if item.get("status") == "ready"]
    total = max(1, len(ready))
    metric_passes = sum(item.get("qualityVerdict") == "pass" for item in ready)
    ball_passes = sum(int(item.get("ballSamples") or 0) >= 3 for item in ready)
    track_passes = sum(int(item.get("trackCount") or 0) >= 6 for item in ready)
    overlapping_passes = sum(item.get("relation") == "replay-overlap" for item in ready)
    coverage = min(1.0, len(ready) / 3.0)
    evidence = (
        coverage * 0.25
        + metric_passes / total * 0.3
        + ball_passes / total * 0.25
        + track_passes / total * 0.2
    )
    return {
        "passesAnalyzed": len(ready),
        "metricPasses": metric_passes,
        "ballPasses": ball_passes,
        "trackPasses": track_passes,
        "overlappingPasses": overlapping_passes,
        "evidenceScore": round(evidence, 3),
    }


def pass_summary(
    scene: dict,
    segment: dict,
    status: str = "ready",
    error: str | None = None,
) -> dict:
    payload = scene.get("payload") or {}
    reconstruction = (payload.get("videoAsset") or {}).get("reconstruction") or {}
    calibration = reconstruction.get("pitchCalibration") or {}
    return {
        "sceneId": scene.get("id"),
        "segmentId": segment["id"],
        "label": segment.get("label") or segment["id"],
        "sourceStart": segment.get("start"),
        "sourceEnd": segment.get("end"),
        "status": status,
        "quality": pass_quality(scene) if status == "ready" else 0.0,
        "trackCount": len(payload.get("tracks") or []),
        "ballSamples": len((payload.get("ball") or {}).get("keyframes") or []),
        "calibrationStatus": calibration.get("status") or "unavailable",
        "calibrationConfidence": calibration.get("confidence"),
        "qualityVerdict": reconstruction.get("qualityVerdict")
        or (reconstruction.get("quality") or {}).get("verdict"),
        "error": error,
    }
