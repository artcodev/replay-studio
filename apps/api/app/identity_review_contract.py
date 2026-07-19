from __future__ import annotations

"""Strict HTTP response contract for the identity-review workbench."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from .transport_contract import TransportContract


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(item[:1].upper() + item[1:] for item in tail)


class IdentityReviewContract(TransportContract):
    model_config = ConfigDict(
        extra="forbid",
        alias_generator=_camel,
        populate_by_name=True,
    )


class IdentityReviewBox(IdentityReviewContract):
    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class IdentityReviewCropDiagnostic(IdentityReviewContract):
    status: str | None = None
    usable: bool | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    number: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class IdentityReviewObservation(IdentityReviewContract):
    observation_id: str = Field(min_length=1)
    frame_index: int = Field(ge=0)
    source_frame_index: int | None = Field(default=None, ge=0)
    scene_time: float = Field(ge=0, allow_inf_nan=False)
    source_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    bbox: IdentityReviewBox
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_quality: float = Field(ge=0, allow_inf_nan=False)
    crop_url: str | None = None
    rejection_reasons: list[str] = Field(default_factory=list)
    reid: IdentityReviewCropDiagnostic | None = None
    jersey_ocr: IdentityReviewCropDiagnostic | None = None


class IdentityReviewJerseyVote(IdentityReviewContract):
    number: str
    support_count: int = Field(ge=0)
    weight_share: float = Field(ge=0, le=1)


class IdentityReviewEvidence(IdentityReviewContract):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: str | int | float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    support_count: int | None = Field(default=None, ge=0)
    sample_count: int | None = Field(default=None, ge=0)
    source: str | None = None
    model: str | None = None
    frame_indices: list[int] | None = None
    manual: bool | None = None
    status: str | None = None
    votes: list[IdentityReviewJerseyVote] | None = None
    unique_evidence_fingerprint_count: int | None = Field(default=None, ge=0)
    duplicate_evidence_fingerprint_count: int | None = Field(default=None, ge=0)
    selection_policy: str | None = None
    selected_frame_indices: list[int] | None = None
    selected_qualities: list[float] | None = None
    selected_evidence_fingerprints: list[str] | None = None
    partition: str | None = None
    source_scene_id: str | None = None
    source_canonical_person_id: str | None = None
    signals: list[str] | None = None
    alignment_confidence: float | None = Field(default=None, ge=0, le=1)
    alignment_method: str | None = None
    observation_count: int | None = Field(default=None, ge=0)


class IdentityReviewCandidateEvidence(IdentityReviewContract):
    code: str
    score_delta: float
    confidence: float = Field(ge=0, le=1)
    source: str
    details: list[str] = Field(default_factory=list)


class IdentityReviewRosterCandidate(IdentityReviewContract):
    external_player_id: str = Field(min_length=1)
    rank: int | None = Field(default=None, ge=0)
    score: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    identity_signal_score: float | None = Field(default=None, ge=0, le=1)
    name: str | None = None
    number: str | None = None
    position: str | None = None
    team_id: str | None = None
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    eligible: bool | None = None
    proposal_status: str | None = None
    requires_manual_confirmation: bool | None = None
    evidence: list[IdentityReviewCandidateEvidence] = Field(default_factory=list)


class IdentityReviewConflict(IdentityReviewContract):
    id: str = Field(min_length=1)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: Literal["review", "blocking"]
    related_canonical_person_ids: list[str] = Field(default_factory=list)
    related_tracklet_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    external_player_id: str | None = None
    expected_number: str | None = None
    observed_number: str | None = None
    binding_annotation_ids: list[str] = Field(default_factory=list)
    roster_status: str | None = None


class IdentityReviewItem(IdentityReviewContract):
    canonical_person_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    identity_status: Literal["resolved", "provisional", "excluded"]
    identity_confidence: float | None = Field(default=None, ge=0, le=1)
    identity_source: str | None = None
    team_id: str | None = None
    role: str | None = None
    jersey_number: str | None = None
    candidate_number: str | None = None
    external_player_id: str | None = None
    render_track_id: str | None = None
    observation_count: int = Field(ge=0)
    resolution_state: Literal[
        "conflict", "suggested", "anonymous", "bound", "excluded"
    ]
    priority: int = Field(ge=0)
    representative_observations: list[IdentityReviewObservation]
    evidence: list[IdentityReviewEvidence]
    roster_candidates: list[IdentityReviewRosterCandidate]
    conflicts: list[IdentityReviewConflict]


class IdentityReviewRosterStatus(IdentityReviewContract):
    status: Literal["ready", "incomplete", "review", "unavailable"]
    player_count: int = Field(ge=0)
    complete: bool
    automatic_identity_eligible: bool
    manual_identity_eligible: bool
    reasons: list[str]
    warnings: list[str]


class IdentityReviewMatchSnapshot(IdentityReviewContract):
    id: str | None = None
    content_hash: str | None = None
    match_id: str | None = None
    roster: IdentityReviewRosterStatus


class IdentityReviewWorkerHealth(IdentityReviewContract):
    configured: bool | None = None
    status: str
    backend: str | None = None
    provider_version: str | None = None
    model_version: str | None = None
    device: str | int | None = None
    batch_size: int | None = Field(default=None, ge=0)
    model_load_seconds: float | None = Field(default=None, ge=0)
    contract_version: str | None = None
    inference_scope: str | None = None
    dimension: int | None = Field(default=None, ge=0)
    normalized: bool | None = None
    evidence_fingerprint_version: str | None = None
    soccer_net_commit: str | None = None
    detail: str | None = None
    requested_observation_count: int | None = Field(default=None, ge=0)
    submitted_crop_count: int | None = Field(default=None, ge=0)
    selected_crop_count: int | None = Field(default=None, ge=0)
    usable_observation_count: int | None = Field(default=None, ge=0)
    recognized_crop_count: int | None = Field(default=None, ge=0)
    raw_usable_observation_count: int | None = Field(default=None, ge=0)
    rejected_observation_count: int | None = Field(default=None, ge=0)
    rejected_crop_count: int | None = Field(default=None, ge=0)
    rejection_reasons: list[str] | None = None


class IdentityReviewWorkers(IdentityReviewContract):
    identity: IdentityReviewWorkerHealth | None = None
    reid: IdentityReviewWorkerHealth | None = None
    jersey_ocr: IdentityReviewWorkerHealth | None = None


class IdentityReviewAvailability(IdentityReviewContract):
    state: Literal[
        "not-started",
        "queued",
        "processing",
        "failed",
        "cancelled",
        "unavailable",
        "ready",
    ]
    available: bool
    reason_code: Literal[
        "identity-review-artifacts-not-published",
        "reconstruction-state-unrecognized",
    ] | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "IdentityReviewAvailability":
        if self.available != (self.state == "ready"):
            raise ValueError("Availability must be true only for the ready state")
        if self.state == "unavailable":
            if self.reason_code is None:
                raise ValueError("Unavailable identity review requires a reason code")
        elif self.reason_code is not None:
            raise ValueError("Only unavailable identity review may have a reason code")
        return self


class IdentityReviewSummary(IdentityReviewContract):
    canonical_person_count: int = Field(ge=0)
    bound_count: int = Field(ge=0)
    suggested_count: int = Field(ge=0)
    conflict_count: int = Field(ge=0)
    anonymous_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)


class IdentityReviewResponse(IdentityReviewContract):
    scene_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    availability: IdentityReviewAvailability
    match_snapshot: IdentityReviewMatchSnapshot
    workers: IdentityReviewWorkers
    summary: IdentityReviewSummary
    items: list[IdentityReviewItem]


__all__ = ("IdentityReviewResponse",)
