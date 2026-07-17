from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ExternalTeam(BaseModel):
    id: str
    name: str
    badge: str | None = None


class ExternalPlayer(BaseModel):
    id: str
    name: str
    team_id: str | None = None
    team_name: str | None = None
    position: str | None = None
    number: str | None = None
    thumbnail: str | None = None
    lineup_role: Literal["starter", "substitute", "unknown"] = "unknown"
    lineup_order: int | None = None


class ExternalLineupEntry(BaseModel):
    """One provider lineup row, retained even when a player row is duplicated."""

    id: str
    player_id: str
    player_name: str
    team_id: str | None = None
    team_name: str | None = None
    side: Literal["home", "away", "unknown"] = "unknown"
    position: str | None = None
    number: str | None = None
    role: Literal["starter", "substitute", "unknown"] = "unknown"
    order: int
    formation: str | None = None
    grid: str | None = None


class ExternalTimelineEvent(BaseModel):
    id: str
    minute: int | None = None
    type: str
    label: str
    player_id: str | None = None
    player_name: str | None = None
    team_id: str | None = None
    team_name: str | None = None
    secondary_player_id: str | None = None
    secondary_player_name: str | None = None
    detail: str | None = None


class ExternalSubstitution(BaseModel):
    """A normalized substitution extracted from the provider match timeline."""

    id: str
    minute: int | None = None
    team_id: str | None = None
    team_name: str | None = None
    player_out_id: str | None = None
    player_out_name: str | None = None
    player_in_id: str | None = None
    player_in_name: str | None = None
    label: str


class ExternalRosterQuality(BaseModel):
    """Whether a provider roster may safely drive automatic identity matching.

    Partial rosters remain available for explicit human bindings.  This keeps
    a free API's truncated response useful without letting five arbitrary
    entries become the closed set for automatic identity resolution.
    """

    status: Literal["automatic-ready", "partial", "unavailable"]
    player_count: int = Field(ge=0)
    home_player_count: int = Field(ge=0)
    away_player_count: int = Field(ge=0)
    automatic_identity_eligible: bool
    manual_identity_eligible: bool
    reasons: list[str] = Field(default_factory=list)


class ExternalEvent(BaseModel):
    id: str
    provider: str | None = None
    name: str
    date: str | None = None
    time: str | None = None
    status: str | None = None
    league: str | None = None
    season: str | None = None
    home: ExternalTeam
    away: ExternalTeam
    home_score: int | None = None
    away_score: int | None = None
    thumbnail: str | None = None


class EventBundle(BaseModel):
    # Provider ids are registry-owned rather than a closed schema enum so a
    # new adapter does not require rewriting persisted project snapshots.
    source: str = Field(default="thesportsdb", min_length=1, max_length=80)
    event: ExternalEvent
    players: list[ExternalPlayer] = Field(default_factory=list)
    lineup: list[ExternalLineupEntry] = Field(default_factory=list)
    timeline: list[ExternalTimelineEvent] = Field(default_factory=list)
    substitutions: list[ExternalSubstitution] = Field(default_factory=list)
    roster_quality: ExternalRosterQuality | None = None
    fetched_at: str
    warnings: list[str] = Field(default_factory=list)


class SceneDocument(BaseModel):
    id: str
    title: str
    version: int = 1
    # Full-document compare-and-swap token. Legacy JSON without this field is
    # revision zero and is upgraded on its first successful write.
    revision: int = Field(default=0, ge=0)
    duration: float = Field(gt=0, le=120)
    payload: dict[str, Any]


class MatchBindingRequest(BaseModel):
    event_id: str
    provider: str | None = Field(default=None, min_length=1, max_length=80)


class ManualExternalTeam(ExternalTeam):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=240)
    badge: str | None = Field(default=None, max_length=2000)


class ManualExternalPlayer(ExternalPlayer):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=240)
    team_id: str | None = Field(default=None, max_length=160)
    team_name: str | None = Field(default=None, max_length=240)
    position: str | None = Field(default=None, max_length=120)
    number: str | None = Field(default=None, max_length=16)
    thumbnail: str | None = Field(default=None, max_length=2000)
    lineup_order: int | None = Field(default=None, ge=0, le=1000)


class ManualExternalLineupEntry(ExternalLineupEntry):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=160)
    player_id: str = Field(min_length=1, max_length=160)
    player_name: str = Field(min_length=1, max_length=240)
    team_id: str | None = Field(default=None, max_length=160)
    team_name: str | None = Field(default=None, max_length=240)
    position: str | None = Field(default=None, max_length=120)
    number: str | None = Field(default=None, max_length=16)
    order: int = Field(ge=0, le=1000)


class ManualExternalTimelineEvent(ExternalTimelineEvent):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=160)
    minute: int | None = Field(default=None, ge=0, le=300)
    type: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=500)


class ManualExternalSubstitution(ExternalSubstitution):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=160)
    minute: int | None = Field(default=None, ge=0, le=300)
    label: str = Field(min_length=1, max_length=500)


class ManualMatchEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=240)
    date: str | None = Field(default=None, max_length=40)
    time: str | None = Field(default=None, max_length=40)
    status: str | None = Field(default=None, max_length=80)
    league: str | None = Field(default=None, max_length=160)
    season: str | None = Field(default=None, max_length=80)
    home_score: int | None = Field(default=None, ge=0, le=100)
    away_score: int | None = Field(default=None, ge=0, le=100)
    thumbnail: str | None = Field(default=None, max_length=2000)


class ManualMatchTeams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    home: ManualExternalTeam
    away: ManualExternalTeam


class ManualMatchProvenance(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    label: str | None = Field(default=None, max_length=240)
    reference: str | None = Field(default=None, max_length=2000)
    captured_at: datetime | None = Field(
        default=None,
        alias="capturedAt",
    )
    notes: str | None = Field(default=None, max_length=4000)


class ManualMatchImportRequest(BaseModel):
    """Strict JSON fallback for a user-supplied, match-scoped roster."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event: ManualMatchEvent
    teams: ManualMatchTeams
    players: list[ManualExternalPlayer] = Field(min_length=1, max_length=100)
    lineup: list[ManualExternalLineupEntry] = Field(default_factory=list, max_length=100)
    timeline: list[ManualExternalTimelineEvent] = Field(default_factory=list, max_length=1000)
    substitutions: list[ManualExternalSubstitution] = Field(default_factory=list, max_length=200)
    provenance: ManualMatchProvenance | None = None


class MultiPassRequest(BaseModel):
    segment_ids: list[str] = Field(min_length=2, max_length=6)
    title: str | None = Field(default=None, max_length=160)
    manual_alignment_anchors: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=120,
        validation_alias=AliasChoices(
            "manualAlignmentAnchors",
            "manual_alignment_anchors",
        ),
    )


class FrameAnalysisRequest(BaseModel):
    scene_time: float = Field(ge=0, le=120)


FrameAnnotationKind = Literal[
    "home-player",
    "away-player",
    "home-goalkeeper",
    "away-goalkeeper",
    "referee",
    "other",
    "ignore",
]

FrameIdentityAction = Literal["confirm", "exclude", "merge", "split"]
FrameIdentityScope = Literal["observation", "range", "identity"]


class FrameAnnotationBox(BaseModel):
    x: float = Field(ge=0, le=8000)
    y: float = Field(ge=0, le=8000)
    width: float = Field(ge=4, le=8000)
    height: float = Field(ge=4, le=8000)


class FramePersonAnnotationRequest(BaseModel):
    annotation_id: str | None = Field(default=None, min_length=1, max_length=80)
    scene_time: float = Field(ge=0, le=120)
    bbox: FrameAnnotationBox
    kind: FrameAnnotationKind
    label: str | None = Field(default=None, max_length=120)
    external_player_id: str | None = Field(default=None, max_length=120)
    # Optional preserves the legacy `kind=ignore` contract.  When an action is
    # supplied it is authoritative and its kind/action pairing is validated by
    # the reconstruction layer.
    action: FrameIdentityAction | None = None
    scope: FrameIdentityScope = "identity"
    merge_target_id: str | None = Field(default=None, min_length=1, max_length=120)
    source_track_id: str | None = Field(default=None, min_length=1, max_length=120)
    # Authoritative identity key. `source_track_id` remains for old clients and
    # identifies only the optional 3D actor.
    canonical_person_id: str | None = Field(default=None, min_length=1, max_length=160)
    # Split corrections target one already-published detector observation. The
    # reconstruction layer snapshots its frame/time/bbox and applies the range
    # as [start, end); a detector row number is never accepted on its own.
    target_observation_id: str | None = Field(default=None, min_length=1, max_length=200)
    range_start: float | None = Field(default=None, ge=0, le=120)
    range_end: float | None = Field(default=None, gt=0, le=120)


class CanonicalRosterBindingRequest(BaseModel):
    """Bind one durable canonical identity to the saved match roster.

    A null player id is an explicit unbind decision.  The server, rather than
    the browser, chooses a persisted detector observation to anchor the
    identity correction.
    """

    external_player_id: str | None = Field(default=None, min_length=1, max_length=120)


ReconstructionModel = Literal[
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
]

BallDetectionBackend = Literal[
    "generic-ultralytics",
    "dedicated-ultralytics",
    "wasb-service",
]


class ReconstructionRequest(BaseModel):
    model: ReconstructionModel | None = None
    ball_backend: BallDetectionBackend | None = None


BallTrajectoryMode = Literal["automatic", "manual"]


class ManualBallKeyframe(BaseModel):
    t: float = Field(ge=0, allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)
    y: float | None = Field(default=None, allow_inf_nan=False)


class BallTrajectoryRequest(BaseModel):
    mode: BallTrajectoryMode
    keyframes: list[ManualBallKeyframe] | None = Field(
        default=None,
        max_length=2000,
    )


PlayerActionType = Literal[
    "idle",
    "walk",
    "run",
    "sprint",
    "turn",
    "jump",
    "fall",
    "get-up",
    "first-touch",
    "drive",
    "pass",
    "cross",
    "shot",
    "header",
    "throw-in",
    "clearance",
    "tackle",
    "slide-tackle",
    "block",
    "interception",
    "feint",
]

PlayerActionKeypointKind = Literal[
    "wind-up",
    "contact",
    "release",
    "apex",
    "impact",
    "recovery",
]


class PlayerActionKeypointRequest(BaseModel):
    """One compact semantic marker inside a player-action interval."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: PlayerActionKeypointKind
    # The owning scene, rather than an API-wide clip assumption, is the upper
    # bound. Highlights assembled from several passes may exceed two minutes.
    time: float = Field(ge=0, allow_inf_nan=False)


class PlayerActionUpsertRequest(BaseModel):
    """Fields a human may author; provenance/review state are server-owned."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    id: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    canonical_person_id: str = Field(
        alias="canonicalPersonId",
        min_length=1,
        max_length=160,
    )
    type: PlayerActionType
    start_time: float = Field(
        alias="startTime",
        ge=0,
        allow_inf_nan=False,
    )
    end_time: float = Field(
        alias="endTime",
        gt=0,
        allow_inf_nan=False,
    )
    keypoints: list[PlayerActionKeypointRequest] = Field(
        default_factory=list,
        max_length=24,
    )


PitchCalibrationPreset = Literal[
    "penalty-area-left",
    "goal-area-left",
    "center-circle",
    "goal-area-right",
    "penalty-area-right",
]


class CalibrationImagePoint(BaseModel):
    x: float = Field(ge=-4000, le=8000)
    y: float = Field(ge=-4000, le=8000)


class CalibrationPitchPoint(BaseModel):
    x: float = Field(ge=-52.5, le=52.5)
    z: float = Field(ge=-34, le=34)


class PitchCalibrationAnchor(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    image: CalibrationImagePoint
    pitch: CalibrationPitchPoint


class PitchCalibrationDraftRequest(BaseModel):
    scene_time: float = Field(ge=0, le=120)
    preset: PitchCalibrationPreset | None = None


class PitchSideRequest(BaseModel):
    side: Literal["left", "right"]


class PitchCalibrationPreviewRequest(BaseModel):
    scene_time: float = Field(ge=0, le=120)
    preset: PitchCalibrationPreset
    anchors: list[PitchCalibrationAnchor] = Field(min_length=4, max_length=12)


class SceneMatchBindingResponse(BaseModel):
    scene: SceneDocument
    bundle: EventBundle


class SceneSummary(BaseModel):
    id: str
    title: str
    duration: float
    kind: Literal["video", "segment", "multi-pass", "demo"]
    parent_scene_id: str | None = None
    updated_at: str | None = None


class CreateSceneRequest(BaseModel):
    event_id: str | None = None
    title: str | None = None
    provider: str | None = Field(default=None, min_length=1, max_length=80)


class VideoAsset(BaseModel):
    id: str
    filename: str
    original_name: str
    content_type: str
    status: Literal["queued", "processing", "ready", "failed"]
    stage: str
    progress: int = Field(ge=0, le=100)
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int = 0
    scene_id: str | None = None
    media_url: str | None = None
    poster_url: str | None = None
    error: str | None = None
    created_at: str | None = None
