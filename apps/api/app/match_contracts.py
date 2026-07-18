"""Provider-neutral and manual-import match transport contracts."""

from datetime import datetime
from typing import Literal

from pydantic import ConfigDict, Field

from .transport_contract import TransportContract


class ExternalTeam(TransportContract):
    id: str
    name: str
    badge: str | None = None


class ExternalPlayer(TransportContract):
    id: str
    name: str
    team_id: str | None = None
    team_name: str | None = None
    position: str | None = None
    number: str | None = None
    thumbnail: str | None = None
    lineup_role: Literal["starter", "substitute", "unknown"] = "unknown"
    lineup_order: int | None = None


class ExternalLineupEntry(TransportContract):
    """One normalized provider lineup row."""

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


class ExternalTimelineEvent(TransportContract):
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


class ExternalSubstitution(TransportContract):
    id: str
    minute: int | None = None
    team_id: str | None = None
    team_name: str | None = None
    player_out_id: str | None = None
    player_out_name: str | None = None
    player_in_id: str | None = None
    player_in_name: str | None = None
    label: str


class ExternalRosterQuality(TransportContract):
    """Whether normalized roster evidence is safe for automatic identity use."""

    status: Literal["automatic-ready", "partial", "unavailable"]
    player_count: int = Field(ge=0)
    home_player_count: int = Field(ge=0)
    away_player_count: int = Field(ge=0)
    automatic_identity_eligible: bool
    manual_identity_eligible: bool
    reasons: list[str] = Field(default_factory=list)


class ExternalEvent(TransportContract):
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


class EventBundle(TransportContract):
    source: str = Field(min_length=1, max_length=80)
    event: ExternalEvent
    players: list[ExternalPlayer] = Field(default_factory=list)
    lineup: list[ExternalLineupEntry] = Field(default_factory=list)
    timeline: list[ExternalTimelineEvent] = Field(default_factory=list)
    substitutions: list[ExternalSubstitution] = Field(default_factory=list)
    roster_quality: ExternalRosterQuality | None = None
    fetched_at: str
    warnings: list[str] = Field(default_factory=list)


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


class ManualMatchEvent(TransportContract):
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


class ManualMatchTeams(TransportContract):
    model_config = ConfigDict(extra="forbid")

    home: ManualExternalTeam
    away: ManualExternalTeam


class ManualMatchProvenance(TransportContract):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    label: str | None = Field(default=None, max_length=240)
    reference: str | None = Field(default=None, max_length=2000)
    captured_at: datetime | None = Field(default=None, alias="capturedAt")
    notes: str | None = Field(default=None, max_length=4000)


class ManualMatchImportRequest(TransportContract):
    """Strict JSON contract for a user-supplied match roster."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event: ManualMatchEvent
    teams: ManualMatchTeams
    players: list[ManualExternalPlayer] = Field(min_length=1, max_length=100)
    lineup: list[ManualExternalLineupEntry] = Field(default_factory=list, max_length=100)
    timeline: list[ManualExternalTimelineEvent] = Field(default_factory=list, max_length=1000)
    substitutions: list[ManualExternalSubstitution] = Field(default_factory=list, max_length=200)
    provenance: ManualMatchProvenance | None = None
