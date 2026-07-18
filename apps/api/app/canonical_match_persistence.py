from __future__ import annotations

"""Atomic application workflow for publishing normalized Match data."""

from copy import deepcopy

from .canonical_match import CANONICAL_MATCH_SCHEMA_VERSION, canonicalize_event_bundle
from .external_reference_repository import (
    ExternalReferenceRepository,
    external_references,
)
from .project_match_repository import ProjectMatchRepository, project_matches
from .project_match_persistence_contract import (
    ExternalReferenceCreate,
    MatchSnapshotCreate,
    MatchUpsert,
)
from .match_contracts import EventBundle


def persist_canonical_match(
    project_id: str,
    bundle: EventBundle,
    *,
    matches: ProjectMatchRepository = project_matches,
    references: ExternalReferenceRepository = external_references,
) -> dict:
    """Atomically publish Match, provider references, and immutable snapshot."""

    canonical = canonicalize_event_bundle(bundle)
    match_id = str(canonical["id"])
    match = MatchUpsert(
        id=match_id,
        name=str(canonical["name"]),
        competition=canonical.get("competition"),
        season=canonical.get("season"),
        kickoff_at=" ".join(
            value
            for value in (canonical.get("date"), canonical.get("time"))
            if value
        )
        or None,
        status=canonical.get("status"),
        home_team_name=canonical["homeTeam"]["name"],
        away_team_name=canonical["awayTeam"]["name"],
        metadata={
            "score": deepcopy(canonical["score"]),
            "homeTeamId": canonical["homeTeam"]["id"],
            "awayTeamId": canonical["awayTeam"]["id"],
        },
    )
    provider_references = [
        ExternalReferenceCreate(
            resource_type="match",
            resource_id=match_id,
            provider=bundle.source,
            external_type="match",
            external_id=str(bundle.event.id),
            payload={"name": bundle.event.name},
        ),
        ExternalReferenceCreate(
            resource_type="team",
            resource_id=canonical["homeTeam"]["id"],
            provider=bundle.source,
            external_type="team",
            external_id=str(bundle.event.home.id),
            payload={"name": bundle.event.home.name},
        ),
        ExternalReferenceCreate(
            resource_type="team",
            resource_id=canonical["awayTeam"]["id"],
            provider=bundle.source,
            external_type="team",
            external_id=str(bundle.event.away.id),
            payload={"name": bundle.event.away.name},
        ),
    ]
    canonical_players = {
        str(player.id): canonical_player
        for player, canonical_player in zip(bundle.players, canonical["roster"])
    }
    provider_references.extend(
        ExternalReferenceCreate(
            resource_type="player",
            resource_id=canonical_player["id"],
            provider=bundle.source,
            external_type="player",
            external_id=str(player.id),
            payload={"name": player.name},
        )
        for player in bundle.players
        if (canonical_player := canonical_players.get(str(player.id))) is not None
    )
    snapshot = MatchSnapshotCreate(
        provider=bundle.source,
        external_event_id=str(bundle.event.id),
        schema_version=CANONICAL_MATCH_SCHEMA_VERSION,
        fetched_at=bundle.fetched_at,
        payload=canonical,
    )
    with matches.transaction() as session:
        matches.publish_in_transaction(
            session,
            project_id,
            match,
            snapshot,
        )
        references.upsert_many_in_transaction(session, provider_references)
    return canonical
