from app.main import _match_binding_snapshot
from app.schemas import (
    EventBundle,
    ExternalEvent,
    ExternalLineupEntry,
    ExternalPlayer,
    ExternalSubstitution,
    ExternalTeam,
    ExternalTimelineEvent,
)


def test_match_binding_snapshot_keeps_offline_roster_without_accepting_identity() -> None:
    bundle = EventBundle(
        event=ExternalEvent(
            id="event-1",
            name="Home v Away",
            home=ExternalTeam(id="home-api", name="Home"),
            away=ExternalTeam(id="away-api", name="Away"),
        ),
        players=[
            ExternalPlayer(
                id="player-8",
                name="Player Eight",
                team_id="home-api",
                team_name="Home",
                number="08",
            )
        ],
        lineup=[
            ExternalLineupEntry(
                id="lineup-8",
                player_id="player-8",
                player_name="Player Eight",
                team_id="home-api",
                team_name="Home",
                side="home",
                position="Midfielder",
                number="08",
                role="starter",
                order=0,
            )
        ],
        timeline=[
            ExternalTimelineEvent(
                id="timeline-1",
                minute=61,
                type="substitution",
                label="Substitution · Player Eight",
                player_id="player-8",
                player_name="Player Eight",
                team_id="home-api",
                secondary_player_id="player-18",
                secondary_player_name="Player Eighteen",
            )
        ],
        substitutions=[
            ExternalSubstitution(
                id="substitution-timeline-1",
                minute=61,
                team_id="home-api",
                player_out_id="player-8",
                player_out_name="Player Eight",
                player_in_id="player-18",
                player_in_name="Player Eighteen",
                label="Substitution · Player Eight",
            )
        ],
        fetched_at="2026-07-17T00:00:00Z",
        warnings=["partial lineup"],
    )

    snapshot = _match_binding_snapshot(bundle)

    assert snapshot["schemaVersion"] == 2
    assert snapshot["eventId"] == "event-1"
    assert snapshot["event"]["name"] == "Home v Away"
    assert snapshot["teams"]["home"]["id"] == "home-api"
    assert snapshot["players"][0]["id"] == "player-8"
    assert snapshot["players"][0]["number"] == "08"
    assert "externalPlayerId" not in snapshot["players"][0]
    assert snapshot["lineup"][0]["role"] == "starter"
    assert snapshot["timeline"][0]["secondary_player_id"] == "player-18"
    assert snapshot["substitutions"][0]["player_in_id"] == "player-18"
    assert snapshot["rosterQuality"] == {
        "status": "partial",
        "playerCount": 1,
        "homePlayerCount": 1,
        "awayPlayerCount": 0,
        "automaticIdentityEligible": False,
        "manualIdentityEligible": True,
        "reasons": ["fewer-than-eleven-players-per-team"],
    }
    assert snapshot["warnings"] == ["partial lineup"]


def test_five_player_snapshot_is_manual_only_even_without_provider_warning() -> None:
    event = ExternalEvent(
        id="event-capped",
        name="Home v Away",
        home=ExternalTeam(id="home-api", name="Home"),
        away=ExternalTeam(id="away-api", name="Away"),
    )
    bundle = EventBundle(
        event=event,
        players=[
            ExternalPlayer(
                id=f"player-{index}",
                name=f"Player {index}",
                team_id="home-api" if index < 3 else "away-api",
            )
            for index in range(5)
        ],
        fetched_at="2026-07-17T00:00:00Z",
    )

    snapshot = _match_binding_snapshot(bundle)

    assert snapshot["rosterQuality"]["automaticIdentityEligible"] is False
    assert snapshot["rosterQuality"]["manualIdentityEligible"] is True
    assert "provider-five-player-cap" in snapshot["rosterQuality"]["reasons"]
