import asyncio

from app.providers.thesportsdb import TheSportsDbProvider
from app.sample import make_video_scene


def test_maps_event_contract():
    provider = TheSportsDbProvider()
    event = provider._map_event(
        {
            "idEvent": "42",
            "strEvent": "Aurora vs Atlas",
            "dateEvent": "2026-07-11",
            "idHomeTeam": "1",
            "strHomeTeam": "Aurora",
            "idAwayTeam": "2",
            "strAwayTeam": "Atlas",
            "intHomeScore": "2",
            "intAwayScore": "1",
        }
    )

    assert event.id == "42"
    assert event.home.name == "Aurora"
    assert event.away_score == 1


def test_maps_lineup_side_to_national_team():
    provider = TheSportsDbProvider()
    event = provider._map_event(
        {
            "idEvent": "2519345",
            "strEvent": "Spain vs Belgium",
            "idHomeTeam": "133909",
            "strHomeTeam": "Spain",
            "idAwayTeam": "134515",
            "strAwayTeam": "Belgium",
        }
    )

    player = provider._map_lineup_player(
        {
            "idPlayer": "34146306",
            "strPlayer": "Thibaut Courtois",
            "strHome": "No",
            "idTeam": "133738",
            "strTeam": "Real Madrid",
        },
        event,
        0,
    )

    assert player.team_id == "134515"
    assert player.team_name == "Belgium"


def test_search_events_accepts_singular_event_root(monkeypatch):
    provider = TheSportsDbProvider()

    async def fake_get(endpoint, params, ttl=300):
        assert endpoint == "searchevents.php"
        assert params == {"e": "Spain_vs_Belgium"}
        return {
            "event": [
                {
                    "idEvent": "2519345",
                    "strEvent": "Spain vs Belgium",
                    "strSport": "Soccer",
                    "idHomeTeam": "133909",
                    "strHomeTeam": "Spain",
                    "idAwayTeam": "134515",
                    "strAwayTeam": "Belgium",
                }
            ]
        }

    monkeypatch.setattr(provider, "_get", fake_get)
    events = asyncio.run(provider.search_events("Spain vs Belgium"))

    assert [event.id for event in events] == ["2519345"]


def test_video_scene_starts_without_fabricated_tracks():
    scene = make_video_scene(
        scene_id="video-test",
        title="Source clip",
        duration=8.25,
        video_asset={"id": "asset-test", "mediaUrl": "/media"},
    )

    assert scene["duration"] == 8.25
    assert scene["payload"]["tracks"] == []
    assert scene["payload"]["videoAsset"]["id"] == "asset-test"
