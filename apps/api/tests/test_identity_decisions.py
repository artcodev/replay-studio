import pytest

from app.identity_decisions import (
    IdentityDecisionError,
    clear_roster_candidate_rejection,
    reject_roster_candidate,
    rejected_roster_candidate_ids,
)


def scene() -> dict:
    return {
        "id": "identity-decisions",
        "payload": {
            "matchBinding": {
                "players": [
                    {"id": "player-8", "name": "Eight"},
                    {"id": "player-10", "name": "Ten"},
                ]
            },
            "canonicalPeople": [
                {
                    "canonicalPersonId": "canonical-1",
                    "externalPlayerId": None,
                    "observations": [{"observationId": "obs-1"}],
                    "rosterCandidates": [
                        {"externalPlayerId": "player-8"},
                        {"externalPlayerId": "player-10"},
                    ],
                }
            ],
        },
    }


def test_roster_candidate_rejection_is_idempotent_and_clearable():
    value = scene()
    first = reject_roster_candidate(value, "canonical-1", "player-8")
    second = reject_roster_candidate(value, "canonical-1", "player-8")

    assert first == second
    assert first["anchorObservationId"] == "obs-1"
    assert rejected_roster_candidate_ids(value, "canonical-1") == {"player-8"}

    removed = clear_roster_candidate_rejection(
        value, "canonical-1", "player-8"
    )
    assert removed == first
    assert rejected_roster_candidate_ids(value, "canonical-1") == set()


def test_cannot_reject_confirmed_or_unpublished_candidate():
    value = scene()
    value["payload"]["canonicalPeople"][0]["externalPlayerId"] = "player-8"
    with pytest.raises(IdentityDecisionError, match="Unbind"):
        reject_roster_candidate(value, "canonical-1", "player-8")

    value["payload"]["canonicalPeople"][0]["externalPlayerId"] = None
    with pytest.raises(IdentityDecisionError, match="published"):
        reject_roster_candidate(value, "canonical-1", "missing")
