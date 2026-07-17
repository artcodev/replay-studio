from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, date as date_type, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import get_settings
from .calibration_worker import calibration_worker_readiness
from .ball_worker import ball_worker_readiness
from .identity_worker import identity_worker_readiness
from .identity_decision_routes import router as identity_decision_router
from .identity_review_routes import router as identity_review_router
from .jersey_ocr_worker import jersey_ocr_worker_readiness
from .database import init_database
from .model_comparison import compare_scene_models
from .multi_pass import MultiPassError, analyze_multi_pass_by_id, create_multi_pass_scene
from .providers.base import MatchDataError
from .providers.registry import sports_provider
from .project_match import (
    MULTI_PASS_MATCH_BINDING_STALE_WARNING,
    is_multi_pass_scene,
    is_single_pass_reconstruction_scene,
    mark_multi_pass_match_binding_stale,
    project_match_binding,
    semantic_match_binding,
)
from .player_actions import (
    PlayerActionError,
    delete_player_action,
    upsert_player_action,
)
from .reconstruction import (
    CANONICAL_ROSTER_BINDING_CORRECTION,
    ReconstructionError,
    StaleReconstructionRun,
    analyze_scene_frame,
    apply_scene_pitch_calibration,
    clear_canonical_roster_binding,
    delete_frame_person_annotation,
    preview_scene_pitch_calibration,
    propose_scene_pitch_calibration,
    queue_reconstruction,
    reconstruct_scene_by_id,
    set_canonical_roster_binding,
    set_scene_ball_trajectory,
    set_scene_pitch_side,
    upsert_frame_person_annotation,
)
from .reconstruction_recovery import start_queued_reconstruction_recovery
from .sample import make_demo_scene
from .segment_layout import propose_segment_layout
from .schemas import (
    BallTrajectoryRequest,
    CanonicalRosterBindingRequest,
    CreateSceneRequest,
    EventBundle,
    ExternalEvent,
    ExternalRosterQuality,
    FrameAnalysisRequest,
    FramePersonAnnotationRequest,
    ManualMatchImportRequest,
    MatchBindingRequest,
    MultiPassRequest,
    PitchCalibrationDraftRequest,
    PitchCalibrationPreviewRequest,
    PitchSideRequest,
    PlayerActionUpsertRequest,
    ReconstructionRequest,
    SceneDocument,
    SceneMatchBindingResponse,
    SceneSummary,
    VideoAsset,
)
from .store import (
    RECONSTRUCTION_INPUT_CHANGED_ERROR,
    SceneRevisionConflict,
    reconstruction_input_fingerprint,
    scene_store,
)
from .video_processing import asset_directory, materialize_segment_scene, process_video_asset
from .video_store import video_store


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    scene_store.seed()
    _migrate_legacy_project_match_bindings()
    recovery_monitor = start_queued_reconstruction_recovery()
    try:
        yield
    finally:
        stop_monitor = getattr(recovery_monitor, "stop", None)
        if callable(stop_monitor):
            stop_monitor()


settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(identity_review_router)
app.include_router(identity_decision_router)


@app.exception_handler(SceneRevisionConflict)
async def scene_revision_conflict_handler(
    _request: Request,
    _error: SceneRevisionConflict,
) -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={
            "detail": "The scene changed while it was being saved; reload and retry."
        },
    )


def _roster_quality_payload(bundle: EventBundle) -> dict:
    quality = bundle.roster_quality
    if quality is not None:
        return {
            "status": quality.status,
            "playerCount": quality.player_count,
            "homePlayerCount": quality.home_player_count,
            "awayPlayerCount": quality.away_player_count,
            "automaticIdentityEligible": quality.automatic_identity_eligible,
            "manualIdentityEligible": quality.manual_identity_eligible,
            "reasons": list(quality.reasons),
        }

    home_count = sum(
        player.team_id == bundle.event.home.id for player in bundle.players
    )
    away_count = sum(
        player.team_id == bundle.event.away.id for player in bundle.players
    )
    reasons: list[str] = []
    if not bundle.players:
        reasons.append("roster-unavailable")
    if bundle.source == "thesportsdb" and len(bundle.players) == 5:
        reasons.append("provider-five-player-cap")
    if bundle.players and (home_count < 11 or away_count < 11):
        reasons.append("fewer-than-eleven-players-per-team")
    automatic = bool(bundle.players) and not reasons
    return {
        "status": (
            "automatic-ready"
            if automatic
            else "partial"
            if bundle.players
            else "unavailable"
        ),
        "playerCount": len(bundle.players),
        "homePlayerCount": home_count,
        "awayPlayerCount": away_count,
        "automaticIdentityEligible": automatic,
        "manualIdentityEligible": bool(bundle.players),
        "reasons": reasons,
    }


def _match_data_http_exception(error: MatchDataError) -> HTTPException:
    status = 502
    if error.code in {"unknown-provider", "team-pair-required"}:
        status = 422
    elif error.code == "provider-not-configured":
        status = 503
    elif error.code == "event-not-found":
        status = 404
    elif error.code == "provider-rate-limit":
        status = 503
    headers = {
        "X-Match-Data-Provider": error.provider,
        "X-Match-Data-Error": error.code,
    }
    if error.retryable:
        headers["Retry-After"] = "60"
    return HTTPException(
        status_code=status,
        detail=str(error),
        headers=headers,
    )


def _match_binding_snapshot(
    bundle: EventBundle,
    *,
    provenance: dict | None = None,
) -> dict:
    """Persist the complete provider snapshot needed by offline identity work.

    The third-party catalog remains the source of truth, but a reconstruction
    must be repeatable while it is temporarily unavailable. Candidate names
    are only hypotheses; this snapshot never creates an accepted player
    binding by itself.
    """

    snapshot = {
        "schemaVersion": 2,
        "source": bundle.source,
        "eventId": bundle.event.id,
        "fetchedAt": bundle.fetched_at,
        "event": bundle.event.model_dump(),
        "teams": {
            "home": bundle.event.home.model_dump(),
            "away": bundle.event.away.model_dump(),
        },
        "players": [player.model_dump() for player in bundle.players],
        "lineup": [entry.model_dump() for entry in bundle.lineup],
        "timeline": [event.model_dump() for event in bundle.timeline],
        "substitutions": [item.model_dump() for item in bundle.substitutions],
        "rosterQuality": _roster_quality_payload(bundle),
        "warnings": list(bundle.warnings),
    }
    if provenance is not None:
        snapshot["provenance"] = provenance
    return snapshot


def _manual_identifier(value: str, label: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError(f"{label} must not be empty")
    if len(identifier) > 160:
        raise ValueError(f"{label} is longer than 160 characters")
    return identifier


def _manual_match_bundle(
    request: ManualMatchImportRequest,
) -> tuple[EventBundle, dict]:
    """Validate cross-record references and build one normalized manual bundle."""

    event_id = _manual_identifier(request.event.id, "event.id")
    home = request.teams.home.model_dump()
    away = request.teams.away.model_dump()
    home["id"] = _manual_identifier(home["id"], "teams.home.id")
    away["id"] = _manual_identifier(away["id"], "teams.away.id")
    if home["id"] == away["id"]:
        raise ValueError("Home and away teams must have different ids")
    if not str(home.get("name") or "").strip() or not str(away.get("name") or "").strip():
        raise ValueError("Both teams must have a name")
    team_by_id = {home["id"]: ("home", home), away["id"]: ("away", away)}

    players: list[dict] = []
    player_by_id: dict[str, dict] = {}
    jersey_owners: dict[tuple[str, str], str] = {}
    for index, source in enumerate(request.players):
        player = source.model_dump()
        player_id = _manual_identifier(player["id"], f"players[{index}].id")
        if player_id in player_by_id:
            raise ValueError(f"Duplicate player id: {player_id}")
        if not str(player.get("name") or "").strip():
            raise ValueError(f"players[{index}].name must not be empty")
        team_id = _manual_identifier(
            player.get("team_id") or "",
            f"players[{index}].team_id",
        )
        if team_id not in team_by_id:
            raise ValueError(f"Player {player_id} references an unknown team")
        canonical_team_name = team_by_id[team_id][1]["name"]
        supplied_team_name = str(player.get("team_name") or "").strip()
        if supplied_team_name and supplied_team_name != canonical_team_name:
            raise ValueError(f"Player {player_id} has a team_name that disagrees with teams")
        number = str(player.get("number") or "").strip() or None
        if number is not None:
            owner_key = (team_id, number)
            if owner_key in jersey_owners:
                raise ValueError(
                    f"Team {team_id} has duplicate jersey number {number}"
                )
            jersey_owners[owner_key] = player_id
        player.update(
            {
                "id": player_id,
                "team_id": team_id,
                "team_name": canonical_team_name,
                "number": number,
                "lineup_order": (
                    int(player["lineup_order"])
                    if player.get("lineup_order") is not None
                    else index
                ),
            }
        )
        players.append(player)
        player_by_id[player_id] = player

    lineup: list[dict] = []
    if request.lineup:
        lineup_ids: set[str] = set()
        lineup_players: set[str] = set()
        lineup_orders: set[int] = set()
        for index, source in enumerate(request.lineup):
            entry = source.model_dump()
            entry_id = _manual_identifier(entry["id"], f"lineup[{index}].id")
            player_id = _manual_identifier(
                entry["player_id"],
                f"lineup[{index}].player_id",
            )
            if entry_id in lineup_ids:
                raise ValueError(f"Duplicate lineup id: {entry_id}")
            if player_id in lineup_players:
                raise ValueError(f"Player {player_id} appears twice in the lineup")
            player = player_by_id.get(player_id)
            if player is None:
                raise ValueError(f"Lineup references unknown player {player_id}")
            order = int(entry["order"])
            if order in lineup_orders:
                raise ValueError(f"Duplicate lineup order: {order}")
            team_id = str(entry.get("team_id") or player["team_id"])
            if team_id != player["team_id"]:
                raise ValueError(f"Lineup team disagrees for player {player_id}")
            expected_side, team = team_by_id[team_id]
            if entry.get("side") not in {"unknown", expected_side}:
                raise ValueError(f"Lineup side disagrees for player {player_id}")
            entry.update(
                {
                    "id": entry_id,
                    "player_id": player_id,
                    "player_name": player["name"],
                    "team_id": team_id,
                    "team_name": team["name"],
                    "side": expected_side,
                    "number": player.get("number"),
                    "order": order,
                }
            )
            player["lineup_role"] = entry["role"]
            player["lineup_order"] = order
            lineup_ids.add(entry_id)
            lineup_players.add(player_id)
            lineup_orders.add(order)
            lineup.append(entry)
    else:
        for player in players:
            side, team = team_by_id[player["team_id"]]
            lineup.append(
                {
                    "id": f"manual-lineup-{player['id']}",
                    "player_id": player["id"],
                    "player_name": player["name"],
                    "team_id": player["team_id"],
                    "team_name": team["name"],
                    "side": side,
                    "position": player.get("position"),
                    "number": player.get("number"),
                    "role": player.get("lineup_role") or "unknown",
                    "order": int(player["lineup_order"]),
                }
            )

    timeline: list[dict] = []
    timeline_ids: set[str] = set()
    for index, source in enumerate(request.timeline):
        item = source.model_dump()
        item_id = _manual_identifier(item["id"], f"timeline[{index}].id")
        if item_id in timeline_ids:
            raise ValueError(f"Duplicate timeline id: {item_id}")
        primary = player_by_id.get(str(item.get("player_id") or ""))
        secondary = player_by_id.get(str(item.get("secondary_player_id") or ""))
        if item.get("player_id") and primary is None:
            raise ValueError(f"Timeline references unknown player {item['player_id']}")
        if item.get("secondary_player_id") and secondary is None:
            raise ValueError(
                f"Timeline references unknown player {item['secondary_player_id']}"
            )
        inferred_team_id = (
            primary.get("team_id") if primary else secondary.get("team_id") if secondary else None
        )
        team_id = str(item.get("team_id") or inferred_team_id or "") or None
        if team_id is not None and team_id not in team_by_id:
            raise ValueError(f"Timeline event {item_id} references an unknown team")
        item.update(
            {
                "id": item_id,
                "player_name": primary["name"] if primary else item.get("player_name"),
                "secondary_player_name": (
                    secondary["name"] if secondary else item.get("secondary_player_name")
                ),
                "team_id": team_id,
                "team_name": team_by_id[team_id][1]["name"] if team_id else None,
            }
        )
        timeline_ids.add(item_id)
        timeline.append(item)

    substitutions: list[dict] = []
    substitution_ids: set[str] = set()
    for index, source in enumerate(request.substitutions):
        item = source.model_dump()
        item_id = _manual_identifier(item["id"], f"substitutions[{index}].id")
        if item_id in substitution_ids:
            raise ValueError(f"Duplicate substitution id: {item_id}")
        out_id = _manual_identifier(
            item.get("player_out_id") or "",
            f"substitutions[{index}].player_out_id",
        )
        in_id = _manual_identifier(
            item.get("player_in_id") or "",
            f"substitutions[{index}].player_in_id",
        )
        outgoing = player_by_id.get(out_id)
        incoming = player_by_id.get(in_id)
        if outgoing is None or incoming is None:
            raise ValueError(f"Substitution {item_id} references an unknown player")
        if out_id == in_id or outgoing["team_id"] != incoming["team_id"]:
            raise ValueError(f"Substitution {item_id} must exchange two players on one team")
        team_id = str(item.get("team_id") or outgoing["team_id"])
        if team_id != outgoing["team_id"]:
            raise ValueError(f"Substitution {item_id} has a conflicting team")
        item.update(
            {
                "id": item_id,
                "team_id": team_id,
                "team_name": team_by_id[team_id][1]["name"],
                "player_out_id": out_id,
                "player_out_name": outgoing["name"],
                "player_in_id": in_id,
                "player_in_name": incoming["name"],
            }
        )
        substitution_ids.add(item_id)
        substitutions.append(item)

    imported_at = datetime.now(UTC).isoformat()
    event = ExternalEvent.model_validate(
        {
            **request.event.model_dump(),
            "id": event_id,
            "home": home,
            "away": away,
        }
    )
    bundle = EventBundle.model_validate(
        {
            "source": "manual",
            "event": event,
            "players": players,
            "lineup": lineup,
            "timeline": timeline,
            "substitutions": substitutions,
            "fetched_at": imported_at,
            "warnings": [],
        }
    )
    quality = _roster_quality_payload(bundle)
    if not quality["automaticIdentityEligible"]:
        bundle.warnings.append(
            "The manually imported roster is incomplete for automatic identity; available players can still be bound manually."
        )
    bundle.roster_quality = ExternalRosterQuality(
        status=quality["status"],
        player_count=quality["playerCount"],
        home_player_count=quality["homePlayerCount"],
        away_player_count=quality["awayPlayerCount"],
        automatic_identity_eligible=quality["automaticIdentityEligible"],
        manual_identity_eligible=quality["manualIdentityEligible"],
        reasons=quality["reasons"],
    )
    supplied_provenance = request.provenance
    captured_at = supplied_provenance.captured_at if supplied_provenance else None
    if captured_at is not None and captured_at.tzinfo is None:
        captured_at = captured_at.replace(tzinfo=UTC)
    provenance = {
        "kind": "manual-json",
        "importedAt": imported_at,
        "capturedAt": captured_at.astimezone(UTC).isoformat() if captured_at else None,
        "label": supplied_provenance.label if supplied_provenance else None,
        "reference": supplied_provenance.reference if supplied_provenance else None,
        "notes": supplied_provenance.notes if supplied_provenance else None,
    }
    return bundle, provenance


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "replay-studio-api",
        "provider": sports_provider.default_provider,
        "match_data": sports_provider.descriptors(),
        "video_pipeline": "ffmpeg",
        "calibration_worker": calibration_worker_readiness(),
        "identity_worker": identity_worker_readiness(),
        "jersey_ocr_worker": jersey_ocr_worker_readiness(),
        "ball_worker": ball_worker_readiness(),
    }


@app.post("/api/videos", response_model=VideoAsset, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
):
    suffix = Path(file.filename or "clip.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(status_code=415, detail="Supported formats: MP4, MOV, MKV, WebM, M4V")
    asset_id = f"asset-{uuid4().hex[:12]}"
    directory = asset_directory(asset_id)
    directory.mkdir(parents=True, exist_ok=False)
    stored_name = f"source{suffix}"
    destination = directory / stored_name
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_video_bytes:
                    raise HTTPException(status_code=413, detail="Video is larger than the 250 MB upload limit")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        directory.rmdir()
        raise
    finally:
        await file.close()

    asset = video_store.create(
        id=asset_id,
        filename=stored_name,
        original_name=Path(file.filename or "clip.mp4").name[:240],
        content_type=(file.content_type or "application/octet-stream")[:120],
        status="queued",
        stage="Waiting for FFmpeg",
        progress=2,
    )
    background_tasks.add_task(process_video_asset, asset_id, title)
    return asset


@app.get("/api/videos", response_model=list[VideoAsset])
def list_videos():
    return video_store.list()


@app.get("/api/videos/{asset_id}", response_model=VideoAsset)
def get_video(asset_id: str):
    asset = video_store.get(asset_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="Video asset not found")
    return asset


@app.get("/api/videos/{asset_id}/media")
def video_media(asset_id: str):
    if video_store.get(asset_id) is None:
        raise HTTPException(status_code=404, detail="Video asset not found")
    path = asset_directory(asset_id) / "proxy.mp4"
    if not path.exists():
        raise HTTPException(status_code=409, detail="Browser proxy is not ready")
    return FileResponse(path, media_type="video/mp4", filename=f"{asset_id}.mp4")


@app.get("/api/videos/{asset_id}/poster")
def video_poster(asset_id: str):
    if video_store.get(asset_id) is None:
        raise HTTPException(status_code=404, detail="Video asset not found")
    path = asset_directory(asset_id) / "poster.jpg"
    if not path.exists():
        raise HTTPException(status_code=409, detail="Poster is not ready")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/videos/{asset_id}/segments/{segment_id}/scene", response_model=SceneDocument, status_code=201)
def create_segment_scene(asset_id: str, segment_id: str):
    asset = video_store.get(asset_id)
    if asset is None or not asset.get("scene_id"):
        raise HTTPException(status_code=404, detail="Processed video asset not found")
    parent = scene_store.get(asset["scene_id"])
    if parent is None:
        raise HTTPException(status_code=404, detail="Parent video scene not found")
    video = parent.get("payload", {}).get("videoAsset") or {}
    segment = next((item for item in video.get("segments", []) if item.get("id") == segment_id), None)
    if segment is None:
        raise HTTPException(status_code=404, detail="Video segment not found")

    scene = materialize_segment_scene(parent, segment)
    scene_store.put(parent)
    return scene


@app.post("/api/videos/{asset_id}/segment-layout/propose", response_model=SceneDocument)
def propose_video_segment_layout(asset_id: str):
    asset = video_store.get(asset_id)
    if asset is None or not asset.get("scene_id"):
        raise HTTPException(status_code=404, detail="Processed video asset not found")
    parent = scene_store.get(asset["scene_id"])
    if parent is None:
        raise HTTPException(status_code=404, detail="Parent video scene not found")
    video = parent.get("payload", {}).get("videoAsset") or {}
    segments = video.get("segments") or []
    if not segments:
        raise HTTPException(status_code=409, detail="No continuous shots were detected")
    source = asset_directory(asset_id) / "proxy.mp4"
    if not source.exists():
        source = asset_directory(asset_id) / asset["filename"]
    video["segmentLayout"] = propose_segment_layout(source, segments, float(parent["duration"]))
    return scene_store.put(parent)


@app.post("/api/videos/{asset_id}/multi-pass", response_model=SceneDocument, status_code=202)
def create_video_multi_pass(asset_id: str, request: MultiPassRequest, background_tasks: BackgroundTasks):
    asset = video_store.get(asset_id)
    if asset is None or not asset.get("scene_id"):
        raise HTTPException(status_code=404, detail="Processed video asset not found")
    parent = scene_store.get(asset["scene_id"])
    if parent is None:
        raise HTTPException(status_code=404, detail="Parent video scene not found")
    requested_ids = list(dict.fromkeys(request.segment_ids))
    if len(requested_ids) < 2:
        raise HTTPException(status_code=422, detail="Choose at least two different camera angles")
    segment_map = {
        item["id"]: item
        for item in (parent.get("payload", {}).get("videoAsset", {}).get("segments") or [])
    }
    missing = [segment_id for segment_id in requested_ids if segment_id not in segment_map]
    if missing:
        raise HTTPException(status_code=404, detail=f"Video segments not found: {', '.join(missing)}")
    try:
        scene = create_multi_pass_scene(
            parent,
            [segment_map[item] for item in requested_ids],
            request.title,
            request.manual_alignment_anchors,
        )
    except MultiPassError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    background_tasks.add_task(analyze_multi_pass_by_id, scene["id"])
    return scene


@app.post("/api/scenes/{scene_id}/reconstruct", response_model=SceneDocument, status_code=202)
def reconstruct_video_scene(
    scene_id: str,
    background_tasks: BackgroundTasks,
    request: ReconstructionRequest | None = None,
):
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if video.get("multiPass"):
        raise HTTPException(
            status_code=409,
            detail=(
                "This is a multi-pass composite; rerun multi-angle analysis "
                "instead of single-pass reconstruction"
            ),
        )
    if not video.get("selectedSegmentId"):
        raise HTTPException(status_code=409, detail="Choose a continuous shot scene before reconstruction")
    if (video.get("reconstruction") or {}).get("status") in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Reconstruction is already running")
    try:
        queued = queue_reconstruction(
            scene,
            request.model if request else None,
            ball_backend=request.ball_backend if request else None,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail="The scene changed while reconstruction was being queued; retry with the latest scene.",
        ) from exc
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    background_tasks.add_task(
        reconstruct_scene_by_id,
        scene_id,
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    return queued


@app.put("/api/scenes/{scene_id}/ball-trajectory", response_model=SceneDocument)
def update_scene_ball_trajectory(
    scene_id: str,
    request: BallTrajectoryRequest,
) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before editing the ball trajectory",
        )
    keyframes = (
        None
        if request.keyframes is None
        else [item.model_dump(exclude_none=True) for item in request.keyframes]
    )
    try:
        return set_scene_ball_trajectory(scene, request.mode, keyframes)
    except ReconstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _ensure_player_actions_editable(scene: dict) -> None:
    reconstruction = (
        scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before editing player actions",
        )


@app.post("/api/scenes/{scene_id}/player-actions", response_model=SceneDocument)
def save_scene_player_action(
    scene_id: str,
    request: PlayerActionUpsertRequest,
) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    _ensure_player_actions_editable(scene)
    try:
        upsert_player_action(scene, request.model_dump(), persist=True)
    except PlayerActionError as exc:
        status_code = 404 if str(exc) == "The canonical person no longer exists" else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return scene


@app.delete(
    "/api/scenes/{scene_id}/player-actions/{action_id}",
    response_model=SceneDocument,
)
def remove_scene_player_action(scene_id: str, action_id: str) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    _ensure_player_actions_editable(scene)
    try:
        delete_player_action(scene, action_id, persist=True)
    except PlayerActionError as exc:
        status_code = 404 if str(exc) == "Player action not found" else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return scene


@app.post("/api/scenes/{scene_id}/analyze-frame")
def analyze_video_scene_frame(scene_id: str, request: FrameAnalysisRequest) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video.get("selectedSegmentId"):
        raise HTTPException(status_code=409, detail="Choose a continuous shot scene before analyzing a frame")
    if request.scene_time > float(scene.get("duration") or 0):
        raise HTTPException(status_code=422, detail="Frame time is outside this scene")
    try:
        return analyze_scene_frame(scene, request.scene_time)
    except ReconstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/scenes/{scene_id}/frame-annotations")
def save_video_scene_frame_annotation(
    scene_id: str,
    request: FramePersonAnnotationRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video.get("selectedSegmentId"):
        raise HTTPException(status_code=409, detail="Choose a continuous shot scene before labeling a frame")
    if (video.get("reconstruction") or {}).get("status") in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Wait for reconstruction to finish before labeling a frame")
    if request.scene_time > float(scene.get("duration") or 0):
        raise HTTPException(status_code=422, detail="Frame time is outside this scene")
    expected_scene_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        annotation = upsert_frame_person_annotation(
            scene,
            request.model_dump(),
            persist=False,
        )
        analysis = analyze_scene_frame(scene, float(annotation["sceneTime"]))
        queued = queue_reconstruction(
            scene,
            expected_scene_fingerprint=expected_scene_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail="The scene changed while the correction was being saved; retry on the latest frame.",
        ) from exc
    except ReconstructionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    background_tasks.add_task(
        reconstruct_scene_by_id,
        scene_id,
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    analysis["reconstruction"] = {
        key: reconstruction.get(key)
        for key in ("status", "model", "runId", "runRevision", "inputFingerprint")
    }
    return analysis


@app.delete("/api/scenes/{scene_id}/frame-annotations/{annotation_id}")
def remove_video_scene_frame_annotation(
    scene_id: str,
    annotation_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if (video.get("reconstruction") or {}).get("status") in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Wait for reconstruction to finish before editing labels")
    expected_scene_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        annotation = delete_frame_person_annotation(
            scene,
            annotation_id,
            persist=False,
        )
        analysis = analyze_scene_frame(scene, float(annotation["sceneTime"]))
        queued = queue_reconstruction(
            scene,
            expected_scene_fingerprint=expected_scene_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail="The scene changed while the correction was being removed; retry on the latest frame.",
        ) from exc
    except ReconstructionError as exc:
        status_code = 404 if str(exc) == "Frame annotation was not found" else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    background_tasks.add_task(
        reconstruct_scene_by_id,
        scene_id,
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    analysis["reconstruction"] = {
        key: reconstruction.get(key)
        for key in ("status", "model", "runId", "runRevision", "inputFingerprint")
    }
    return analysis


@app.put(
    "/api/scenes/{scene_id}/canonical-people/{canonical_person_id}/roster-binding",
    response_model=SceneDocument,
    status_code=202,
)
def update_canonical_roster_binding(
    scene_id: str,
    canonical_person_id: str,
    request: CanonicalRosterBindingRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """Persist a roster decision and queue its rebuild as one CAS write.

    The correction is anchored to a detector observation already stored on the
    canonical person.  Consequently an off-screen identity can be edited and
    an unavailable detector/ReID/OCR worker cannot make the decision vanish.
    """

    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video.get("selectedSegmentId"):
        raise HTTPException(
            status_code=409,
            detail="Choose a continuous shot scene before binding its roster",
        )
    if (video.get("reconstruction") or {}).get("status") in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before editing roster identities",
        )

    expected_scene_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        set_canonical_roster_binding(
            scene,
            canonical_person_id,
            request.external_player_id,
            persist=False,
        )
        queued = queue_reconstruction(
            scene,
            expected_scene_fingerprint=expected_scene_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail="The scene changed while the roster binding was being saved; retry.",
        ) from exc
    except ReconstructionError as exc:
        status_code = 404 if str(exc) == "The canonical person no longer exists" else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    background_tasks.add_task(
        reconstruct_scene_by_id,
        scene_id,
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    return queued


@app.delete(
    "/api/scenes/{scene_id}/canonical-people/{canonical_person_id}/roster-binding",
    response_model=SceneDocument,
    status_code=202,
)
def clear_canonical_roster_binding_decision(
    scene_id: str,
    canonical_person_id: str,
    background_tasks: BackgroundTasks,
) -> dict:
    """Clear an existing Unbind tombstone and queue a guarded rebuild."""

    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video.get("selectedSegmentId"):
        raise HTTPException(
            status_code=409,
            detail="Choose a continuous shot scene before clearing its roster decision",
        )
    if (video.get("reconstruction") or {}).get("status") in {
        "queued",
        "processing",
    }:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before editing roster identities",
        )

    expected_scene_fingerprint = reconstruction_input_fingerprint(scene)
    try:
        clear_canonical_roster_binding(
            scene,
            canonical_person_id,
            persist=False,
        )
        queued = queue_reconstruction(
            scene,
            expected_scene_fingerprint=expected_scene_fingerprint,
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail="The scene changed while the roster decision was being cleared; retry.",
        ) from exc
    except ReconstructionError as exc:
        status_code = (
            404
            if str(exc)
            in {
                "The canonical person no longer exists",
                "This canonical person has no roster decision to clear",
            }
            else 422
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    background_tasks.add_task(
        reconstruct_scene_by_id,
        scene_id,
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    return queued


@app.post("/api/scenes/{scene_id}/compare-models")
def compare_video_scene_models(scene_id: str) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video.get("selectedSegmentId"):
        raise HTTPException(status_code=409, detail="Choose a continuous shot scene before comparing models")
    reconstruction = video.get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Wait for reconstruction to finish before comparing models")
    try:
        report = compare_scene_models(scene)
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    reconstruction["modelComparison"] = report
    video["reconstruction"] = reconstruction
    scene_store.put(scene)
    return report


def _calibration_scene(scene_id: str) -> dict:
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    video = scene.get("payload", {}).get("videoAsset") or {}
    if not video.get("selectedSegmentId"):
        raise HTTPException(status_code=409, detail="Choose a continuous shot scene before calibration")
    return scene


@app.post("/api/scenes/{scene_id}/pitch-calibration/auto")
def auto_pitch_calibration(scene_id: str, request: PitchCalibrationDraftRequest) -> dict:
    scene = _calibration_scene(scene_id)
    if request.scene_time > float(scene.get("duration") or 0):
        raise HTTPException(status_code=422, detail="Frame time is outside this scene")
    try:
        return propose_scene_pitch_calibration(scene, request.scene_time, request.preset)
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/scenes/{scene_id}/pitch-calibration/preview")
def preview_pitch_calibration(scene_id: str, request: PitchCalibrationPreviewRequest) -> dict:
    scene = _calibration_scene(scene_id)
    if request.scene_time > float(scene.get("duration") or 0):
        raise HTTPException(status_code=422, detail="Frame time is outside this scene")
    try:
        return preview_scene_pitch_calibration(
            scene,
            request.scene_time,
            request.preset,
            [anchor.model_dump() for anchor in request.anchors],
        )
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/scenes/{scene_id}/pitch-calibration/apply", response_model=SceneDocument, status_code=202)
def apply_pitch_calibration(
    scene_id: str,
    request: PitchCalibrationPreviewRequest,
    background_tasks: BackgroundTasks,
):
    scene = _calibration_scene(scene_id)
    if request.scene_time > float(scene.get("duration") or 0):
        raise HTTPException(status_code=422, detail="Frame time is outside this scene")
    reconstruction = scene.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    if reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(status_code=409, detail="Reconstruction is already running")
    try:
        queued = apply_scene_pitch_calibration(
            scene,
            request.scene_time,
            request.preset,
            [anchor.model_dump() for anchor in request.anchors],
        )
    except StaleReconstructionRun as exc:
        raise HTTPException(
            status_code=409,
            detail="The scene changed while calibration was being applied; reopen the latest frame and retry.",
        ) from exc
    except (ReconstructionError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    reconstruction = queued["payload"]["videoAsset"]["reconstruction"]
    background_tasks.add_task(
        reconstruct_scene_by_id,
        scene_id,
        reconstruction["runId"],
        reconstruction["inputFingerprint"],
    )
    return queued


@app.post("/api/scenes/{scene_id}/pitch-side", response_model=SceneDocument)
def change_scene_pitch_side(scene_id: str, request: PitchSideRequest) -> dict:
    scene = _calibration_scene(scene_id)
    try:
        return set_scene_pitch_side(scene, request.side)
    except ReconstructionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/catalog/providers")
def catalog_providers() -> dict:
    """List server-side match providers without exposing credentials or URLs."""

    return sports_provider.descriptors()


@app.get("/api/catalog/events", response_model=list[ExternalEvent])
async def catalog_events(
    date: str = Query(default_factory=lambda: date_type.today().isoformat()),
    provider: str | None = Query(default=None, min_length=1, max_length=80),
):
    try:
        return await sports_provider.events_by_date_for(provider, date)
    except MatchDataError as exc:
        raise _match_data_http_exception(exc) from exc


@app.get("/api/catalog/events/search", response_model=list[ExternalEvent])
async def search_catalog_events(
    q: str = Query(min_length=3, max_length=120),
    provider: str | None = Query(default=None, min_length=1, max_length=80),
):
    try:
        return await sports_provider.search_events_for(provider, q)
    except MatchDataError as exc:
        raise _match_data_http_exception(exc) from exc


@app.get("/api/catalog/events/{event_id}", response_model=EventBundle)
async def catalog_event(
    event_id: str,
    provider: str | None = Query(default=None, min_length=1, max_length=80),
):
    try:
        return await sports_provider.event_bundle_for(provider, event_id)
    except MatchDataError as exc:
        raise _match_data_http_exception(exc) from exc


@app.get("/api/scenes", response_model=list[SceneSummary])
def list_scenes():
    return scene_store.list()


@app.get("/api/scenes/{scene_id}", response_model=SceneDocument)
def get_scene(scene_id: str):
    scene = scene_store.get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene


def _legacy_binding_rank(scene: dict, root_scene_id: str, binding: dict) -> tuple:
    """Prefer intentional, complete snapshots over provider-capped remnants."""

    quality = binding.get("rosterQuality") or {}
    provenance = binding.get("provenance") or {}
    is_canonical_root = (
        scene.get("id") == root_scene_id
        and binding.get("scope") == "project"
        and binding.get("projectSceneId") == root_scene_id
        and binding.get("inherited") is False
    )
    is_manual = binding.get("source") == "manual" or provenance.get("kind") == "manual-json"
    return (
        int(is_canonical_root),
        int(is_manual),
        int(binding.get("schemaVersion") or 0),
        int(bool(quality.get("automaticIdentityEligible"))),
        len(binding.get("players") or []),
        len(binding.get("lineup") or []),
        len(binding.get("timeline") or []),
        len(binding.get("substitutions") or []),
        int(bool(binding.get("fetchedAt"))),
    )


def _migrate_legacy_project_match_bindings() -> list[str]:
    """Promote the best legacy child snapshot to every scene in its project.

    Older builds stored Match Data independently.  Migration runs before
    reconstruction recovery, prefers an explicit full manual import (the
    common 52-player project roster) over capped provider snapshots, and
    never queues reconstruction: startup has no explicitly selected/current
    scene, so it may repair project metadata but must not manufacture hidden
    background work. It also repairs multi-pass presentation residue produced
    when an old API monitor raced this migration. Multi-pass composites remain
    explicit multi-angle jobs. A project with an active lease or an ambiguous
    equal-ranked conflict is left untouched.
    """

    migrated: list[str] = []
    visited_roots: set[str] = set()
    for summary in scene_store.list():
        root, scenes = scene_store.project_scenes(summary["id"])
        if root is None or root["id"] in visited_roots:
            continue
        visited_roots.add(root["id"])
        candidates = [
            (scene, scene.get("payload", {}).get("matchBinding") or {})
            for scene in scenes
            if scene.get("payload", {}).get("matchBinding")
        ]
        if not candidates:
            continue
        ranked = [
            (_legacy_binding_rank(scene, root["id"], binding), scene, binding)
            for scene, binding in candidates
        ]
        best_rank = max(item[0] for item in ranked)
        winners = [item for item in ranked if item[0] == best_rank]
        winner_signatures = {
            json.dumps(
                semantic_match_binding(item[2]),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            for item in winners
        }
        if len(winner_signatures) != 1:
            # Equal evidence for different matches is not safe to guess.
            continue
        _rank, source_scene, source_binding = winners[0]
        canonical_binding = semantic_match_binding(source_binding)
        if not canonical_binding:
            continue

        source_teams = source_scene.get("payload", {}).get("teams") or []
        changed_semantic_ids: set[str] = set()
        needs_write = False
        for scene in scenes:
            current_binding = scene.get("payload", {}).get("matchBinding")
            desired_binding = project_match_binding(
                canonical_binding,
                root["id"],
                inherited=scene["id"] != root["id"],
            )
            if current_binding != desired_binding:
                needs_write = True
            if semantic_match_binding(current_binding) != canonical_binding:
                changed_semantic_ids.add(scene["id"])
            scene["payload"]["matchBinding"] = desired_binding
            teams = scene["payload"].get("teams") or []
            for index, team in enumerate(teams[:2]):
                if index >= len(source_teams):
                    continue
                old = (team.get("name"), team.get("externalTeamId"))
                source_team = source_teams[index]
                team["name"] = source_team.get("name", team.get("name"))
                team["externalTeamId"] = source_team.get("externalTeamId")
                if old != (team.get("name"), team.get("externalTeamId")):
                    needs_write = True

            video = scene.get("payload", {}).get("videoAsset") or {}
            reconstruction = video.get("reconstruction") or {}
            fingerprint_failure_is_compatible = (
                reconstruction.get("status") == "failed"
                and reconstruction.get("error")
                == RECONSTRUCTION_INPUT_CHANGED_ERROR
                and reconstruction.get("inputFingerprint")
                == reconstruction_input_fingerprint(scene)
            )
            if is_multi_pass_scene(scene):
                if scene["id"] in changed_semantic_ids:
                    mark_multi_pass_match_binding_stale(scene)
                    needs_write = True
                multi_pass = video.get("multiPass") or {}
                progress = reconstruction.get("progress") or {}
                failed_progress_residue = (
                    reconstruction.get("status") == "ready"
                    and reconstruction.get("error") is None
                    and progress.get("phase") == "failed"
                    and progress.get("detail")
                    == RECONSTRUCTION_INPUT_CHANGED_ERROR
                )
                if (
                    fingerprint_failure_is_compatible
                    and multi_pass.get("status") == "ready"
                    and scene.get("payload", {}).get("tracks")
                ):
                    # The composite was already valid before an old monitor
                    # incorrectly treated it as a single camera shot. Restore
                    # that last-good result, but require a later multi-angle
                    # run to refresh identity fusion for the new roster.
                    previous_result = reconstruction.get("previousResult") or {}
                    reconstruction.update(
                        {
                            "status": "ready",
                            "processingStatus": "completed",
                            "qualityVerdict": "review",
                            "error": None,
                            "completedAt": previous_result.get("completedAt")
                            or reconstruction.get("completedAt"),
                        }
                    )
                    video["processingState"] = "multi-pass-ready"
                    mark_multi_pass_match_binding_stale(scene)
                    needs_write = True
                if (
                    (fingerprint_failure_is_compatible or failed_progress_residue)
                    and multi_pass.get("status") == "ready"
                    and scene.get("payload", {}).get("tracks")
                ):
                    reconstruction["progress"] = {
                        **progress,
                        "phase": "review",
                        "label": "Multi-angle result needs refresh",
                        "detail": MULTI_PASS_MATCH_BINDING_STALE_WARNING,
                        "phasePercent": 100,
                        "overallPercent": 100,
                        "etaSeconds": 0.0,
                    }
                    mark_multi_pass_match_binding_stale(scene)
                    needs_write = True
                continue
        if not needs_write:
            continue
        try:
            scene_store.put_many(scenes)
        except SceneRevisionConflict:
            # Another API/worker process owns the project; a later restart or
            # explicit match mutation will retry without a partial migration.
            continue
        migrated.append(root["id"])
    return migrated


def _editable_match_binding_project(
    scene_id: str,
) -> tuple[dict, dict, list[dict], dict[str, str]]:
    root, scenes = scene_store.project_scenes(scene_id)
    if root is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    requested = next(scene for scene in scenes if scene["id"] == scene_id)
    for scene in scenes:
        reconstruction = (
            scene.get("payload", {})
            .get("videoAsset", {})
            .get("reconstruction")
            or {}
        )
        if reconstruction.get("status") in {"queued", "processing"}:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Wait for reconstruction to finish in every project scene "
                    "before changing the bound match"
                ),
            )
    fingerprints = {
        scene["id"]: reconstruction_input_fingerprint(scene) for scene in scenes
    }
    return requested, root, scenes, fingerprints


def _validate_match_binding_snapshot(scene: dict, snapshot: dict) -> None:
    """Reject a project roster that would orphan durable scene decisions."""

    current_reconstruction = (
        scene.get("payload", {})
        .get("videoAsset", {})
        .get("reconstruction")
        or {}
    )

    players_by_id: dict[str, list[dict]] = {}
    for player in snapshot.get("players") or []:
        identifier = str(player.get("id") or "").strip()
        if identifier:
            players_by_id.setdefault(identifier, []).append(player)
    active_people = [
        person
        for person in scene.get("payload", {}).get("canonicalPeople") or []
        if str(person.get("externalPlayerId") or "").strip()
    ]
    # A failed/legacy reconstruction can leave the durable roster correction
    # available while ``canonicalPeople`` still reflects the last-good output
    # and does not expose its optimistic binding.  Do not let changing the
    # match silently orphan that persisted decision.  Published canonical
    # ownership wins when present (a split can intentionally move the anchor
    # before the correction is re-keyed); corrections are only a fallback for
    # external IDs absent from the published people list.
    published_external_ids = {
        str(person.get("externalPlayerId") or "").strip()
        for person in active_people
    }
    fallback_corrections: dict[str, list[dict]] = {}
    for correction in current_reconstruction.get("frameAnnotations") or []:
        external_id = str(correction.get("externalPlayerId") or "").strip()
        if (
            correction.get("correctionKind")
            != CANONICAL_ROSTER_BINDING_CORRECTION
            or correction.get("rosterBindingState") != "bound"
            or not external_id
            or external_id in published_external_ids
        ):
            continue
        fallback_corrections.setdefault(external_id, []).append(correction)
    for external_id, corrections in fallback_corrections.items():
        if len(corrections) != 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Unbind canonical roster player {external_id} before changing "
                    "the match because multiple durable bindings claim that player"
                ),
            )
        correction = corrections[0]
        kind = str(correction.get("kind") or "")
        active_people.append(
            {
                "canonicalPersonId": correction.get("canonicalPersonId"),
                "externalPlayerId": external_id,
                "teamId": (
                    "home"
                    if kind.startswith("home-")
                    else "away"
                    if kind.startswith("away-")
                    else None
                ),
            }
        )
    for person in active_people:
        external_id = str(person.get("externalPlayerId") or "").strip()
        candidates = players_by_id.get(external_id) or []
        local_team = str(person.get("teamId") or "").strip()
        expected_team = (snapshot.get("teams") or {}).get(local_team) or {}
        expected_team_id = str(
            (
                expected_team.get("id")
                if isinstance(expected_team, dict)
                else expected_team
            )
            or ""
        ).strip()
        roster_team_ids = {
            str(item.get("team_id") or item.get("teamId") or "").strip()
            for item in candidates
        }
        if len(candidates) != 1 or (
            expected_team_id
            and any(roster_team_ids)
            and roster_team_ids != {expected_team_id}
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Unbind canonical roster player {external_id} before changing "
                    "to a match whose roster does not contain the same player and team"
                ),
            )


def _persist_match_binding_bundle(
    scene_id: str,
    root: dict,
    scenes: list[dict],
    expected_scene_fingerprints: dict[str, str],
    bundle: EventBundle,
    background_tasks: BackgroundTasks,
    *,
    provenance: dict | None = None,
) -> dict:
    """Persist project metadata and rebuild only the explicitly requested scene.

    Match data is project-scoped, so every member receives the same canonical
    snapshot. Reconstruction is scene-scoped: a bind/import/refresh request may
    queue the URL's ``scene_id`` only when it is a single-pass shot. Requesting
    the root video therefore updates metadata without starting every shot, and
    multi-pass composites remain available through their explicit analysis
    endpoint.
    """

    snapshot = _match_binding_snapshot(bundle, provenance=provenance)
    for scene in scenes:
        _validate_match_binding_snapshot(scene, snapshot)

    queued_scene_ids: list[str] = []
    for scene in scenes:
        semantic_binding_changed = (
            semantic_match_binding(
                scene.get("payload", {}).get("matchBinding")
            )
            != semantic_match_binding(snapshot)
        )
        inherited = scene["id"] != root["id"]
        scene["payload"]["matchBinding"] = project_match_binding(
            snapshot,
            root["id"],
            inherited=inherited,
        )
        teams = scene["payload"].get("teams") or []
        if len(teams) >= 2:
            teams[0].update(
                {
                    "name": bundle.event.home.name,
                    "externalTeamId": bundle.event.home.id,
                }
            )
            teams[1].update(
                {
                    "name": bundle.event.away.name,
                    "externalTeamId": bundle.event.away.id,
                }
            )
        if is_multi_pass_scene(scene):
            if semantic_binding_changed:
                mark_multi_pass_match_binding_stale(scene)
            continue
        if scene["id"] == scene_id and is_single_pass_reconstruction_scene(scene):
            try:
                queue_reconstruction(
                    scene,
                    expected_scene_fingerprint=expected_scene_fingerprints[scene["id"]],
                    persist=False,
                )
            except ReconstructionError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            queued_scene_ids.append(scene["id"])

    try:
        saved_scenes = scene_store.put_many(scenes)
        saved_by_id = {scene["id"]: scene for scene in saved_scenes}
        for queued_scene_id in queued_scene_ids:
            queued_reconstruction = (
                saved_by_id[queued_scene_id]
                .get("payload", {})
                .get("videoAsset", {})
                .get("reconstruction", {})
            )
            background_tasks.add_task(
                reconstruct_scene_by_id,
                queued_scene_id,
                queued_reconstruction["runId"],
                queued_reconstruction["inputFingerprint"],
            )
    except (SceneRevisionConflict, StaleReconstructionRun) as exc:
        raise HTTPException(
            status_code=409,
            detail=(
                "The project changed while its match data was being saved; "
                "reload and retry"
            ),
        ) from exc
    return {"scene": saved_by_id[scene_id], "bundle": bundle}


@app.post("/api/scenes/{scene_id}/match-binding", response_model=SceneMatchBindingResponse)
async def bind_scene_match(
    scene_id: str,
    request: MatchBindingRequest,
    background_tasks: BackgroundTasks,
):
    _requested, root, scenes, fingerprints = _editable_match_binding_project(scene_id)
    try:
        bundle = await sports_provider.event_bundle_for(
            request.provider, request.event_id
        )
    except MatchDataError as exc:
        raise _match_data_http_exception(exc) from exc
    return _persist_match_binding_bundle(
        scene_id,
        root,
        scenes,
        fingerprints,
        bundle,
        background_tasks,
    )


@app.post(
    "/api/scenes/{scene_id}/match-binding/import",
    response_model=SceneMatchBindingResponse,
)
def import_scene_match_binding(
    scene_id: str,
    request: ManualMatchImportRequest,
    background_tasks: BackgroundTasks,
):
    """Persist a strict user-owned JSON roster through the normal guarded path."""

    _requested, root, scenes, fingerprints = _editable_match_binding_project(scene_id)
    try:
        bundle, provenance = _manual_match_bundle(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _persist_match_binding_bundle(
        scene_id,
        root,
        scenes,
        fingerprints,
        bundle,
        background_tasks,
        provenance=provenance,
    )


@app.post(
    "/api/scenes/{scene_id}/match-binding/refresh",
    response_model=SceneMatchBindingResponse,
)
async def refresh_scene_match_binding(
    scene_id: str,
    background_tasks: BackgroundTasks,
):
    """Upgrade a legacy/partial binding from its persisted event id.

    Refresh intentionally follows the exact bind path, including the running
    reconstruction lock, durable roster-decision validation, input
    fingerprint guard, full-document CAS, and optional queued rebuild.  A
    legacy ``{source,eventId,fetchedAt}`` document is therefore replaced in
    one guarded write rather than incrementally patched in place.
    """

    requested, root, scenes, _fingerprints = _editable_match_binding_project(scene_id)
    binding = root.get("payload", {}).get("matchBinding") or {}
    if not binding and requested["id"] != root["id"]:
        # A refresh explicitly aimed at a legacy child is also an explicit
        # choice of which old binding should become project-canonical.
        binding = requested.get("payload", {}).get("matchBinding") or {}
    if not binding:
        legacy_bindings = [
            scene.get("payload", {}).get("matchBinding") or {}
            for scene in scenes
            if scene.get("payload", {}).get("matchBinding")
        ]
        legacy_keys = {
            (
                str(candidate.get("source") or "thesportsdb"),
                str(candidate.get("eventId") or "").strip(),
            )
            for candidate in legacy_bindings
            if str(candidate.get("eventId") or "").strip()
        }
        if len(legacy_keys) > 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Legacy child scenes contain conflicting match bindings; "
                    "bind or import the intended project match explicitly"
                ),
            )
        if legacy_bindings:
            binding = legacy_bindings[0]
    source = str(binding.get("source") or "thesportsdb").strip()
    if source == "manual":
        raise HTTPException(
            status_code=409,
            detail="Manual match data must be replaced by another JSON import",
        )
    if not sports_provider.has(source):
        raise HTTPException(
            status_code=409,
            detail=f"The saved match-data provider is no longer supported: {source}",
        )
    event_id = str(binding.get("eventId") or "").strip()
    if not event_id:
        raise HTTPException(
            status_code=409,
            detail="This scene has no saved match event id to refresh",
        )
    return await bind_scene_match(
        scene_id,
        MatchBindingRequest(event_id=event_id, provider=source),
        background_tasks,
    )


@app.put("/api/scenes/{scene_id}", response_model=SceneDocument)
def update_scene(scene_id: str, scene: SceneDocument):
    if scene_id != scene.id:
        raise HTTPException(status_code=400, detail="Scene id does not match URL")
    current = scene_store.get(scene_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Scene not found")
    incoming = scene.model_dump()
    current_reconstruction = (
        current.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    incoming_reconstruction = (
        incoming.get("payload", {}).get("videoAsset", {}).get("reconstruction") or {}
    )
    if current_reconstruction.get("status") in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="Wait for reconstruction to finish before saving the scene",
        )
    runtime_fields = ("runId", "runRevision", "inputFingerprint", "status")
    if any(
        current_reconstruction.get(field) != incoming_reconstruction.get(field)
        for field in runtime_fields
    ):
        raise HTTPException(
            status_code=409,
            detail="The scene reconstruction changed; reload before saving",
        )
    if reconstruction_input_fingerprint(current) != reconstruction_input_fingerprint(
        incoming
    ):
        raise HTTPException(
            status_code=409,
            detail="Reconstruction inputs must be edited through their dedicated endpoints",
        )
    return scene_store.put(incoming)


@app.post("/api/scenes", response_model=SceneDocument, status_code=201)
async def create_scene(request: CreateSceneRequest):
    scene_id = f"moment-{uuid4().hex[:8]}"
    title = request.title or "Untitled football moment"
    scene = make_demo_scene(scene_id=scene_id, title=title)
    if request.event_id:
        selected_provider = request.provider or sports_provider.default_provider
        try:
            bundle = await sports_provider.event_bundle_for(
                request.provider, request.event_id
            )
            scene["title"] = request.title or bundle.event.name
            scene["payload"]["matchBinding"] = _match_binding_snapshot(bundle)
            scene["payload"]["teams"][0].update(
                {"name": bundle.event.home.name, "externalTeamId": bundle.event.home.id}
            )
            scene["payload"]["teams"][1].update(
                {"name": bundle.event.away.name, "externalTeamId": bundle.event.away.id}
            )
        except MatchDataError:
            scene["payload"]["matchBinding"] = {
                "source": selected_provider,
                "eventId": request.event_id,
                "fetchedAt": None,
            }
    return scene_store.put(scene)
