from __future__ import annotations

from copy import deepcopy

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, ReconstructionJobRow, SceneRow
from app.sample import make_video_scene
from app.reconstruction_job_queries import ReconstructionJobQueries
from app.scene_repository import SceneRepository


def _store():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, sessions, SceneRepository(sessions)


def _video(asset_id: str, **values) -> dict:
    return {
        "id": asset_id,
        "filename": f"{asset_id}.mp4",
        "selectedSegmentId": values.pop("selectedSegmentId", None),
        **values,
    }


def test_scene_repository_returns_owned_snapshots_without_mutating_callers() -> None:
    engine, _, store = _store()
    first_input = make_video_scene("first", "1-A", 6.0, _video("asset-1"))
    second_input = make_video_scene("second", "1-B", 6.0, _video("asset-1"))

    first_saved = store.put(first_input)
    second_saved = store.put(second_input)

    assert "revision" not in first_input
    assert "revision" not in second_input
    assert first_saved["revision"] == second_saved["revision"] == 1

    first_saved["title"] = "Local mutation"
    assert store.get("first")["title"] == "1-A"

    first_update = store.get("first")
    second_update = store.get("second")
    assert first_update is not None and second_update is not None
    first_update["title"] = "Saved A"
    second_update["title"] = "Saved B"
    batch_saved = store.put_many([first_update, second_update])

    assert first_update["revision"] == second_update["revision"] == 1
    assert [document["revision"] for document in batch_saved] == [2, 2]
    assert store.get("first")["title"] == "Saved A"
    assert store.get("second")["title"] == "Saved B"

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_scene_writes_atomically_maintain_compact_index_metadata() -> None:
    engine, sessions, store = _store()
    root = make_video_scene("root", "Match", 18.5, _video("asset-1"))
    child = make_video_scene(
        "child",
        "1-A",
        6.25,
        _video(
            "asset-1",
            parentSceneId="root",
            selectedSegmentId="shot-01",
        ),
    )
    composite = make_video_scene(
        "composite",
        "1 multi-pass",
        6.25,
        _video(
            "asset-1",
            parentSceneId="root",
            selectedSegmentId="shot-01",
            multiPass={"parentSceneId": "root", "status": "ready"},
        ),
    )

    store.put(root)
    store.put(child)
    store.put(composite)

    with sessions() as session:
        rows = {
            row.id: row
            for row in session.scalars(select(SceneRow).order_by(SceneRow.id)).all()
        }
        assert rows["root"].duration == 18.5
        assert rows["root"].kind == "video"
        assert rows["root"].parent_scene_id is None
        assert rows["child"].duration == 6.25
        assert rows["child"].kind == "segment"
        assert rows["child"].parent_scene_id == "root"
        assert rows["child"].selected_segment_id == "shot-01"
        assert rows["composite"].kind == "multi-pass"

    updated_root = deepcopy(store.get("root"))
    updated_child = deepcopy(store.get("child"))
    assert updated_root is not None
    assert updated_child is not None
    updated_root["duration"] = 20.0
    updated_child["duration"] = 7.0
    updated_child["payload"]["videoAsset"]["selectedSegmentId"] = "shot-02"
    store.put_many([updated_root, updated_child])

    with sessions() as session:
        root_row = session.get(SceneRow, "root")
        child_row = session.get(SceneRow, "child")
        assert root_row is not None
        assert child_row is not None
        assert root_row.duration == 20.0
        assert child_row.duration == 7.0
        assert child_row.selected_segment_id == "shot-02"

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_list_and_segment_lookup_do_not_scan_scene_payload() -> None:
    engine, _, store = _store()
    store.put(make_video_scene("root", "Match", 18.5, _video("asset-1")))
    store.put(
        make_video_scene(
            "child",
            "1-A",
            6.25,
            _video(
                "asset-1",
                parentSceneId="root",
                selectedSegmentId="shot-01",
            ),
        )
    )

    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(" ".join(str(statement).lower().split()))

    event.listen(engine, "before_cursor_execute", capture_statement)
    summaries = store.list()
    list_statements = list(statements)
    statements.clear()
    found = store.find_segment_scene("root", "shot-01")
    lookup_statements = list(statements)
    event.remove(engine, "before_cursor_execute", capture_statement)

    assert {summary["id"] for summary in summaries} == {"root", "child"}
    assert found is not None
    assert found["id"] == "child"

    scene_list_queries = [
        statement for statement in list_statements if " from scenes" in statement
    ]
    assert len(scene_list_queries) == 1
    assert "payload" not in scene_list_queries[0]
    assert "scenes.duration" in scene_list_queries[0]
    assert "scenes.kind" in scene_list_queries[0]

    scene_lookup_queries = [
        statement for statement in lookup_statements if " from scenes" in statement
    ]
    compact_lookup = next(
        statement
        for statement in scene_lookup_queries
        if "scenes.parent_scene_id" in statement
    )
    assert "payload" not in compact_lookup
    assert "scenes.selected_segment_id" in compact_lookup
    assert "scenes.kind" in compact_lookup

    dense_queries = [
        statement for statement in scene_lookup_queries if "scenes.payload" in statement
    ]
    assert len(dense_queries) == 1
    assert "where scenes.id =" in dense_queries[0]

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_batch_reconstruction_statuses_never_select_scene_payload() -> None:
    engine, sessions, store = _store()
    store.put(make_video_scene("scene-a", "1-A", 6.0, _video("asset-1")))
    store.put(make_video_scene("scene-b", "1-B", 6.0, _video("asset-1")))
    with sessions.begin() as session:
        session.add(
            ReconstructionJobRow(
                scene_id="scene-a",
                run_id="run-a",
                input_fingerprint="sha256:a",
                input_revision=1,
                status="processing",
                requested_at=1.0,
                updated_at=2.0,
            )
        )

    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(" ".join(str(statement).lower().split()))

    event.listen(engine, "before_cursor_execute", capture_statement)
    statuses = ReconstructionJobQueries(sessions).statuses(
        ["scene-a", "scene-b", "scene-a"]
    )
    event.remove(engine, "before_cursor_execute", capture_statement)

    assert statuses == {"scene-a": "processing"}
    assert len(statements) == 1
    assert "reconstruction_jobs.scene_id" in statements[0]
    assert "reconstruction_jobs.status" in statements[0]
    assert "scenes" not in statements[0]
    assert "payload" not in statements[0]

    Base.metadata.drop_all(engine)
    engine.dispose()


def test_list_by_ids_is_bounded_and_never_selects_scene_payload() -> None:
    engine, _, store = _store()
    store.put(make_video_scene("scene-a", "1-A", 6.0, _video("asset-1")))
    store.put(make_video_scene("scene-b", "1-B", 6.0, _video("asset-1")))

    statements: list[str] = []

    def capture_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(" ".join(str(statement).lower().split()))

    event.listen(engine, "before_cursor_execute", capture_statement)
    summaries = store.list_by_ids(["scene-b", "scene-b", "missing"])
    event.remove(engine, "before_cursor_execute", capture_statement)

    assert [item["id"] for item in summaries] == ["scene-b"]
    assert len(statements) == 1
    assert "scenes.payload" not in statements[0]
    assert "where scenes.id in" in statements[0]
    assert "scene-a" not in str(summaries)

    Base.metadata.drop_all(engine)
    engine.dispose()
