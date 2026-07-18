from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx

import app.main as main
import app.pipeline_job as pipeline_job
import app.project_resource_access as resource_access
import app.scene_analysis_routes as scene_analysis_routes
from app.pipeline_domain import PipelineJob


def test_compare_models_endpoint_only_enqueues_pipeline_work(monkeypatch) -> None:
    scene = {
        "id": "scene-compare",
        "payload": {
            "videoAsset": {
                "selectedSegmentId": "shot-1",
                "reconstruction": {"status": "ready"},
            }
        },
    }
    calls: list[dict] = []

    class FakeModelComparisonPipeline:
        def enqueue(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                id="model-comparison-run",
                kind="model-comparison",
                status="queued",
            )

    monkeypatch.setattr(
        resource_access,
        "project_scene_or_404",
        lambda project, scene_id: scene,
    )
    monkeypatch.setattr(
        scene_analysis_routes,
        "model_comparison_pipeline",
        FakeModelComparisonPipeline(),
    )

    async def request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/api/projects/project-1/scenes/scene-compare/compare-models"
            )

    response = asyncio.run(request())

    assert response.status_code == 202
    assert response.json() == {
        "runId": "model-comparison-run",
        "sceneId": "scene-compare",
        "kind": "model-comparison",
        "status": "queued",
    }
    assert calls == [
        {
            "job_id": calls[0]["job_id"],
            "project_id": "project-1",
            "scene_id": "scene-compare",
            "baseline_model": "yolo26n.pt",
            "candidate_model": "yolo26m.pt",
        }
    ]
    assert str(calls[0]["job_id"]).startswith("model-comparison-")
    assert not hasattr(main, "compare_scene_models")


def test_model_comparison_inference_executes_inside_pipeline_job(monkeypatch) -> None:
    scene = {"id": "scene-compare", "revision": 7, "payload": {}}
    progress: list[dict] = []
    published: list[dict] = []

    class FakeSceneRepository:
        def get(self, scene_id):
            assert scene_id == "scene-compare"
            return scene

    class FakePipelineStore:
        def update_progress(self, job_id, token, values):
            assert (job_id, token) == ("comparison-run", "lease-token")
            progress.append(values)
            return True

        def publish(self, job_id, token, *, report):
            assert (job_id, token) == ("comparison-run", "lease-token")
            published.append(report)
            return True

        def fail(self, *_args):
            raise AssertionError("runner should not fail a valid comparison")

    class CurrentHeartbeat:
        @staticmethod
        def current():
            return True

    def compare(received, *, on_progress):
        assert received is scene
        on_progress("baseline", 0, 2)
        on_progress("candidate", 1, 2)
        return {"sceneId": "scene-compare", "comparison": {"verdict": "review"}}

    fake_pipeline = FakePipelineStore()
    monkeypatch.setattr(pipeline_job, "scenes", FakeSceneRepository())
    monkeypatch.setattr(pipeline_job, "pipeline_store", fake_pipeline)
    monkeypatch.setattr(pipeline_job, "pipeline_terminals", fake_pipeline)
    monkeypatch.setattr(pipeline_job, "model_comparison_pipeline", fake_pipeline)
    monkeypatch.setattr(pipeline_job, "compare_scene_models", compare)
    job = PipelineJob(
        id="comparison-run",
        project_id="project-1",
        kind="model-comparison",
        subject_id="scene-compare",
        status="processing",
        state={},
        parameters={"sceneRevision": 7},
        available_at=0.0,
        attempts=1,
        error=None,
        requested_at=0.0,
        updated_at=0.0,
        lease_token="lease-token",
    )

    pipeline_job._execute_model_comparison(  # noqa: SLF001 - executor contract
        job,
        "lease-token",
        CurrentHeartbeat(),
    )

    assert [item["phase"] for item in progress] == ["baseline", "candidate"]
    assert published == [
        {"sceneId": "scene-compare", "comparison": {"verdict": "review"}}
    ]
