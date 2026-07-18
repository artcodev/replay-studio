# WASB ball worker

Optional server boundary for the pinned WASB-SBDT soccer model. It exposes
the official three-frame HRNet checkpoint without importing its older pinned
PyTorch environment into the main API process.

The service never substitutes a random, generic, or fake detector. Liveness
only proves that HTTP is serving; readiness verifies the checkpoint SHA-256,
imports the pinned HRNet source, loads every state-dict key strictly, and moves
the model to the requested device.

Internally, the provider-neutral candidate/status contract, environment
configuration, verified HRNet loader, affine/component geometry, inference
runtime, and provider factory are separate owners. `main.py` only composes the
contract with the factory; there is no aggregate provider facade.

## Pinned assets

| Asset | Project path | Runtime path |
| --- | --- | --- |
| WASB soccer checkpoint | `models/wasb-soccer-best.pth.tar` | `/models/wasb-soccer-best.pth.tar` |
| Upstream HRNet source | `.references/WASB-SBDT/src/models/hrnet.py` | `/opt/wasb-reference/hrnet.py` |
| Upstream license | `.references/WASB-SBDT/LICENSE.md` | remains with the checked-in reference |

Expected checkpoint SHA-256:

```text
d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c
```

The implementation keeps the upstream `wasb.yaml` architecture, three-frame
input/output, 512x288 affine preprocessing, ImageNet normalization, sigmoid
heatmaps, 0.5 threshold, and weighted connected-component centers. Candidate
`confidence` is the component's probability peak in `[0, 1]`; the upstream
component sum is retained separately as `componentScore` because it is not a
calibrated probability.

## HTTP contract

- `GET /health/live` — process liveness; does not claim the model works.
- `GET /health/ready` — real asset verification and model load.
- `POST /v1/detections` — repeated multipart `frames` plus a JSON `manifest`.

Example batch manifest:

```json
{
  "contractVersion": 1,
  "maxCandidates": 12,
  "targetIndex": 1,
  "frames": [
    {"fileIndex": 0, "frameIndex": 120, "timestampMs": 4000},
    {"fileIndex": 1, "frameIndex": 121, "timestampMs": 4040},
    {"fileIndex": 2, "frameIndex": 122, "timestampMs": 4080}
  ]
}
```

Every returned candidate contains `x`, `y`, `confidence`, `backend`,
`modelVersion`, `heatmapPeak`, component evidence, and `sourceFrameIndex`.
Short first/last windows use explicit edge-repeat padding and report
`temporalPadding: true`; it is never presented as extra observed evidence.

## Docker Compose integration

The image intentionally contains neither the checkpoint nor copied upstream
source. Provision both read-only so a missing asset remains visible. Add this
service to the root `docker-compose.yml`:

```yaml
  ball-worker:
    build:
      context: ./services/ball-worker
      args:
        TORCH_VERSION: 2.2.2
        TORCH_INDEX_URL: ${WASB_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}
    environment:
      WASB_DEVICE: ${WASB_DEVICE:-cpu}
      WASB_PRELOAD: ${WASB_PRELOAD:-1}
      WASB_WEIGHTS: /models/wasb-soccer-best.pth.tar
      WASB_HRNET_SOURCE: /opt/wasb-reference/hrnet.py
      WASB_SCORE_THRESHOLD: ${WASB_SCORE_THRESHOLD:-0.5}
    volumes:
      - ./models/wasb-soccer-best.pth.tar:/models/wasb-soccer-best.pth.tar:ro
      - ./.references/WASB-SBDT/src/models/hrnet.py:/opt/wasb-reference/hrnet.py:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8092/health/ready', timeout=15)"]
      interval: 20s
      timeout: 18s
      retries: 20
      start_period: 120s
    ports:
      - "8092:8092"
```

For NVIDIA/CUDA, use a Torch wheel index compatible with the host runtime,
for example `https://download.pytorch.org/whl/cu121`, set
`WASB_DEVICE=cuda:0`, and add the Compose GPU reservation. A requested CUDA
device without working CUDA makes readiness return `503`; it never falls back
silently to CPU. CPU is supported for local correctness checks but is slow.

The API integration needs these values when WASB is selected:

```text
BALL_DETECTION_BACKEND=wasb-service
BALL_WASB_WORKER_URL=http://ball-worker:8092/v1/detections
```

The complete detector, resolver, QA, and fallback contract is documented in
[`docs/BALL_TRACKING.md`](../../docs/BALL_TRACKING.md).

Do not make the main API depend on `condition: service_healthy` unless WASB is
mandatory. As an optional challenger, `condition: service_started` preserves
the editor while `/health/ready` and reconstruction diagnostics expose a
missing model/runtime. Any fallback detector must be configured explicitly in
the API; this worker itself has no silent fallback.

## Local test without Torch or checkpoint loading

The contract suite injects a fake provider and does not import the pinned
model runtime:

```bash
.venv/bin/pytest -q services/ball-worker/tests
```

To exercise the real local checkpoint with the repository environment:

```bash
PYTHONPATH=services/ball-worker WASB_DEVICE=cpu \
  .venv/bin/uvicorn ball_worker_service.main:app --host 127.0.0.1 --port 8092
curl --fail http://127.0.0.1:8092/health/ready
```
