# Identity worker

The worker exposes the SoccerNet PRTReID/BPBreID appearance model through a
small HTTP boundary. It deliberately remains **not ready** when a dependency,
checkpoint, checksum, requested device, or actual model load fails. It never
returns random or generic substitute embeddings.

## Pinned reference

- SoccerNet `sn-gamestate`: `1c958345067218297d221e45e1a6405f975f83e0`
- PRTReID: `30617a75967e84d5d516959c4b84cbeea6f56493`
- BPBreID/torchreid: `02570d5c893977685de7a547a58c0aa87fa4788c`
- Python 3.10, CPU-only PyTorch 1.13.1, CPU-only torchvision 0.14.1

The Docker build downloads and verifies these official checkpoints by default:

| File | Source | MD5 |
| --- | --- | --- |
| `prtreid-soccernet-baseline.pth.tar` | `https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1` | `9633825232bc89f23a94522c5561650e` |
| `hrnetv2_w32_imagenet_pretrained.pth` | `https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1` | `58ea12b0420aa3adaa2f74114c9f9721` |

For an offline build, place them in `services/identity-worker/models/`, verify
them, and set `REID_DOWNLOAD_MODELS=0`:

```bash
md5 services/identity-worker/models/prtreid-soccernet-baseline.pth.tar
md5 services/identity-worker/models/hrnetv2_w32_imagenet_pretrained.pth
REID_DOWNLOAD_MODELS=0 docker compose build identity-worker
```

The same checks run inside the provider before model construction.

## Run

```bash
docker compose up --build -d identity-worker
curl --fail http://127.0.0.1:8091/health/live
curl --fail http://127.0.0.1:8091/health/ready
```

`/health/live` proves only that FastAPI is serving. `/health/ready` performs the
real checkpoint verification and model load. The first readiness request can
therefore take time on CPU.

The checkpoints are embedded in the image. Compose intentionally does not
mount the host `models/` directory over `/models`, because an empty mount would
silently hide build-time assets and leave the service permanently unready.

Embedding requests use multipart form data:

- repeated `frames` JPEG files;
- `manifest`, a JSON object with frame/file indices and observation bboxes.

The response contains normalized 256D embeddings only for usable crops.
Small, empty, or blurry crops remain in the result with `usable: false` and an
explicit rejection reason.

## Content cache and request deduplication

The worker keeps a bounded process-local TTL-LRU of both valid embeddings and
auditable crop-QA rejections. A cache key includes:

- exact decoded crop bytes, shape and dtype;
- the exact source bbox and crop/quality policy;
- backend, model version, full checkpoint SHA-256 and SoccerNet commit;
- embedding contract and cache schema version.

Changing the bbox, model/checkpoint, crop policy or schema therefore produces
a safe miss. Cached entries are revalidated before use; malformed, non-finite,
wrong-dimension or non-normalized data is discarded and recomputed. Identical
crops within one request are embedded once, and concurrent requests share one
in-flight provider call. Provider failures wake waiters without publishing a
partial entry.

Defaults can be tuned or disabled without changing the HTTP contract:

```bash
REID_CACHE_MAX_ENTRIES=4096
REID_CACHE_TTL_SECONDS=86400
REID_CACHE_WAIT_TIMEOUT_SECONDS=900
```

Set either max entries or TTL to `0` to disable persistent reuse; concurrent
single-flight result sharing remains active. `/health/ready` reports capacity,
current size, in-flight work, hits, misses, deduplication, expiration,
eviction, corruption and provider-failure counters. Every embedding response
also reports request-local cache diagnostics and `cacheSource` per item.

## Test without model assets

The service contract is tested through an injected fake provider; production
startup still uses only PRTReID:

```bash
.venv/bin/pytest -q services/identity-worker/tests
```

## Labelled real-model validation

Worker contract tests are not an accuracy benchmark. The opt-in harness in
[`../model-validation/README.md`](../model-validation/README.md) measures the
already-ready provider on labelled same/different-person crop pairs, verifies
the normalized 256D contract and checkpoint provenance, and reports distance
distributions plus role confusion. It never downloads weights. Use an explicit
versioned manifest and retain its dataset fingerprint with the report.
