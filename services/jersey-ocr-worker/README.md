# Jersey OCR worker

This isolated service turns already-cropped player images into **jersey-number
evidence**. Its HTTP contract is independent of the selected OCR library. The
default production provider matches the SoccerNet `sn-gamestate` baseline:
MMOCR 1.0.1 with DBNet (`dbnet_resnet18_fpnc_1200e_icdar2015`) and SAR. Set
`JERSEY_OCR_PROVIDER=easyocr` to use the lighter EasyOCR 1.7.1 alternative.
The image uses Python 3.10, CPU-only PyTorch 1.13.1, and the official
MMCV 2.0.1 CPU wheel. Compose pins this worker to Linux x86_64 because that
MMCV wheel is not published for Linux arm64; Docker Desktop runs it under
emulation on Apple Silicon.

The service never converts an unavailable model into fake evidence. HTTP stays
live, `/health/ready` returns 503, and the main API reports the optional worker
as unavailable. Ambiguous or low-confidence readings are retained for audit but
do not publish an accepted number.

## Run

```bash
docker compose up --build -d jersey-ocr-worker
curl --fail http://127.0.0.1:8093/health/live
curl --fail http://127.0.0.1:8093/health/ready
```

The Docker build downloads the selected provider's official model assets. To
build in a pre-provisioned/offline environment set
`JERSEY_OCR_DOWNLOAD_MODELS=0`; readiness will remain false unless the required
cache is present in the image. EasyOCR can be selected at build and runtime:

```bash
JERSEY_OCR_PROVIDER=easyocr docker compose build jersey-ocr-worker
JERSEY_OCR_PROVIDER=easyocr docker compose up -d jersey-ocr-worker
```

Provider dependencies are image-specific: the MMOCR image contains only
`opencv-python`, while the EasyOCR variant contains only
`opencv-python-headless`. This avoids two distributions overwriting the same
`cv2` package and makes the selected runtime reproducible.

MMOCR is the accuracy-oriented SoccerNet reference, but is heavy and its first
build/model load is slow. EasyOCR is operationally simpler and is useful as a
measured challenger, not a silent fallback.

## HTTP contract

`POST /v1/analyze` uses multipart form data:

- repeated `crops` image files, each containing one tightly cropped person;
- `manifest`, a JSON object with `contractVersion: "jersey-ocr.v1"` and an
  `items` array;
- each item has a unique `cropId`, a `fileIndex`, and optional
  `observationId`, `trackletId`, `frameIndex`, and `timestamp`.

Each input item always has a corresponding output item. `status` is one of
`recognized`, `no-number`, `low-confidence`, `ambiguous`, or `rejected`.
Only `recognized` carries an accepted `number`; raw provider text, competing
numeric candidates, crop QA, and decision reasons remain available for later
tracklet voting and debugging. The worker accepts at most two-digit jersey
numbers and preserves leading zeroes.

For crop-level providers, decoded crop pixels are cached by model version and
crop/decision policy in a bounded TTL LRU. Identical inputs within one request
are deduplicated. Cache use is automatically disabled for a future provider
that reports `inferenceScope: tracklet`, because its result may depend on the
other views in the group.

Crop-level recognition is deliberately not canonical identity. The API
pipeline aggregates quality-ranked independent observations, rejects
conflicting reliable votes, keeps a bounded reservoir per prospective manual
split partition, and re-fuses evidence after final observation ownership is
known. A reliable number may strengthen a compatible canonical-person link,
but roster matches remain review candidates until the user confirms them
through the dedicated canonical bind endpoint.

## Tests

Contract tests inject a fake provider and therefore need no model downloads:

```bash
.venv/bin/pytest -q services/jersey-ocr-worker/tests
```

## Labelled real-model validation

The fake-provider suite validates only the HTTP contract. The opt-in harness in
[`../model-validation/README.md`](../model-validation/README.md) calls an
already-ready real worker and measures exact readable numbers, substitutions,
expected abstention, and same-person number conflicts. It never downloads
weights, and it refuses to run without an explicit labelled manifest and
`MODEL_VALIDATION_OPT_IN=1`.
