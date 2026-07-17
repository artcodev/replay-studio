# Real-model validation harness

This opt-in harness measures the already configured `identity-worker` and
`jersey-ocr-worker` against an explicitly labelled crop set. It does not import
either model runtime, fetch checkpoints, build images, or download weights. It
only calls workers that are already `ready` over their public HTTP contracts.

The canonical input contract is
[`manifest.schema.json`](manifest.schema.json). Start from
[`manifest.example.json`](manifest.example.json), replace every placeholder,
and keep the manifest next to its relative `crops/` paths.

## What must be labelled

Every crop has:

- an immutable crop ID and relative image path;
- ground-truth `personId` and role (`player`, `goalkeeper`, `referee`, `other`,
  or `ball`);
- a jersey label: an exact one/two-digit number when it is genuinely readable,
  or `{ "readable": false, "number": null }` when OCR should abstain.

`identityPairs` explicitly labels same-person and different-person pairs. The
harness rejects a pair whose `samePerson` value disagrees with the two
`personId` labels. It also rejects datasets without both pair classes, without
readable and expected-abstention OCR examples, or with conflicting readable
shirt labels for one person.

Acceptance thresholds live in the manifest rather than in code. They therefore
travel with the dataset version and are copied verbatim into every report.
Changing a threshold does not change the dataset fingerprint; changing a
label, pair, crop path, or crop bytes does.

## Metrics and report

The versioned `football-model-validation-report.v1` JSON contains the manifest
version, immutable dataset metadata, `sha256:` dataset fingerprint, selected
thresholds, provider provenance, checks, aggregate metrics, and auditable
per-crop/per-pair outcomes. Real HTTP runs also record end-to-end wall time,
throughput, request batch size, and each worker's cache/provider-inference
diagnostics so a warm-cache result cannot be presented as cold model latency.

Identity checks include:

- exactly 256 finite, L2-normalized dimensions for every accepted embedding;
- backend/model version, both checkpoint SHA-256 values, and SoccerNet commit;
- usable crop and labelled-pair coverage;
- cosine-distance distributions (`p05`, `p50`, `p95`) for same-person and
  different-person pairs plus their median separation;
- full role confusion matrix, including `__abstain__`.

Jersey OCR checks include:

- backend/provider/model provenance and response completeness;
- exact accuracy on readable labels;
- substitutions and abstentions on readable labels;
- accuracy and precision of abstention on explicitly unreadable crops;
- conflicting recognized numbers across crops of one labelled person.

Exit codes are `0` for pass, `1` for measured threshold failure, `2` for a
missing opt-in/invalid manifest, `3` for unavailable workers/assets, and `4`
for a worker contract/provenance failure. Unavailable runs still write a
versioned report with the dataset fingerprint and reason.

## Docker workers, no downloads

Use already provisioned images. `--no-build` is intentional: this validation
command must not turn into a checkpoint download or a Docker build.

```bash
docker compose up -d --no-build identity-worker jersey-ocr-worker
curl --fail http://127.0.0.1:8091/health/ready
curl --fail http://127.0.0.1:8093/health/ready

MODEL_VALIDATION_OPT_IN=1 \
  .venv/bin/python services/model-validation/run_validation.py \
  --manifest /absolute/path/to/labelled/manifest.json \
  --output /absolute/path/to/reports/worker-validation.json
```

If an image has not already been provisioned, follow each worker's offline
instructions and use `REID_DOWNLOAD_MODELS=0` /
`JERSEY_OCR_DOWNLOAD_MODELS=0`. Jersey runtime downloads must remain disabled
(`JERSEY_OCR_ALLOW_RUNTIME_DOWNLOADS=0`, already enforced by Compose). The
harness refuses inference unless `MODEL_VALIDATION_OPT_IN=1` is set.

Run only one worker when diagnosing it:

```bash
MODEL_VALIDATION_OPT_IN=1 \
  .venv/bin/python services/model-validation/run_validation.py \
  --worker identity \
  --manifest /absolute/path/to/labelled/manifest.json \
  --output /tmp/identity-validation.json

MODEL_VALIDATION_OPT_IN=1 \
  .venv/bin/python services/model-validation/run_validation.py \
  --worker jersey-ocr \
  --manifest /absolute/path/to/labelled/manifest.json \
  --output /tmp/jersey-validation.json
```

## Locally started workers

Start the real workers by the commands in their own READMEs, verify their
readiness, then point the host harness at them. The normal project `.venv`
already has these dependencies; a standalone environment can install
`services/model-validation/requirements.txt`.

```bash
MODEL_VALIDATION_OPT_IN=1 \
IDENTITY_WORKER_URL=http://127.0.0.1:8091 \
JERSEY_OCR_WORKER_URL=http://127.0.0.1:8093 \
  .venv/bin/python services/model-validation/run_validation.py \
  --manifest /absolute/path/to/labelled/manifest.json \
  --output /tmp/real-model-validation.json
```

The test suite is model-free by default:

```bash
.venv/bin/pytest -q services/model-validation/tests -rs
```

It runs the tiny synthetic fixture only to verify parsing, fingerprinting,
metric math, threshold failures, provenance checks, and report versioning. The
two real-worker tests are skipped with an explicit reason until all of these
are supplied:

```bash
MODEL_VALIDATION_OPT_IN=1 \
MODEL_VALIDATION_MANIFEST=/absolute/path/to/labelled/manifest.json \
  .venv/bin/pytest -q services/model-validation/tests/test_real_workers.py -rs
```

## Accuracy-claim rule

No result from the fake fixture, an unlabelled clip, a hand-picked pair, or a
worker readiness check is an accuracy result. Do not publish an accuracy,
threshold, provider comparison, or “model validated” claim without retaining
the exact manifest, dataset version/license/source, dataset fingerprint,
provider model/checkpoint provenance, and generated report. A passing report
means only that one provider build met the declared gates on that manifest; it
is not a universal football benchmark.
