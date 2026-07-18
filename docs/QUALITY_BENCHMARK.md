# Reconstruction quality benchmark

## Why this exists

Runtime diagnostics can tell us that a trajectory is too fast, calibration
coverage is low, or a result used a fallback. They cannot tell us that the
reconstruction is accurate. Accuracy needs independently labelled frames and a
repeatable evaluator.

The benchmark therefore has two inputs:

1. a versioned, reviewable ground-truth manifest;
2. a prediction export produced by one immutable pipeline run.

The evaluator never substitutes a runtime confidence or smoothness heuristic
for ground truth. If a label category is absent, its metrics are `null` and the
category is unavailable. A draft or partially labelled manifest is not an
accuracy claim.

## Files

- `benchmarks/schema/benchmark-manifest-v1.schema.json` — ground-truth JSON
  Schema;
- `benchmarks/schema/predictions-v1.schema.json` — pipeline export JSON Schema;
- `benchmarks/manifests/spain-belgium-highlight-v1.json` — deliberately
  unlabelled draft for the local video;
- `apps/api/app/quality_benchmark_contract.py` and
  `quality_benchmark_validation.py` — versioned contract and input invariants;
- `apps/api/app/quality_benchmark_context.py` — manifest/prediction pairing;
- `apps/api/app/quality_benchmark_people.py`,
  `quality_benchmark_calibration.py`, and `quality_benchmark_ball.py` —
  independent deterministic evaluators;
- `apps/api/app/quality_benchmark_report.py` — pure report orchestration;
- `apps/api/app/quality_benchmark_cli.py` — file/CLI adapter.

Both document types use `schemaVersion: "1.0"`. A breaking meaning or shape
change requires a new schema version; do not silently reinterpret an existing
frozen manifest.

## Coordinate contract

- Image points and boxes use source-frame pixels with the origin at the top
  left. A box is `[x, y, width, height]`.
- Pitch points use metres in the project pitch coordinate system: centre
  origin, `x` along pitch length and `z` along pitch width. A point is `[x, z]`.
- `frameIndex` refers to the original asset, not a sampled-frame ordinal.
- Person labels contain only visible people. The same real person keeps the
  same ground-truth `id` throughout one sample.
- A ball label has `visible: true` only when the image provides an observable
  ball centre. Interpolated or inferred ball positions must not be exported as
  observed predictions for this metric.

For a calibration annotation, the manifest point contains a known image/pitch
correspondence. A prediction with the same point `id` has two optional values:

- `image`: where the estimated camera projects the labelled pitch point;
- `pitch`: where the estimated inverse projection maps the labelled image
  point.

This makes the pixel reprojection error and metric projection error separately
measurable. A pipeline adapter may derive these point predictions from a
homography or camera model; the evaluator itself is model-independent.

## Metrics

### Person detection

Detections are assigned per frame using a maximum-cardinality bipartite match,
then maximum IoU, at `evaluation.personIouThreshold` (default `0.5`). The report
contains TP, FP, FN, precision, recall, F1 and matched-IoU distribution.

`averagePrecisionAtIou` is confidence-ranked AP at this one IoU threshold. It
is useful for comparing detector configurations on the same frozen set, but it
is not COCO mAP and must not be named as such.

### Calibration

For labelled semantic points, the report contains:

- reprojection coverage and pixel error (mean, median, p95, RMSE, maximum);
- inverse/metric projection coverage and metre error with the same summary.

Coverage is reported next to error so a method cannot appear accurate by
returning only a few easy points.

### Ball

A prediction is correct when its centre is within
`evaluation.ballPointThresholdPx` (default `24 px`) of a visible labelled ball.
The report contains recall, precision, F1 and point-error distribution. A far
prediction is both a miss and a false positive. A prediction on a frame whose
label says the ball is not visible is a false positive.

### Identity

Predicted person boxes are first matched to labelled boxes at the configured
person IoU threshold. Those assignments are passed to the existing global
bipartite IDF1 evaluator. The report contains ID precision, ID recall, IDF1,
ID switches, fragments and fragments per labelled identity.

HOTA and GS-HOTA remain `null`. They require the official SoccerNet evaluation
code and are intentionally not approximated here.

## Minimal prediction export

```json
{
  "$schema": "../schema/predictions-v1.schema.json",
  "schemaVersion": "1.0",
  "benchmarkId": "spain-belgium-highlight-v1",
  "run": {
    "id": "analysis-run-id",
    "pipelineVersion": "git-commit-or-image-digest",
    "modelVersions": {
      "personDetector": "model-name-and-checksum",
      "ballDetector": "model-name-and-checksum"
    }
  },
  "samples": [
    {
      "sampleId": "opening-wide-shot",
      "frames": [
        {
          "frameIndex": 0,
          "persons": [
            {
              "trackId": "track-17",
              "bbox": [100, 80, 22, 61],
              "confidence": 0.91
            }
          ],
          "ball": {
            "center": [301, 214],
            "confidence": 0.73,
            "source": "observed"
          },
          "calibrationPoints": [
            {
              "id": "left-penalty-top",
              "image": [87.2, 176.5],
              "pitch": [-36.0, -20.1]
            }
          ]
        }
      ]
    }
  ]
}
```

The run metadata should identify the code revision, model names/checksums and
configuration required to reproduce it. Do not overwrite a previous export;
write a new run ID.

## Running the evaluator

From `apps/api`:

```bash
../../.venv/bin/python -m app.quality_benchmark_cli \
  ../../benchmarks/manifests/spain-belgium-highlight-v1.json \
  /path/to/predictions.json \
  --output /path/to/report.json
```

The top-level `status` is:

- `evaluated` when all four metric families have ground truth;
- `partial` when at least one, but not all, can be evaluated;
- `unavailable` when the manifest contains no usable labels.

These values describe label availability, not a pass/fail quality gate. Product
acceptance thresholds should be selected only after the gold set is large and
representative enough to measure normal variation.

## Building the gold set

Start with a small, diverse frozen slice rather than many adjacent easy frames:

1. include wide, medium and cropped broadcast shots;
2. include both pitch directions, camera pans/zooms and partial-field views;
3. include occlusion, crowded penalty areas, goalkeepers, referees, replay
   transitions and ball blur;
4. sample identity labels across gaps and reappearances, not just one frame;
5. label calibration points across visible semantic line families and record
   only unambiguous correspondences;
6. have a second reviewer inspect labels before changing manifest status to
   `frozen`;
7. record the video SHA-256 and annotation revision before using scores in CI.

The current Spain–Belgium manifest is `draft` and contains no gold frames. Its
only purpose is to make the expected asset and initial segment explicit. Until
reviewed annotations are added, the evaluator will correctly return
`unavailable` rather than a favourable score.

## CI and regression policy

Once a manifest is frozen, keep its labels immutable and store a baseline
report for a named pipeline version. A regression job should:

1. export predictions from the candidate pipeline;
2. validate both JSON documents against their schemas;
3. run this evaluator;
4. compare metric deltas and coverage against reviewed tolerances;
5. retain the prediction and report artifacts for diagnosis.

Do not gate on a synthetic fixture, on the unlabelled draft, or on runtime QA
confidence. Synthetic tests only verify evaluator arithmetic.
