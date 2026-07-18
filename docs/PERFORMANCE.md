# Performance invariants

This document records the control-plane I/O regression that motivated the
durable scheduler and defines measurable acceptance criteria for the cutover.
It is a regression contract, not an estimate of model inference speed.

## 2026-07-18 pre-cutover baseline

The old local Docker stack was observed while no reconstruction process was
doing CPU work and the reconstruction lease table was empty.

Two short samples of the containers' exact Linux network counters showed:

| Container counter | First sample | Second sample | Delta |
| --- | ---: | ---: | ---: |
| reconstruction-runner `eth0` RX | 74,777,158,778 B | 74,817,103,394 B | 39,944,616 B |
| postgres `eth0` TX | 77,533,653,034 B | 77,563,734,598 B | 30,081,564 B |

Docker's cumulative counters at the same point reported approximately
73.7 GB received by the reconstruction runner and 76.5 GB transmitted by
PostgreSQL. Before it was stopped later in the same investigation, the idle
runner had reached 77.9 GB received and PostgreSQL 80.8 GB transmitted. This
traffic was control-plane polling, not model input/output.

The runner was stopped only after confirming 0 active reconstruction leases
and 0% runner CPU. PostgreSQL's exact `eth0` transmit counter then moved from
80,812,201,351 to 80,812,324,156 bytes over a multi-minute observation window:
122,805 bytes instead of tens of megabytes. PostgreSQL, API and the persisted
data remained online. This isolates the retired recovery loop as the traffic
source.

The dominant Scene used for reconstruction measured 4,468,127 bytes as
minified JSON. Repeated recovery/list/status scans deserialised whole Scene
payloads, so an idle poll multiplied payload size by scene count and uptime.
The 80+ GB figure was repeated network transfer/deserialization, not stored
database size.

## Live cutover verification

The live cutover first reduced **Shot 2** from 4,468,127 to 332,876 bytes
(92.55%) while its dense reconstruction series moved to immutable artifacts.
After the clean-cut contract removed the superseded hydrated fields, all eight
persisted Scene payloads measured **81,565 bytes total** and the largest Scene
measured **70,472 bytes**.

The same live measurement kept the data plane outside PostgreSQL: immutable
reconstruction artifacts occupied approximately **2.8 MB** on the shared
filesystem, and the imported immutable video generation occupied approximately
**105 MB**.

A multi-minute idle observation after the cutover produced these exact network
counter deltas:

| Container counter | Delta |
| --- | ---: |
| PostgreSQL `eth0` TX | 935,923 B |
| reconstruction runner `eth0` RX | 11,404 B |
| pipeline runner `eth0` RX | 22,982 B |

Both runners were polling their compact job/lease tables during the sample.
The deltas are bounded control-plane traffic and are no longer proportional to
Scene payload size.

A second independent idle window, with no project/API reads between samples,
measured PostgreSQL TX at 1,456,177 B, reconstruction-runner RX at 8,303 B and
pipeline-runner RX at 17,020 B. It independently remained below the same
acceptance thresholds.

After the final `0008` migration and container rebuild, a clean **69-second**
idle sample measured PostgreSQL TX at **79,333 B**, reconstruction-runner RX at
**6,118 B** and pipeline-runner RX at **12,210 B**. This is the release
verification sample for the current images.

The one-time safety backup created immediately before the cutover was
`/private/tmp/replay-studio-before-20260718-cutover.dump` (888,709 bytes), with
SHA-256
`fbec6154bf887fde9cbaa41e1374749c040261ae2fb7949f6c2aca5c0b4b3980`.
This ephemeral local path records cutover provenance only; it is not the
production backup/restore policy.

## Canonical architecture

- PostgreSQL is the compact control plane: project ownership, job state,
  lease/fingerprint fencing, revisions, artifact references and QA summaries.
- Dense frame series are immutable SHA-256 artifacts on the data plane. The
  current local implementation uses the shared media volume; the manifest is
  backend-neutral.
- Reconstruction discovery, liveness and heartbeat query only
  `reconstruction_jobs` and `reconstruction_leases`.
- Video/multi-pass discovery, liveness and heartbeat query only
  `pipeline_jobs` and `pipeline_job_leases`.
- `AnalysisRun` is telemetry. It is never a scheduling source.
- Scene list/project dashboard/status polling must not select `scenes.payload`.
- A dense result is fetched explicitly through a bounded reconstruction-series
  window, never through ordinary status polling.

The measured live values above are the acceptance baseline: the representative
cutover exceeded the 92% size gate, and the final compact rows are materially
smaller again after removing the superseded fields.

## Automated regression gates

The test suite must prove all of the following:

1. idle job discovery, claim liveness and heartbeat execute no query containing
   `scenes.payload` or `FROM scenes`;
2. project scene/segment lists use indexed compact columns and do not load one
   Scene JSON per item;
3. a published representative reconstruction is at least 92% smaller than its
   hydrated working document;
4. reconstruction-series requests are limited to 30 seconds and 900 frames;
5. missing/corrupt artifacts fail explicitly rather than falling back to an
   embedded dense field;
6. cancellation fences stale publication, and final result/job/telemetry state
   is committed atomically.

## Repeatable live verification

After migrating and recreating only the application services:

1. confirm that no physical job is active;
2. sample `/proc/net/dev` in PostgreSQL and both runners;
3. wait at least 30 seconds, covering several recovery polls;
4. sample the same counters again;
5. query compact jobs, leases, Scene byte sizes and stale `AnalysisRun` rows;
6. open Projects, Timeline and one Editor scene and repeat the sample.

Acceptance thresholds for an idle 30-second window are:

- each runner receives less than 1 MB from PostgreSQL;
- PostgreSQL transmits less than 2 MB total to application containers;
- no idle query selects `scenes.payload`;
- no job without a current lease remains in `processing` after recovery;
- no active telemetry row exists without a matching authoritative compact job.

These thresholds intentionally allow health checks and compact polling while
rejecting any return to payload-proportional idle traffic.
