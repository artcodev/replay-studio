# Reconstruction performance plan

Measured on the current local CPU with a real `Shot01/frame_00001.jpg` input:

| Path | Time |
|---|---:|
| Cold HTTP request before optimization (model load included) | 15.34 s |
| Warm uncached HTTP requests before optimization | 9.47 s / 8.04 s |
| Model construction after instrumentation | 5.76 s |
| First unique frame after instrumentation | 9.32 s |
| Same content from the new cache | 4.2 ms |
| Same accepted highlights frame through HTTP (verified after restart) | 7.1 ms server / 10.0 ms client |
| Full 1-A reconstruction, cold cache (41 calibrated frames) | 431.0 s |
| Full 1-A reconstruction, warm cache (same inputs) | 13.6 s |

The real full-shot comparison is a roughly **31.7×** warm-path speedup. The two
runs emitted identical tracks and ball samples; all 41/41 calibration results
remained valid. This is an end-to-end reconstruction measurement, not a
single-model microbenchmark.

The first-frame profile is more useful than one total number: model forward
passes take about 2.80 s, official heatmap decoding takes about 6.44 s, and
geometry takes about 0.05 s. On this machine the CPU heatmap decoder, not the
homography solver, is the main hot path.

## Implemented now

- Both PnLCalib models preload before worker readiness, removing hidden cold load
  from the first accepted request.
- Results (including valid negative results) use a bounded TTL LRU keyed by exact
  frame SHA-256 and model/config/checkpoint version.
- Duplicate content inside one request is inferred once.
- Decode and inference run in a threadpool so the FastAPI event loop remains
  responsive; the model lock reports queue wait separately.
- Responses expose decode, tensor assembly, each model, heatmap decode, geometry,
  cache, batching, and total timings.
- Manual identity corrections automatically queue tracking rebuild, and cached
  calibration frames return in milliseconds.

## Next options, ordered by value

| Option | Expected effect | Accuracy / engineering risk | Recommendation |
|---|---|---|---|
| Persist detector/calibration/tracker-state artifacts by content+config hash | Manual role/identity edits rebuild from evidence without rerunning either model | Requires versioned artifact schema and invalidation tests | Highest priority; it changes the edit loop from full inference to seconds/sub-second |
| Adaptive calibration anchors | Run PnLCalib on the first/last frame, camera-motion changes, and line-novelty frames; use the existing temporal graph between them | A sparse plan can miss a bad pan/zoom unless max-gap and residual gates remain strict | Implement behind benchmark flag; fall back to exact frame inference when QA fails |
| CUDA worker | HRNet forward and heatmap operations move off CPU; multiple jobs can batch | Deployment/VRAM and CUDA image maintenance | Best production speedup; keep the neutral worker JSON contract |
| Replace/vectorize official heatmap decoder | Targets the measured 6.44 s hot path directly | Highest numerical-regression risk; keypoint/line ordering and thresholds must remain identical | Prototype with golden-output comparison on the validation set before enabling |
| ONNX Runtime / TensorRT | Faster fixed-shape HRNet forward, quantization options | Conversion compatibility, output drift, two model variants | Evaluate after decoder profiling; forward pass is currently smaller than decode |
| Persistent disk/object cache | Cache survives worker restart and can be shared | Storage eviction, schema migration, privacy of uploaded frames | Store result JSON rather than source pixels; key includes model version |
| Progressive preview | Fast low-resolution/keypoint preview followed by exact PnLCalib result | UI must never present preview as final quality | Useful for interaction, not a substitute for exact accepted reconstruction |
| Larger batches | Better GPU throughput | CPU RAM/latency can worsen; current default is 2 | Tune separately for CPU and GPU with real shot-length benchmarks |

PyTorch in the current environment is built with MPS support but reports MPS as
unavailable, and CUDA is unavailable. Device switching therefore is not a local
optimization on this host; a server/GPU worker is the practical accelerator.

## Performance gates

Every benchmark records cold and warm paths separately. A change is accepted only
when calibration acceptance, semantic keypoints/lines, homography, and QA verdict
remain equal (or improve) on the validation set.

Initial targets:

- cached current-frame calibration: under 50 ms server time;
- editor correction to queued tracking feedback: under 300 ms;
- API/event loop remains responsive while a worker job runs;
- no repeated detector/calibrator inference when only identity/role metadata
  changed;
- GPU target after worker deployment: under 1 s per unique calibration frame or
  under 5 s for an adaptively sampled short highlight, measured end to end.
