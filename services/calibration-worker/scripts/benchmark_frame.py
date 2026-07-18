#!/usr/bin/env python3
"""Measure cold-model and repeated single-frame PnLCalib latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from calibration_worker_service.engine_factory import create_pnlcalib_engine
from calibration_worker_service.frame_decoder import decode_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", type=Path, help="JPEG or PNG frame to calibrate")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeated requests")
    parser.add_argument(
        "--uncached",
        action="store_true",
        help="Disable the response cache so every run executes both models",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    payload = args.frame.read_bytes()
    engine_started = perf_counter()
    engine = create_pnlcalib_engine(cache_max_entries=0 if args.uncached else None)
    engine_wall_seconds = perf_counter() - engine_started

    runs = []
    for run_index in range(args.runs):
        request_started = perf_counter()
        decode_started = perf_counter()
        frame = decode_frame(run_index, payload)
        decode_seconds = perf_counter() - decode_started
        result = engine.calibrate([frame])
        runs.append(
            {
                "run": run_index + 1,
                "accepted": bool(result.frames),
                "wallSeconds": round(perf_counter() - request_started, 6),
                "decodeSeconds": round(decode_seconds, 6),
                "diagnostics": result.diagnostics.to_wire(),
            }
        )

    print(
        json.dumps(
            {
                "frame": str(args.frame.resolve()),
                "mode": "uncached" if args.uncached else "content-cache",
                "modelVersion": engine.model_version,
                "engineConstructionWallSeconds": round(engine_wall_seconds, 6),
                "modelLoadSeconds": round(engine.model_load_seconds, 6),
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
