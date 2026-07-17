#!/usr/bin/env python3
"""Run the labelled harness against already-ready real worker services."""

from __future__ import annotations

import argparse
import json
import os
import sys

from validation_harness import (
    ManifestError,
    WorkerProtocolError,
    WorkerUnavailable,
    build_unavailable_report,
    load_manifest,
    run_http_validation,
    write_report,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate configured identity/OCR workers on an explicitly labelled crop manifest. "
            "The command never downloads or loads model weights itself."
        )
    )
    parser.add_argument("--manifest", required=True, help="Path to a v1 labelled manifest JSON")
    parser.add_argument("--output", required=True, help="Path for the versioned JSON report")
    parser.add_argument(
        "--worker",
        choices=("all", "identity", "jersey-ocr"),
        default="all",
        help="Worker subset to evaluate",
    )
    parser.add_argument(
        "--identity-url",
        default=os.environ.get("IDENTITY_WORKER_URL", "http://127.0.0.1:8091"),
    )
    parser.add_argument(
        "--jersey-ocr-url",
        default=os.environ.get("JERSEY_OCR_WORKER_URL", "http://127.0.0.1:8093"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-request seconds")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if os.environ.get("MODEL_VALIDATION_OPT_IN") != "1":
        print(
            "Refusing real-worker inference: set MODEL_VALIDATION_OPT_IN=1 after provisioning "
            "the configured worker assets. The harness never downloads weights.",
            file=sys.stderr,
        )
        return 2
    if arguments.batch_size <= 0 or arguments.timeout <= 0:
        print("--batch-size and --timeout must be positive", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(arguments.manifest)
    except ManifestError as exc:
        print(f"Invalid validation manifest: {exc}", file=sys.stderr)
        return 2
    workers = (
        ("identity", "jersey-ocr")
        if arguments.worker == "all"
        else (arguments.worker,)
    )
    try:
        report = run_http_validation(
            manifest,
            workers=workers,
            identity_url=arguments.identity_url,
            jersey_ocr_url=arguments.jersey_ocr_url,
            batch_size=arguments.batch_size,
            timeout_seconds=arguments.timeout,
        )
    except WorkerUnavailable as exc:
        report = build_unavailable_report(
            manifest,
            selected_workers=workers,
            reason=str(exc),
        )
        target = write_report(arguments.output, report)
        print(f"Real model assets/worker unavailable; report written to {target}: {exc}")
        return 3
    except WorkerProtocolError as exc:
        report = build_unavailable_report(
            manifest,
            selected_workers=workers,
            reason=f"worker-protocol-error: {exc}",
        )
        target = write_report(arguments.output, report)
        print(f"Worker protocol failed; report written to {target}: {exc}", file=sys.stderr)
        return 4
    target = write_report(arguments.output, report)
    print(json.dumps({"status": report["status"], "report": str(target)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
