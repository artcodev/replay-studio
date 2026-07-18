from __future__ import annotations

"""Command-line adapter for the reconstruction quality evaluator."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .quality_metrics import evaluate_reconstruction_quality


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction QA gates for a scene JSON document."
    )
    parser.add_argument("scene", type=Path, help="Path to a scene JSON document")
    parser.add_argument(
        "--evidence",
        type=Path,
        help="Optional JSON array of per-frame calibration evidence",
    )
    parser.add_argument("--compact", action="store_true", help="Write compact JSON")
    parser.add_argument(
        "--fail-on",
        choices=("never", "reject", "review"),
        default="never",
        help="Return a non-zero exit status for CI",
    )
    args = parser.parse_args(argv)
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    evidence = (
        json.loads(args.evidence.read_text(encoding="utf-8"))
        if args.evidence
        else None
    )
    report = evaluate_reconstruction_quality(scene, evidence)
    print(json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2))
    if args.fail_on == "review" and report["verdict"] in {"review", "reject"}:
        return 2
    if args.fail_on == "reject" and report["verdict"] == "reject":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
