from __future__ import annotations

"""Command-line adapter for the labelled reconstruction benchmark."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .quality_benchmark_contract import BenchmarkValidationError
from .quality_benchmark_report import evaluate_benchmark


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BenchmarkValidationError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a Replay Studio quality benchmark"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = evaluate_benchmark(
        _load_json(arguments.manifest),
        _load_json(arguments.predictions),
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
