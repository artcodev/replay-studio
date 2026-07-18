from __future__ import annotations

"""One process-isolated reconstruction job owned by the dedicated runner."""

import argparse

from .reconstruction_worker import reconstruct_scene_by_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id")
    parser.add_argument("run_id")
    parser.add_argument("input_fingerprint")
    arguments = parser.parse_args(argv)
    reconstruct_scene_by_id(
        arguments.scene_id,
        arguments.run_id,
        arguments.input_fingerprint,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
