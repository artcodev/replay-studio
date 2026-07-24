"""Download and install the owner-supplied football.pt person detector.

Run this yourself — the assistant deliberately never downloads third-party
checkpoint pickles. The owner manages the licensing of these weights
(SoccerMaster publishes no license; see docs/APPLICATION_IMPROVEMENT_PROJECT.md
§20).

    ./.venv/bin/python scripts/install_football_weights.py
    ./.venv/bin/python scripts/install_football_weights.py --file SoccerNetGSR_Detection.pt

What it does:
1. downloads the checkpoint from the SoccerMaster HuggingFace repository;
2. prints its SHA-256 and size (record them: this is your provenance);
3. loads it with Ultralytics and prints the class-name map, verifying that
   the pipeline's name-based mapping (player/goalkeeper/referee/ball) will
   recognize people and the ball;
4. installs it to apps/api/models/football.pt (baked into the api image on
   rebuild) and symlinks models/football.pt for local dev runs;
5. prints the docker rebuild commands.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


REPOSITORY = "https://huggingface.co/xleprime/SoccerMaster/resolve/main"
DEFAULT_FILE = "yolo_v8x6_finetuned.pt"
PERSON_CLASS_NAMES = {"person", "player", "goalkeeper", "referee", "staff"}
BALL_CLASS_NAMES = {"ball", "sports ball"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def download(url: str, target: Path) -> str:
    digest = hashlib.sha256()
    received = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if total:
                print(
                    f"\r  {received / 1e6:7.1f} / {total / 1e6:.1f} MB",
                    end="",
                    flush=True,
                )
    print()
    return digest.hexdigest()


def inspect_class_map(checkpoint: Path) -> bool:
    from ultralytics import YOLO

    names = YOLO(str(checkpoint)).names or {}
    print("Class map:")
    for index in sorted(names):
        print(f"  {index}: {names[index]}")
    lowered = {str(name).strip().lower() for name in names.values()}
    people_ok = bool(lowered & PERSON_CLASS_NAMES)
    ball_ok = bool(lowered & BALL_CLASS_NAMES)
    if not people_ok:
        print(
            "WARNING: no person-like class names found — the pipeline would "
            "fall back to COCO ids, which is wrong for a football model. "
            "Report this class map before using the checkpoint."
        )
    if not ball_ok:
        print("Note: no ball class — generic ball candidates stay COCO-only.")
    return people_ok


def install(source: Path) -> None:
    root = _repo_root()
    baked = root / "apps" / "api" / "models" / "football.pt"
    baked.parent.mkdir(parents=True, exist_ok=True)
    baked.write_bytes(source.read_bytes())
    dev_link = root / "models" / "football.pt"
    dev_link.parent.mkdir(parents=True, exist_ok=True)
    if dev_link.is_symlink() or dev_link.exists():
        dev_link.unlink()
    dev_link.symlink_to(Path("..") / "apps" / "api" / "models" / "football.pt")
    print(f"Installed: {baked}")
    print(f"Dev symlink: {dev_link} -> {dev_link.readlink()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help="Checkpoint filename inside the SoccerMaster repository "
        f"(default: {DEFAULT_FILE}; the light alternative is "
        "SoccerNetGSR_Detection.pt)",
    )
    parser.add_argument(
        "--from-path",
        type=Path,
        default=None,
        help="Skip the download and install an already-downloaded file",
    )
    arguments = parser.parse_args()

    if arguments.from_path is not None:
        source = arguments.from_path.expanduser().resolve()
        if not source.is_file():
            print(f"No such file: {source}")
            return 2
        print(f"SHA-256: {hashlib.sha256(source.read_bytes()).hexdigest()}")
    else:
        source = _repo_root() / ".weights-download" / arguments.file
        url = f"{REPOSITORY}/{arguments.file}"
        print(f"Downloading {url}")
        sha = download(url, source)
        print(f"SHA-256: {sha}")
    print(f"Size: {source.stat().st_size / 1e6:.1f} MB")

    if not inspect_class_map(source):
        return 3
    install(source)
    print(
        "\nNext steps (all three services build from apps/api and run "
        "reconstruction code):\n"
        "  docker compose build api reconstruction-runner pipeline-runner\n"
        "  docker compose up -d api reconstruction-runner pipeline-runner\n"
        "  (local dev runs pick models/football.pt up without a rebuild)\n"
        "Then select 'football · custom weights' in the People detector "
        "dropdown."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
