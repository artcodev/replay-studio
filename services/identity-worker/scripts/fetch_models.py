"""Fetch and checksum the two official SoccerNet PRTReID assets."""

from __future__ import annotations

from hashlib import md5
import os
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


ASSETS = (
    (
        "prtreid-soccernet-baseline.pth.tar",
        "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1",
        "9633825232bc89f23a94522c5561650e",
    ),
    (
        "hrnetv2_w32_imagenet_pretrained.pth",
        "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1",
        "58ea12b0420aa3adaa2f74114c9f9721",
    ),
)


def checksum(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url, expected in ASSETS:
        target = target_dir / filename
        if target.is_file() and checksum(target) == expected:
            print(f"verified {filename}")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        partial.unlink(missing_ok=True)
        request = Request(url, headers={"User-Agent": "ReplayStudio/1.0"})
        print(f"downloading {filename}")
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = checksum(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected}, received {actual}"
            )
        partial.replace(target)
        print(f"verified {filename}")


if __name__ == "__main__":
    fetch(Path(os.environ.get("REID_MODEL_DIRECTORY", "/models")))
