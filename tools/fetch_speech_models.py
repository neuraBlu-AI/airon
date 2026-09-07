#!/usr/bin/env python3
"""Download the models aiRon listens with: Silero VAD and Whisper (gitignored)."""

from __future__ import annotations

import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.audio.service import MODEL_DIR, VAD_MODEL        # noqa: E402
from airon.speech.listener import WHISPER                   # noqa: E402

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"


def download(url: str, target: Path) -> None:
    print(f"  {url.rsplit('/', 1)[-1]} ...", end="", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response, \
            open(target, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            print(".", end="", flush=True)
    print(f" {target.stat().st_size / 1e6:.0f} MB")


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if VAD_MODEL.exists():
        print("silero VAD: already present")
    else:
        print("silero VAD:")
        download(f"{RELEASES}/silero_vad.onnx", VAD_MODEL)

    whisper_dir = MODEL_DIR / f"sherpa-onnx-whisper-{WHISPER}"
    if whisper_dir.exists():
        print(f"whisper-{WHISPER}: already present")
        return 0

    # The archive carries fp32 and int8 side by side; aiRon runs the int8
    # pair, which is a third of the size and measurably faster on this CPU.
    print(f"whisper-{WHISPER} (multilingual, several hundred MB):")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "whisper.tar.bz2"
        download(f"{RELEASES}/sherpa-onnx-whisper-{WHISPER}.tar.bz2", archive)
        print("  extracting ...", flush=True)
        with tarfile.open(archive) as tar:
            tar.extractall(MODEL_DIR)
    print(f"whisper-{WHISPER}: {whisper_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
