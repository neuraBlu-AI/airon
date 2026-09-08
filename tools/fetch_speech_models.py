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
from airon.speech.listener import GGML_DIR, WHISPER         # noqa: E402

RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

#: The same Whisper weights in ggml form, for the engine that runs on the GPU.
GGML = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"


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

    # Both engines run the same weights in different formats, and which one
    # aiRon uses depends on whether whisper.cpp was built against CUDA. Fetch
    # both: together they are under a gigabyte, and a robot that will not
    # listen because the wrong format is on disk is a bad trade for the space.
    GGML_DIR.mkdir(parents=True, exist_ok=True)
    ggml = GGML_DIR / f"ggml-{WHISPER}.bin"
    if ggml.exists():
        print(f"whisper-{WHISPER} (ggml, for the GPU): already present")
    else:
        print(f"whisper-{WHISPER} (ggml, for the GPU):")
        download(f"{GGML}/ggml-{WHISPER}.bin", ggml)

    whisper_dir = MODEL_DIR / f"sherpa-onnx-whisper-{WHISPER}"
    if whisper_dir.exists():
        print(f"whisper-{WHISPER} (onnx, for the CPU): already present")
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
