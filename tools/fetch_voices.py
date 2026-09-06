#!/usr/bin/env python3
"""Download the Piper voices aiRon speaks with (~61 MB each, gitignored)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper.download_voices import download_voice     # noqa: E402

from airon.speech.service import VOICES, VOICE_DIR   # noqa: E402


def main() -> int:
    VOICE_DIR.mkdir(exist_ok=True)
    for lang, name in VOICES.items():
        if (VOICE_DIR / f"{name}.onnx").exists():
            print(f"{lang}: {name} already present")
            continue
        print(f"{lang}: downloading {name} ...", flush=True)
        download_voice(name, VOICE_DIR)
    print(f"voices in {VOICE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
