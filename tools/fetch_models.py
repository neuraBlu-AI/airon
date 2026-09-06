#!/usr/bin/env python3
"""Download the MyriadX blobs aiRon runs on the camera (gitignored)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import blobconverter                                   # noqa: E402

from airon.vision.camera import FACE_BLOB, MODEL_DIR, SHAVES   # noqa: E402


def main() -> int:
    MODEL_DIR.mkdir(exist_ok=True)
    target = MODEL_DIR / FACE_BLOB
    if target.exists():
        print(f"{target} already present")
        return 0
    print(f"compiling/downloading face detector for {SHAVES} shaves ...", flush=True)
    path = blobconverter.from_zoo(name="face-detection-retail-0004",
                                  shaves=SHAVES, zoo_type="intel")
    shutil.copy(path, target)
    print(f"{target}  {target.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
