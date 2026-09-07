#!/usr/bin/env python3
"""Download the MyriadX blobs aiRon runs on the camera (gitignored)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import blobconverter                                   # noqa: E402

from airon.vision.camera import (FACE_BLOB, LANDMARK_BLOB, LANDMARK_SHAVES,   # noqa: E402
                                 MODEL_DIR, REID_BLOB, REID_SHAVES, SHAVES)

#: filename -> (Intel open-model-zoo name, shaves it is compiled for).
BLOBS = {
    FACE_BLOB: ("face-detection-retail-0004", SHAVES),
    LANDMARK_BLOB: ("landmarks-regression-retail-0009", LANDMARK_SHAVES),
    REID_BLOB: ("face-reidentification-retail-0095", REID_SHAVES),
}


def main() -> int:
    MODEL_DIR.mkdir(exist_ok=True)
    for filename, (name, shaves) in BLOBS.items():
        target = MODEL_DIR / filename
        if target.exists():
            print(f"{name}: already present")
            continue
        print(f"{name}: compiling/downloading for {shaves} shaves ...", flush=True)
        path = blobconverter.from_zoo(name=name, shaves=shaves, zoo_type="intel")
        shutil.copy(path, target)
        print(f"{name}: {target.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
