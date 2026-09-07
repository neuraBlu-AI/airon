#!/usr/bin/env python3
"""
Tuning bench for vision_service - NOT part of aiRon's face.

Shows the camera feed with what the tracker found drawn on top, plus the raw
numbers behind the smile and mouth-open mapping. Use it to tune thresholds;
use run_airon.py to see the face.

    python tools/vision_bench.py        q quit | d boxes | c recalibrate
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.core import EventBus, StateStore          # noqa: E402
from airon.vision import VisionService               # noqa: E402


def main() -> int:
    store, bus = StateStore(), EventBus()
    bus.subscribe(lambda e: print(f"[event] {e}"))

    vision = VisionService(store, bus, want_depth="--no-depth" not in sys.argv,
                           stream_depth=True)
    print(f"[bench] {vision.camera.name}, depth={vision.camera.has_depth}, "
          f"on-device detector={vision.camera.has_detector}")
    vision.start()

    debug = True
    try:
        while True:
            latest = vision.latest_frame
            if latest is None:
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
                continue

            frame, obs = latest
            view = frame.color.copy()
            person = store.get().primary

            # Boxes the camera's own tracker reports, with their persistent ids.
            for t in frame.tracks:
                x, y, w, h = t.bbox
                live = t.status in ("NEW", "TRACKED")
                cv2.rectangle(view, (x, y), (x + w, y + h),
                              (90, 220, 120) if live else (90, 140, 200), 2)
                label = f"#{t.id} {t.status}"
                if t.distance_m:
                    label += f" {t.distance_m:.2f}m"
                cv2.putText(view, label, (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (90, 220, 120), 1, cv2.LINE_AA)

            if debug and obs.found:
                for ex, ey, ew, eh in obs.debug.get("eyes", []):
                    cv2.rectangle(view, (ex, ey), (ex + ew, ey + eh), (230, 200, 90), 1)
                curve = obs.debug.get("mouth_curve")
                if curve is not None and len(curve) > 2:
                    cv2.polylines(view, [curve], False, (120, 140, 250), 2, cv2.LINE_AA)

            curv, base, open_raw = obs.debug.get("curv", (0.0, 0.0, 0.0))
            distance = f"{person.distance_m:.2f}m" if person and person.distance_m else "--"
            if person is None:
                who = "nobody"
            else:
                who = f"{person.id} = {person.name}" if person.name else person.id
            hud = [
                f"{vision.fps:4.1f} fps   depth {distance}   {who}",
                f"attention {obs.attention_x:+.2f},{obs.attention_y:+.2f}"
                f"  roll {np.degrees(obs.head_roll):+5.1f}",
                f"eyes {obs.eyes_open:.2f}  curve {obs.mouth_curve:+.2f}"
                f"  mouth {obs.mouth_open:.2f}",
                f"curv {curv:+.3f}  neutral {base:+.3f}  open {open_raw:.3f}",
            ]
            for i, line in enumerate(hud):
                cv2.putText(view, line, (12, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (235, 235, 235), 1, cv2.LINE_AA)

            if frame.depth is not None:
                depth_view = cv2.applyColorMap(
                    cv2.convertScaleAbs(frame.depth, alpha=0.06), cv2.COLORMAP_TURBO)
                depth_view = cv2.resize(depth_view, (view.shape[1] // 3, view.shape[0] // 3))
                view[-depth_view.shape[0]:, -depth_view.shape[1]:] = depth_view

            cv2.imshow("aiRon vision bench", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
            if key == ord("c"):
                vision.tracker.recalibrate()
                print("[bench] neutral face recalibrated")
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
