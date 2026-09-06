"""
Wiring for Phase 1.

    CAMERA -> vision_service -> WorldState + events
                                   |
                                   v
                              face_service

brain_service, speech_service and memory_service are not built yet; for now a
console subscriber stands in for the brain so the event stream is visible.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .core import EventBus, EventType, StateStore
from .face import FaceAnimator, FaceWindow
from .vision import VisionService


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="airon", description="aiRon - Phase 1 brain")
    parser.add_argument("--windowed", action="store_true",
                        help="run in a window instead of fullscreen on the face display")
    parser.add_argument("--no-depth", action="store_true",
                        help="skip the stereo pair (RGB only)")
    parser.add_argument("--force-depth", action="store_true",
                        help="attempt stereo even on a USB 2 link (may crash the OAK)")
    parser.add_argument("--no-mirror", action="store_true",
                        help="start in emotion mode instead of mirroring the human")
    parser.add_argument("--fps", type=int, default=30, help="camera frame rate")
    parser.add_argument("--depth-fps", type=int, default=None,
                        help="stereo frame rate (default 10; higher has crashed this device)")
    parser.add_argument("--debug", action="store_true",
                        help="start with the state overlay visible")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    store = StateStore()
    bus = EventBus()
    bus.subscribe(lambda event: print(f"[event] {event}"))

    try:
        vision = VisionService(store, bus, fps=args.fps,
                               want_depth=not args.no_depth,
                               force_depth=args.force_depth,
                               depth_fps=args.depth_fps)
    except Exception as exc:
        print(f"[aiRon] vision failed to start: {exc}", file=sys.stderr)
        return 1

    print(f"[aiRon] eyes online: {vision.camera.name}"
          f"{' with depth' if vision.camera.has_depth else ' (no depth)'}")
    vision.start()

    app = QApplication(sys.argv[:1])
    animator = FaceAnimator(mirror=not args.no_mirror)

    def on_camera(event):
        """Losing the camera should be visible on aiRon's face, not just in a log."""
        if event.type is EventType.CAMERA_LOST:
            animator.mirror = False
            animator.command.emotion = "sleepy"
        elif event.type is EventType.CAMERA_READY:
            animator.command.emotion = "curious"
            animator.mirror = not args.no_mirror

    bus.subscribe(on_camera)
    window = FaceWindow(store, animator, vision=vision, fullscreen=not args.windowed)
    window.show_debug = args.debug
    window.setWindowTitle("aiRon")

    try:
        return app.exec()
    finally:
        vision.stop()
