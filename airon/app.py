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
import random
import sys
import time

from PySide6.QtWidgets import QApplication

from .core import EventBus, EventType, StateStore
from .face import FaceAnimator, FaceWindow
from .speech import SpeechService
from .vision import VisionService

#: Stand-in for brain_service, which does not exist yet. The cooldown stops
#: aiRon greeting the same person twice as they settle; it is deliberately
#: short now that presence tracking no longer treats a glance away as leaving.
GREET_COOLDOWN_S = 20.0

#: --name is a placeholder for face recognition (spec section 12). It greets
#: whoever appears by that name, which is simply wrong for anyone else - the
#: point is to prove the milestone's shape until embeddings give real identity.
GREETINGS = {
    "en": {
        "named": ["Hello {name}, nice to see you.", "Hi {name}, good to see you again."],
        "anon": ["Hello there.", "Hi, nice to see you."],
    },
    "de": {
        "named": ["Hallo {name}, schön dich zu sehen.", "Hallo {name}, schön dass du da bist."],
        "anon": ["Hallo!", "Schön dich zu sehen."],
    },
}


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
    parser.add_argument("--lang", default="en", choices=["en", "de"],
                        help="default voice language")
    parser.add_argument("--name", default=None,
                        help="greet by this name, until face recognition lands")
    parser.add_argument("--no-voice", action="store_true",
                        help="run silently, without speech_service")
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

    speech = None
    if not args.no_voice:
        speech = SpeechService(bus, default_lang=args.lang)
        if speech.available():
            speech.start()
            print(f"[aiRon] voice ready: {', '.join(speech.available())}")
        else:
            print("[aiRon] no voices installed - run tools/fetch_voices.py")
            speech = None

    app = QApplication(sys.argv[:1])
    animator = FaceAnimator(mirror=not args.no_mirror, speech=speech)

    last_greeting = 0.0

    def on_person(event):
        """Placeholder for brain_service: say hello to whoever turns up."""
        nonlocal last_greeting
        if speech is None or event.type is not EventType.PERSON_ENTERED:
            return
        now = time.monotonic()
        if now - last_greeting < GREET_COOLDOWN_S:
            return
        last_greeting = now
        lines = GREETINGS[args.lang]["named" if args.name else "anon"]
        speech.say(random.choice(lines).format(name=args.name), lang=args.lang)

    bus.subscribe(on_person)

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
        if speech is not None:
            speech.stop()
