"""
Wiring for Phase 1.

    CAMERA ---> vision_service ---> WorldState + events ---> brain_service
                                          |                    |
    MICS -----> audio_service ---> speech_service (ears)       |
                                                               v
                                       face_service  <---  speech_service (voice)

Every service owns a thread and talks over the bus; nothing here does any
work of its own beyond deciding what exists. memory_service is the one piece
of spec section 7 still missing - for now the only thing aiRon remembers
between runs is who people are, which the identity gallery holds.
"""

from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .audio import AudioService
from .brain import BrainService
from .brain.conversation import Conversation
from .core import EventBus, EventType, StateStore
from .face import FaceAnimator, FaceWindow
from .memory import MemoryService
from .speech import Listener, SpeechService
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
    parser.add_argument("--lang", default="en", choices=["en", "de"],
                        help="default voice language")
    parser.add_argument("--no-recognition", action="store_true",
                        help="skip face recognition; everyone stays a guest")
    parser.add_argument("--speech-rate", type=float, default=1.05,
                        help="Piper length_scale: 1.0 natural pace, higher is slower")
    parser.add_argument("--no-voice", action="store_true",
                        help="run silently, without speech_service")
    parser.add_argument("--no-ears", action="store_true",
                        help="skip the microphone array and speech recognition")
    parser.add_argument("--no-memory", action="store_true",
                        help="do not remember anything between runs")
    parser.add_argument("--no-llm", action="store_true",
                        help="stay scripted; do not use the language model")
    parser.add_argument("--debug", action="store_true",
                        help="start with the state overlay visible")
    return parser.parse_args(argv)


def start_hearing(bus, args, speech):
    """
    Bring up the microphone array and speech recognition, if we can.

    Both are optional: aiRon without ears still sees, recognises and speaks,
    it just cannot ask a stranger who they are. Returning (None, None) is a
    normal outcome, not an error.
    """
    if args.no_ears:
        return None, None

    # aiRon must not hear itself. The array can cancel its own echo, but only
    # given the played audio as a reference, and aiRon's voice goes out of the
    # display's speakers where the array never hears about it.
    ears = AudioService(bus, is_muted=(lambda: speech.speaking) if speech else None)
    if not ears.available():
        print("[aiRon] no voice-activity model - run tools/fetch_speech_models.py")
        return None, None

    listener = Listener(bus, ears, lang=args.lang)
    if not listener.available():
        print("[aiRon] no speech recognition model - run tools/fetch_speech_models.py")
        return None, None

    ears.start()
    listener.start()
    return ears, listener


def main(argv=None) -> int:
    args = parse_args(argv)

    store = StateStore()
    bus = EventBus()
    bus.subscribe(lambda event: print(f"[event] {event}"))

    try:
        vision = VisionService(store, bus, fps=args.fps,
                               want_depth=not args.no_depth,
                               force_depth=args.force_depth,
                               depth_fps=args.depth_fps,
                               recognise=not args.no_recognition)
    except Exception as exc:
        print(f"[aiRon] vision failed to start: {exc}", file=sys.stderr)
        return 1

    print(f"[aiRon] eyes online: {vision.camera.name}"
          f"{' with depth' if vision.camera.has_depth else ' (no depth)'}")
    if vision.recognizer is None:
        print("[aiRon] face recognition off - everyone will be a guest")
    else:
        known = vision.gallery.names()
        print(f"[aiRon] knows {len(known)} "
              f"{'person' if len(known) == 1 else 'people'}"
              f"{': ' + ', '.join(known) if known else ' - run tools/enroll_face.py'}")
    vision.start()

    speech = None
    if not args.no_voice:
        speech = SpeechService(bus, default_lang=args.lang,
                               length_scale=args.speech_rate)
        if speech.available():
            speech.start()
            print(f"[aiRon] voice ready: {', '.join(speech.available())}")
        else:
            print("[aiRon] no voices installed - run tools/fetch_voices.py")
            speech = None

    # Before the brain, deliberately: handlers run in subscription order, and
    # the brain asks memory how long since it last saw you. Memory has to have
    # noted the arrival before that question is worth asking.
    memory = None if args.no_memory else MemoryService(bus)
    if memory is not None:
        counts = memory.store.stats()
        print(f"[aiRon] remembers {counts['people']} "
              f"{'person' if counts['people'] == 1 else 'people'}, "
              f"{counts['person_memories']} things about them")

    ears, listener = start_hearing(bus, args, speech)

    app = QApplication(sys.argv[:1])
    animator = FaceAnimator(mirror=not args.no_mirror, speech=speech)

    conversation = None
    if not args.no_llm:
        conversation = Conversation(lang=args.lang)
        if conversation.available():
            print(f"[aiRon] conversation: {conversation.model}")
        else:
            print(f"[aiRon] no conversation - {conversation.last_error}. "
                  "aiRon will still greet people and ask names.")
            conversation = None

    brain = BrainService(bus, speech=speech, vision=vision, face=animator,
                         memory=memory, conversation=conversation, lang=args.lang,
                         can_listen=listener is not None)
    brain.start()

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
        brain.stop()
        if listener is not None:
            listener.stop()
        if ears is not None:
            ears.stop()
        vision.stop()
        if speech is not None:
            speech.stop()
