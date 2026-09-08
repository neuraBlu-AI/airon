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
from .core import EventBus, EventType, StateStore, load_env, log
from .face import FaceAnimator, FaceWindow
from .face.weathercard import WeatherCard
from .face.overlay import build as build_overlays
from .memory import MemoryService
from .speech import ENGINES, Listener, SpeechService
from .world import Weather
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
    parser.add_argument("--asr", default="auto", choices=list(ENGINES),
                        help="speech recognition engine: whisper.cpp on the GPU, "
                             "sherpa-onnx on the CPU, or whichever is installed")
    parser.add_argument("--no-memory", action="store_true",
                        help="do not remember anything between runs")
    parser.add_argument("--no-llm", action="store_true",
                        help="stay scripted; do not use the language model")
    parser.add_argument("--brain", default="cloud", choices=["cloud", "local"],
                        help="where the conversation is thought: the Anthropic API, "
                             "or a model on this robot's own GPU (AIRON-9)")
    parser.add_argument("--debug", action="store_true",
                        help="start with the state overlay visible")
    parser.add_argument("--preview", action="store_true",
                        help="show what the camera sees, small, in the corner (key P)")
    parser.add_argument("--transcript", action="store_true",
                        help="show the conversation as text on screen (key T)")
    parser.add_argument("--inspect", action="store_true",
                        help="everything useful while testing: --debug --preview --transcript")
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
        log("[aiRon] no voice-activity model - run tools/fetch_speech_models.py")
        return None, None

    listener = Listener(bus, ears, lang=args.lang, engine=args.asr)
    if not listener.available():
        log("[aiRon] no speech recognition model - run tools/fetch_speech_models.py")
        return None, None

    ears.start()
    listener.start()
    return ears, listener


def main(argv=None) -> int:
    args = parse_args(argv)

    # Before anything asks for a key. The brain decides whether it has one
    # the moment it is constructed, so a .env read later than this is a .env
    # that does nothing.
    loaded = load_env()
    if loaded:
        log(f"[aiRon] .env: {', '.join(sorted(loaded))}")

    store = StateStore()
    bus = EventBus()
    weather = Weather()
    if weather.configured():
        log(f"[aiRon] weather: {weather.place or weather.latlon}")
    else:
        log("[aiRon] no weather - set AIRON_WEATHER_PLACE in .env")
    bus.subscribe(lambda event: log(f"[event] {event}"))

    try:
        vision = VisionService(store, bus, fps=args.fps,
                               want_depth=not args.no_depth,
                               force_depth=args.force_depth,
                               depth_fps=args.depth_fps,
                               recognise=not args.no_recognition)
    except Exception as exc:
        log(f"[aiRon] vision failed to start: {exc}", err=True)
        return 1

    # The link speed is worth saying out loud every time. This OAK-D Lite is
    # USB 3 hardware that has been seen negotiating HIGH, which is a cable
    # rather than a setting, and is invisible unless something prints it.
    log(f"[aiRon] eyes online: {vision.camera.name}"
          f"{' with depth' if vision.camera.has_depth else ' (no depth)'}"
          f", link {vision.camera.usb_speed}")
    if vision.recognizer is None:
        log("[aiRon] face recognition off - everyone will be a guest")
    else:
        known = vision.gallery.names()
        log(f"[aiRon] knows {len(known)} "
              f"{'person' if len(known) == 1 else 'people'}"
              f"{': ' + ', '.join(known) if known else ' - run tools/enroll_face.py'}")

    speech = None
    if not args.no_voice:
        speech = SpeechService(bus, default_lang=args.lang,
                               length_scale=args.speech_rate)
        if speech.available():
            speech.start()
            log(f"[aiRon] voice ready: {', '.join(speech.available())}")
        else:
            log("[aiRon] no voices installed - run tools/fetch_voices.py")
            speech = None

    # Before the brain, deliberately: handlers run in subscription order, and
    # the brain asks memory how long since it last saw you. Memory has to have
    # noted the arrival before that question is worth asking.
    memory = None if args.no_memory else MemoryService(bus)
    if memory is not None:
        counts = memory.store.stats()
        log(f"[aiRon] remembers {counts['people']} "
              f"{'person' if counts['people'] == 1 else 'people'}, "
              f"{counts['person_memories']} things about them")

    ears, listener = start_hearing(bus, args, speech)

    app = QApplication(sys.argv[:1])
    animator = FaceAnimator(mirror=not args.no_mirror, speech=speech)

    conversation = None
    if not args.no_llm:
        if args.brain == "local":
            from .brain.local import LocalConversation
            conversation = LocalConversation(lang=args.lang)
        else:
            conversation = Conversation(lang=args.lang)
        if conversation.available():
            # The whole configuration, not just the model: these are tuning
            # knobs now, and a recording of aiRon sounding good is only worth
            # anything if the log says what it was configured with at the time.
            log(f"[aiRon] conversation: {conversation.summary}")
            if args.brain == "local":
                # Half a minute of weights, paid now rather than by whoever
                # happens to say hello first.
                log("[aiRon] warming the local brain ...")
                log(f"[aiRon] local brain ready in {conversation.warm():.1f}s")
        else:
            log(f"[aiRon] no conversation - {conversation.last_error}. "
                  "aiRon will still greet people and ask names.")
            conversation = None

    brain = BrainService(bus, speech=speech, vision=vision, face=animator,
                         memory=memory, conversation=conversation, lang=args.lang,
                         can_listen=listener is not None,
                         ears=ears, listener=listener, store=store,
                         weather=weather)
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
    # Test panels only. aiRon behaves identically whether or not they are on;
    # they are built either way so a key can still reveal one mid-session.
    overlays = build_overlays(bus, vision,
                              preview=args.preview or args.inspect,
                              transcript=args.transcript or args.inspect)
    # Not a test panel: this one is aiRon showing somebody something, so it
    # is on whenever there is a forecast to show and off the rest of the time.
    overlays.append(WeatherCard(weather, lang=args.lang))

    # Only now, once everything that listens has been built. Vision publishes
    # PERSON_ENTERED and KNOWN_PERSON_DETECTED the instant it sees a face, and
    # an event published before its subscribers exist is not delivered late,
    # it is not delivered at all.
    #
    # Somebody already standing in front of the robot when it was switched on -
    # which is how one switches on a robot - was recognised in under five
    # seconds, before the brain had been constructed. The brain learns who is
    # in the room only from those events, and a person who stays in frame never
    # arrives a second time, so it spent the whole session believing the room
    # was empty: hearing every word, transcribing it correctly, and dropping
    # all of it in silence. Being a race, it came and went with how fast
    # whisper happened to load that morning, which is what made one bug look
    # like several.
    #
    # This is the last line of the setup for that reason. Anything added below
    # it that subscribes to the bus is a subscriber that can miss an arrival.
    vision.start()
    window = FaceWindow(store, animator, vision=vision,
                        fullscreen=not args.windowed, overlays=overlays)
    window.show_debug = args.debug or args.inspect
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
