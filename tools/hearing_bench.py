#!/usr/bin/env python3
"""
Can aiRon hear you? - NOT part of the robot.

    python tools/hearing_bench.py            English
    python tools/hearing_bench.py --lang de  German

Shows the live microphone level, every utterance the voice detector cuts out,
what Whisper made of it, and what the name parser would take from it. Use it
to check the array before blaming the conversation.

The language is pinned rather than detected, exactly as it is in aiRon, so
run it with the language you intend to speak. A German sentence transcribed
as English comes back as "[speaking in foreign language]".
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.audio import AudioService, find_capture_device   # noqa: E402
from airon.brain.naming import is_refusal, parse_name       # noqa: E402
from airon.core import EventBus, EventType                  # noqa: E402
from airon.speech import Listener                           # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hearing_bench")
    parser.add_argument("--lang", default="en", choices=["en", "de"])
    parser.add_argument("--device", default=None, help="ALSA capture device")
    args = parser.parse_args(argv)

    bus = EventBus()

    def on_event(event):
        if event.type is EventType.HEARD:
            text = event.payload["text"]
            name = parse_name(text)
            verdict = (f"name -> {name!r}" if name
                       else "declined" if is_refusal(text) else "no name in that")
            print(f"\r  [{event.payload['seconds']:4.1f}s] {text!r}\n"
                  f"           {verdict}", flush=True)
        elif event.type is EventType.VOICE_STARTED:
            print("\r  (speech)   ", end="", flush=True)

    bus.subscribe(on_event)

    ears = AudioService(bus, device=args.device or find_capture_device())
    if not ears.available():
        print("no voice-activity model - run tools/fetch_speech_models.py", file=sys.stderr)
        return 1
    listener = Listener(bus, ears, lang=args.lang)
    if not listener.available():
        print("no speech recognition model - run tools/fetch_speech_models.py",
              file=sys.stderr)
        return 1

    print(f"device {ears.device}, listening in {args.lang}. Speak; Ctrl-C to stop.")
    listener.start()
    ears.start()
    try:
        while True:
            time.sleep(0.1)
            bar = "#" * min(int(ears.level * 120), 40)
            print(f"\r  {ears.level:.3f} {bar:<40}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        ears.stop()
        listener.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
