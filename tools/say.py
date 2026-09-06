#!/usr/bin/env python3
"""
Speak a line through aiRon's voice, without starting the robot.

    python tools/say.py "Hallo Pierre, schön dich zu sehen."
    python tools/say.py --lang en "Hello there."
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.core import EventBus                      # noqa: E402
from airon.speech import SpeechService, detect_language   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="speak a line as aiRon")
    ap.add_argument("text", nargs="+")
    ap.add_argument("--lang", choices=["en", "de"], default=None,
                    help="force a language instead of detecting it")
    args = ap.parse_args()

    text = " ".join(args.text)
    bus = EventBus()
    bus.subscribe(lambda e: print(f"[event] {e}"))
    speech = SpeechService(bus)
    if not speech.available():
        print("no voices installed - run tools/fetch_voices.py", file=sys.stderr)
        return 1

    print(f"language: {args.lang or detect_language(text)}")
    speech.start()
    speech.say(text, lang=args.lang)
    time.sleep(0.3)
    while speech.speaking or not speech._queue.empty():
        time.sleep(0.05)
    speech.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
