"""
Every line aiRon prints, with the time it happened.

Durations were being inferred from the order lines appeared in, and that is
how a 0.4 s pause got blamed for a truncation that turned out to be whisper
dropping the last word. The order was right; the intervals were invented. A
log that carries its own clock would have settled it in one session instead of
three.

Elapsed seconds since the process started, so a duration is a subtraction:

       0.02 [aiRon] .env: ANTHROPIC_API_KEY
       4.51 [listener] whisper-small ready in 4.5s, listening in de
      12.30 [event] PERSON_ENTERED person=guest_001
      19.44 [event] KNOWN_PERSON_DETECTED person=person_001 name=André

Recognition took 7.14 s there, and nobody has to guess.

The tag stays inside the message rather than becoming an argument, so adopting
this was renaming print to log and nothing else - which matters when the point
is to add timestamps, not to rewrite thirty-five call sites.

Flushed on every line. Python block-buffers stdout when it is redirected to a
file, which is how a whole session's worth of output once sat invisible in a
buffer while the robot was being tested.
"""

from __future__ import annotations

import sys
import time

_START = time.monotonic()


def elapsed() -> float:
    """Seconds since aiRon started."""
    return time.monotonic() - _START


def log(message: str, *, err: bool = False) -> None:
    """One line, stamped. `message` carries its own [tag]."""
    print(f"{elapsed():8.2f} {message}",
          file=sys.stderr if err else sys.stdout, flush=True)
