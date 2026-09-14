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
from collections import deque

_START = time.monotonic()

#: What a line looks like when something has gone wrong. Matched on the text
#: rather than declared at the call site, because the alternative is thirty-
#: five call sites growing an argument they do not otherwise need - the same
#: reasoning that keeps the [tag] inside the message.
#:
#: It will over-match, and that is the right way round: a line kept that did
#: not need keeping costs twenty characters of a ring buffer, and a line
#: missed is a fault aiRon cannot tell anybody about (AIRON-32).
TROUBLE = ("failed", "could not", "cannot", "error", "gave up", "giving up",
           "no such", "stopped delivering", "went blind", "too slow",
           "unreadable", "refused", "crash", "ignoring")

#: Short on purpose. This is what aiRon can say has recently gone wrong with
#: it, not a log file - the log file is the log file. Twenty lines is about
#: one bad minute, which is the span somebody in the room is asking about.
_trouble: deque[str] = deque(maxlen=20)


def recent_trouble(limit: int = 10) -> list[str]:
    """The last few things that went wrong, newest last.

    Kept in memory only, and only for this run: aiRon noticing that its own
    microphone dropped out two minutes ago is the point, and remembering it
    across a restart is not - by then the thing has either been fixed or is
    still happening, and either way it will say so again.
    """
    return list(_trouble)[-limit:]


def elapsed() -> float:
    """Seconds since aiRon started."""
    return time.monotonic() - _START


def log(message: str, *, err: bool = False) -> None:
    """One line, stamped. `message` carries its own [tag]."""
    stamped = f"{elapsed():8.2f} {message}"
    lowered = message.lower()
    if err or any(word in lowered for word in TROUBLE):
        _trouble.append(stamped.strip())
    print(stamped, file=sys.stderr if err else sys.stdout, flush=True)
