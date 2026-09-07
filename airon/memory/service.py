"""
memory_service - the last of the services in spec section 7.

It listens to the same bus everything else does and keeps three things:

  short-term   the current exchange, in memory only, never written down
  person       who somebody is, how often they come, what language they speak
  episodic     distilled things worth keeping, written by the brain

What it deliberately does not do is decide that something is worth
remembering. Nothing here can read a sentence and tell a fact from small talk;
that judgement belongs to brain_service, which is where the language model
will eventually live. This module owns *how* memory behaves - what decays,
what deduplicates, what a visit is - and exposes remember() for the part of
the system that can actually judge.

So on its own it stores almost nothing, and that is the correct amount. Spec
section 11: memory should be selective and relevance-based. A memory service
that wrote down every sentence it overheard would be easier to build and worse
at its job.

The one judgement it does make is that seeing somebody is not an event. Visits
are counted, not narrated.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from ..core import EventBus, EventType
from .store import MemoryStore

#: Turns of the current exchange kept for context. Short: this is what is
#: being said right now, not a log.
SHORT_TERM_TURNS = 16

#: An exchange this quiet is over, and the short-term buffer is stale.
CONVERSATION_IDLE_S = 180.0

#: Memories offered to the brain per person, strongest first.
RECALL_LIMIT = 6

#: Someone not seen for this long has been away rather than in the next room.
LONG_ABSENCE_S = 3 * 86400.0

#: Spoken names for the languages aiRon speaks, so a context block reads as a
#: sentence rather than as a config value.
LANGUAGES = {"de": "German", "en": "English"}


def describe_gap(seconds: float) -> str:
    """A human way of saying how long ago, for a line aiRon has to speak."""
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return "an hour ago" if hours == 1 else f"{hours} hours ago"
    days = int(seconds // 86400)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days} days ago"
    return time.strftime("%-d %B", time.localtime(time.time() - seconds))


class MemoryService:
    """
    Remembers people between runs, and hands the brain what it knows.

    Everything is synchronous and small; there is no thread here because there
    is no work slow enough to need one. Writes go to disk when something
    actually changed, the way the identity gallery does.
    """

    def __init__(self, bus: EventBus, *, path: Path | str | None = None):
        self.bus = bus
        self.store = MemoryStore(path)
        self._turns: deque[tuple[str, str]] = deque(maxlen=SHORT_TERM_TURNS)
        self._last_turn = 0.0
        self._current: str | None = None
        # Noting a visit moves last_met to now, which erases the very gap the
        # brain needs in order to say "it has been a while". So the gap is
        # captured at the moment of arrival and kept until they leave.
        self._arrival: tuple[str, int, float] | None = None
        bus.subscribe(self._on_event)

    # --------------------------------------------------------------- events

    def _on_event(self, event) -> None:
        kind = event.type
        if kind in (EventType.KNOWN_PERSON_DETECTED, EventType.PERSON_NAMED):
            self._arrived(event.payload.get("person"), event.payload.get("name", ""))
        elif kind is EventType.HEARD:
            self._heard(event.payload.get("text", ""), event.payload.get("lang", ""))
        elif kind is EventType.SPEECH_STARTED:
            self._said(event.payload.get("text", ""))
        elif kind is EventType.PERSON_LEFT:
            if event.payload.get("person") == self._current:
                self._current = None
                self._arrival = None

    def _arrived(self, person_id: str | None, name: str) -> None:
        if not person_id:
            return
        if self._arrival is not None and self._arrival[0] == person_id:
            return                       # same arrival, seen twice on the bus
        self._current = person_id
        visits, gap = self.store.visit(person_id, name)
        self._arrival = (person_id, visits, gap)
        if visits == 1 or gap > CONVERSATION_IDLE_S:
            self._turns.clear()          # a new exchange, not a continuation
        self.store.save()

    def _heard(self, text: str, lang: str) -> None:
        if not text:
            return
        self._note("them", text)
        # Which language somebody actually speaks is worth knowing and costs
        # nothing to observe - spec section 11 lists it under person memory.
        if lang and self._current:
            person = self.store.person(self._current)
            if person.language != lang:
                person.language = lang
                self.store.save()

    def _said(self, text: str) -> None:
        if text:
            self._note("airon", text)

    def _note(self, speaker: str, text: str) -> None:
        now = time.time()
        if now - self._last_turn > CONVERSATION_IDLE_S:
            self._turns.clear()
        self._last_turn = now
        self._turns.append((speaker, " ".join(text.split())))

    # --------------------------------------------------------------- public

    @property
    def current_person(self) -> str | None:
        return self._current

    def conversation(self) -> list[tuple[str, str]]:
        """The exchange so far. In memory only; this is never written down."""
        if time.time() - self._last_turn > CONVERSATION_IDLE_S:
            self._turns.clear()
        return list(self._turns)

    def remember(self, text: str, *, person_id: str | None = None,
                 kind: str = "episodic", importance: float = 0.5):
        """Store one distilled thing. The caller decides it was worth it."""
        memory = self.store.remember(text, person_id=person_id, kind=kind,
                                     importance=importance)
        if memory is not None:
            self.store.save()
        return memory

    def acquaintance(self, person_id: str) -> tuple[int, float]:
        """
        (visits, seconds since they were previously here) - enough to choose
        between "hello", "good to see you again" and "it has been a while".

        The gap is the one measured when they arrived, not the one now, which
        noting the visit has already reset to zero.
        """
        if self._arrival is not None and self._arrival[0] == person_id:
            return self._arrival[1], self._arrival[2]
        person = self.store.people.get(person_id)
        if person is None:
            return 0, 0.0
        return person.visits, max(time.time() - person.last_met, 0.0)

    def context_for(self, person_id: str | None, name: str = "") -> str:
        """
        A short block describing who this is, for the brain to reason over.

        Deliberately prose rather than JSON: it exists to be read by a
        language model, and it is short enough to read at a glance when
        debugging what aiRon thinks it knows.
        """
        if not person_id:
            return "Somebody aiRon has not met before."

        person = self.store.people.get(person_id)
        if person is None:
            return f"{name or 'Someone'}, not seen before."

        who = person.name or name or person.person_id
        lines = [f"{who}. Met {person.visits} "
                 f"{'time' if person.visits == 1 else 'times'}, first on "
                 f"{time.strftime('%-d %B %Y', time.localtime(person.first_met))}, "
                 f"last seen {describe_gap(time.time() - person.last_met)}."]
        if person.language:
            lines.append(f"Speaks {LANGUAGES.get(person.language, person.language)}.")

        memories = self.store.recall(person_id, limit=RECALL_LIMIT)
        if memories:
            lines.append("Known about them:")
            lines += [f"- {m.text}" for m in memories]
        return "\n".join(lines)

    def forget(self, person_id: str) -> bool:
        return self.store.forget_person(person_id)
