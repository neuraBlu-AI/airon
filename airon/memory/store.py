"""
What aiRon remembers about people, and how it forgets.

Spec section 11 asks for four kinds of memory and then sets the hard
constraint: *the system should NOT blindly store every conversation. Memory
should be selective and relevance-based.* Storage is the easy half. This
module is mostly about what never gets written down, and what quietly stops
being true.

Three rules do that work:

**Counters, not stories.** Seeing somebody is not an event worth remembering.
Visits are a number and two timestamps on the person, so a hundred sightings
cost one line, and "I have met you three times" stays answerable without a
hundred memories saying "saw André".

**A memory that repeats itself is one memory.** Somebody who mentions their
dog on Monday and again on Friday has one fact about a dog, not two. A near
duplicate refreshes and strengthens what is already there instead of landing
beside it.

**Strength decays, recall renews it.** Every memory has an importance and a
last-touched time, and its strength is the one decayed by the other. Recalling
something touches it, so what gets used survives and what never comes up again
fades below the floor and is dropped. That is a deliberate design choice rather
than a storage limit: a companion that remembers every passing remark with
equal weight forever is not a good memory, it is a transcript.

Storage follows the identity gallery: one JSON file per person, no recordings,
no transcripts. Deleting the file really is the whole of deleting the person.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from ..core.log import log

MEMORY_DIR = Path(__file__).resolve().parent.parent.parent / "memories"
WORLD_FILE = "world.json"

#: How long a memory of average importance takes to lose half its strength.
#: Long enough that something mentioned last week is still there, short enough
#: that a passing remark does not outlive its usefulness.
HALF_LIFE_DAYS = 30.0

#: Below this strength a memory is not worth keeping, and is dropped on save.
FORGET_BELOW = 0.08

#: Ceiling per person. Reached only by someone with a great many distinct
#: things said about them; the weakest go first.
MAX_PER_PERSON = 200

#: Word overlap above which two memories are the same memory.
DUPLICATE_OVERLAP = 0.6

#: Two sightings closer together than this are one visit. Somebody walking in
#: and out of frame has not visited twice.
VISIT_GAP_S = 30 * 60.0

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class Memory:
    """One thing worth remembering. Distilled, never a transcript."""

    text: str
    kind: str = "episodic"          # episodic | fact | preference
    importance: float = 0.5         # 0..1, how much it mattered when stored
    created: float = field(default_factory=time.time)
    touched: float = field(default_factory=time.time)
    recalled: int = 0

    def strength(self, now: float | None = None) -> float:
        """Importance, decayed by how long since anything touched this."""
        now = now if now is not None else time.time()
        age_days = max(now - self.touched, 0.0) / 86400.0
        return self.importance * math.exp(-age_days * math.log(2) / HALF_LIFE_DAYS)

    def touch(self, now: float | None = None) -> None:
        self.touched = now if now is not None else time.time()
        self.recalled += 1


@dataclass
class Person:
    """
    Everything aiRon knows about one person that is not their face.

    `person_id` is the identity gallery's id, so the two records are the same
    person without either module having to know about the other.
    """

    person_id: str
    name: str = ""
    language: str = ""
    first_met: float = field(default_factory=time.time)
    last_met: float = field(default_factory=time.time)
    visits: int = 0
    memories: list[Memory] = field(default_factory=list)

    def to_dict(self) -> dict:
        record = asdict(self)
        record["memories"] = [asdict(m) for m in self.memories]
        return record

    @classmethod
    def from_dict(cls, record: dict) -> "Person":
        memories = [Memory(**m) for m in record.pop("memories", [])]
        known = {f: record[f] for f in
                 ("person_id", "name", "language", "first_met", "last_met", "visits")
                 if f in record}
        return cls(memories=memories, **known)


def _words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD.findall(text) if len(w) > 2}


def overlap(a: str, b: str) -> float:
    """Jaccard similarity over words - crude, cheap, and good enough to spot
    the same sentence said twice."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class MemoryStore:
    """
    The memories on disk, readable and writable from any thread.

    One file per person plus one for the world, mirroring the gallery, so a
    person's entire record can be read in a text editor and removed with rm.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else MEMORY_DIR
        self._lock = threading.Lock()
        self.people: dict[str, Person] = {}
        self.world: list[Memory] = []
        self.load()

    # ------------------------------------------------------------------ io

    def load(self) -> None:
        people: dict[str, Person] = {}
        world: list[Memory] = []
        if self.path.exists():
            for file in sorted(self.path.glob("*.json")):
                try:
                    record = json.loads(file.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    log(f"[memory] cannot read {file.name}: {exc}")
                    continue
                if file.name == WORLD_FILE:
                    world = [Memory(**m) for m in record.get("memories", [])]
                else:
                    try:
                        person = Person.from_dict(record)
                    except TypeError as exc:
                        log(f"[memory] {file.name} is not a person record: {exc}")
                        continue
                    people[person.person_id] = person
        with self._lock:
            self.people, self.world = people, world

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            people = list(self.people.values())
            world = list(self.world)
        for person in people:
            (self.path / f"{person.person_id}.json").write_text(
                json.dumps(person.to_dict(), indent=2, ensure_ascii=False) + "\n")
        (self.path / WORLD_FILE).write_text(
            json.dumps({"memories": [asdict(m) for m in world]},
                       indent=2, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------- people

    def person(self, person_id: str, name: str = "") -> Person:
        """The record for this person, created on first sight."""
        with self._lock:
            record = self.people.get(person_id)
            if record is None:
                record = Person(person_id=person_id, name=name, visits=0)
                self.people[person_id] = record
            elif name and record.name != name:
                record.name = name
            return record

    def by_name(self, name: str) -> Person | None:
        wanted = name.strip().casefold()
        with self._lock:
            for person in self.people.values():
                if person.name.casefold() == wanted:
                    return person
        return None

    def visit(self, person_id: str, name: str = "") -> tuple[int, float]:
        """
        Note that somebody is here.

        Returns (visits, seconds since the last visit). Sightings closer
        together than VISIT_GAP_S are the same visit, so stepping out of frame
        and back does not inflate the count.
        """
        person = self.person(person_id, name)
        now = time.time()
        with self._lock:
            gap = now - person.last_met
            if person.visits == 0 or gap > VISIT_GAP_S:
                person.visits += 1
            person.last_met = now
            return person.visits, gap

    def forget_person(self, person_id: str) -> bool:
        """Delete somebody's entire record - the file and everything in it."""
        with self._lock:
            gone = self.people.pop(person_id, None) is not None
        (self.path / f"{person_id}.json").unlink(missing_ok=True)
        return gone

    # ------------------------------------------------------------ remember

    def remember(self, text: str, *, person_id: str | None = None,
                 kind: str = "episodic", importance: float = 0.5) -> Memory | None:
        """
        Store one distilled thing. Returns the memory, new or strengthened.

        A near duplicate does not land beside what is already there: it
        refreshes it and raises its importance, because something said twice
        matters more than something said once, not twice as often.
        """
        text = " ".join(text.split())
        if len(text) < 3:
            return None

        with self._lock:
            shelf = (self.people[person_id].memories
                     if person_id is not None and person_id in self.people
                     else self.world if person_id is None else None)
        if shelf is None:                       # unknown person: make a record
            shelf = self.person(person_id).memories

        with self._lock:
            for existing in shelf:
                if overlap(existing.text, text) >= DUPLICATE_OVERLAP:
                    existing.touch()
                    existing.importance = min(1.0, existing.importance + 0.1)
                    return existing
            memory = Memory(text=text, kind=kind,
                            importance=max(0.0, min(importance, 1.0)))
            shelf.append(memory)
            self._prune(shelf)
            return memory

    def _prune(self, shelf: list[Memory]) -> None:
        """Drop what has faded, then the weakest if still over the ceiling.
        Caller holds the lock."""
        now = time.time()
        shelf[:] = [m for m in shelf if m.strength(now) >= FORGET_BELOW]
        if len(shelf) > MAX_PER_PERSON:
            shelf.sort(key=lambda m: m.strength(now), reverse=True)
            del shelf[MAX_PER_PERSON:]

    # --------------------------------------------------------------- recall

    def recall(self, person_id: str | None = None, limit: int = 6,
               touch: bool = True) -> list[Memory]:
        """
        The strongest things aiRon knows, strongest first.

        Recalling touches what it returns, so memories that keep proving
        useful stop decaying and the rest fade on their own.
        """
        with self._lock:
            shelf = list(self.people[person_id].memories
                         if person_id is not None and person_id in self.people
                         else self.world if person_id is None else [])
        now = time.time()
        shelf.sort(key=lambda m: m.strength(now), reverse=True)
        chosen = shelf[:limit]
        if touch:
            with self._lock:
                for memory in chosen:
                    memory.touch(now)
        return chosen

    def stats(self) -> dict:
        with self._lock:
            return {
                "people": len(self.people),
                "person_memories": sum(len(p.memories) for p in self.people.values()),
                "world_memories": len(self.world),
            }
