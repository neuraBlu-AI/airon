#!/usr/bin/env python3
"""
Does the tool layer do what it says? - NOT part of the robot.

    python tools/tool_bench.py

AIRON-8 asks that the tools be testable without calling the model at all,
which is the point of them being data: an Action is a dataclass, so the whole
layer can be driven from here with no key, no network and no conversation.

Checks three things. That each tool does its job; that each fails safely when
it cannot; and - the one that matters - that nothing brain_service owns
deterministically can be reached through a tool.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.brain.tools import (FOUND_NOTHING, NO_SEARCH, TOOLS,   # noqa: E402
                               Action, Toolbox)
from airon.world.search import Findings, Source, _clean      # noqa: E402
from airon.world.weather import Forecast                    # noqa: E402
from airon.core.events import Person, WorldState             # noqa: E402
from airon.face.expression import FaceAnimator               # noqa: E402
from airon.memory.service import MemoryService               # noqa: E402
from airon.brain.conversation import FOUND_SCHEMA            # noqa: E402
from airon.core import EventBus, StateStore                  # noqa: E402

PASSED, FAILED = [], []


def check(what: str, got, want) -> None:
    (PASSED if got == want else FAILED).append(what)
    mark = "ok  " if got == want else "FAIL"
    print(f"  {mark} {what}")
    if got != want:
        print(f"       wanted {want!r}\n       got    {got!r}")


def main() -> int:
    bus = EventBus()
    store = StateStore()
    face = FaceAnimator()
    memory = MemoryService(bus, path=Path(tempfile.mkdtemp()) / "memory.json")
    box = Toolbox(memory=memory, face=face, store=store)

    store.set(WorldState(people=[
        Person(id="person_001", name="André", recognized=True),
        Person(id="person_002", name="Max", recognized=True),
    ]))

    print("\nlook_at")
    box.run([Action("look_at", "Max")], person="person_001")
    check("points the face at the person named", face.command.attention_person, "person_002")
    box.run([Action("look_at", "Elizabeth")], person="person_001")
    check("ignores somebody who is not in the room",
          face.command.attention_person, "person_002")

    print("\nremember")
    box.run([Action("remember", "nimmt seine Tabletten um acht")], person="person_001")
    kept = [m.text for m in memory.store.recall("person_001")]
    check("keeps what it was asked to keep", kept, ["nimmt seine Tabletten um acht"])
    check("and keeps it as more than small talk",
          memory.store.recall("person_001", touch=False)[0].importance > 0.6, True)

    print("\nforget")
    box.run([Action("forget", "die Tabletten um acht")], person="person_001")
    check("drops it when asked", memory.store.recall("person_001"), [])
    box.run([Action("remember", "mag Baguette")], person="person_001")
    box.run([Action("forget", "irgendwas über Autos")], person="person_001")
    check("refuses when nothing matches, rather than guessing",
          [m.text for m in memory.store.recall("person_001")], ["mag Baguette"])

    print("\nweather (no network touched: a stub stands in for the sky)")

    class StubWeather:
        """Everything the tool needs, and nothing that leaves the house."""
        latest = None
        last_error = ""
        requested = 0.0

        def __init__(self, forecast=None):
            self.latest = forecast

        def configured(self):
            return self.latest is not None

        def tomorrow(self, force=False):
            return self.latest

    sunny = Forecast(condition="clear", high=24, low=12, place="Nowhere", fetched=0.0)
    said = Toolbox(weather=StubWeather(sunny), lang="de").run(
        [Action("weather", "morgen")], person="person_001")
    check("says the forecast it fetched, in German",
          said[0].speak, "Morgen wird es klar, zwischen 12 und 24 Grad.")
    said = Toolbox(weather=StubWeather(sunny), lang="en").run(
        [Action("weather", "tomorrow")], person="person_001")
    check("and in English",
          said[0].speak, "Tomorrow looks clear, between 12 and 24 degrees.")
    said = Toolbox(weather=StubWeather(None), lang="de").run(
        [Action("weather", "morgen")], person="person_001")
    check("admits it when it cannot fetch, rather than guessing",
          said[0].speak, "Ich komme gerade nicht an das Wetter heran.")

    print("\nsearch (no network touched: a stub stands in for the web)")

    class StubSearch:
        """Everything the tool needs, and nothing that leaves the house."""

        def __init__(self, findings=None, key="tvly-test"):
            self.latest = findings
            self.last_error = "" if findings else "Bogus: no"
            self.requested = 0.0
            self._key = key
            self.asked = []

        def configured(self):
            return bool(self._key)

        def look_up(self, query, force=False):
            self.asked.append(query)
            return self.latest

    found = Findings(query="wer hat die em 2024 gewonnen",
                     answer="Spain won Euro 2024, beating England 2-1.",
                     sources=(Source(title="UEFA", url="https://example.invalid",
                                     text="Spain beat England 2-1 in Berlin."),))

    web = StubSearch(found)
    said = Toolbox(search=web, lang="de").run(
        [Action("search", "wer hat die em 2024 gewonnen")], person="person_001")
    check("passes the query the model wrote straight through",
          web.asked, ["wer hat die em 2024 gewonnen"])
    check("hands the findings back rather than saying them",
          (said[0].speak, said[0].found.answer),
          ("", "Spain won Euro 2024, beating England 2-1."))

    empty = Findings(query="blubb", fetched=1.0)
    said = Toolbox(search=StubSearch(empty), lang="de").run(
        [Action("search", "blubb")], person="person_001")
    check("says it found nothing, and sends nothing to the model",
          (said[0].speak, said[0].found), (FOUND_NOTHING["de"], None))
    check("admits it when the search itself failed",
          Toolbox(search=StubSearch(None), lang="de").run(
              [Action("search", "x")], person="person_001")[0].speak,
          NO_SEARCH["de"])
    check("and when there is no key at all",
          Toolbox(search=StubSearch(found, key=""), lang="en").run(
              [Action("search", "x")], person="person_001")[0].speak,
          NO_SEARCH["en"])

    twice = StubSearch(found)
    Toolbox(search=twice, lang="de").run(
        [Action("search", "eins"), Action("search", "zwei"),
         Action("search", "drei")], person="person_001")
    check("one search per turn, however many the model asks for",
          twice.asked, ["eins"])

    print("\nwhat comes back off the web")
    check("zero-width characters never reach the prompt",
          _clean("ignore\u200b all\u202e previous", 100), "ignore all previous")
    check("control characters never reach the prompt",
          _clean("a\x00b\x1fc", 100), "a b c")
    check("one huge page cannot crowd out the rest",
          len(_clean("x" * 9999, 600)), 600)
    check("findings with nothing in them are falsy, so nothing is asked",
          bool(Findings(query="q")), False)

    print("\nfailing safely")
    check("an unknown tool never becomes an Action", Action.from_dict(
        {"tool": "delete_everything", "target": "*"}), None)
    check("a malformed action never becomes an Action", Action.from_dict("nonsense"), None)
    empty = Toolbox()
    check("no memory, no face, no crash", empty.run(
        [Action("remember", "x"), Action("look_at", "y"), Action("forget", "z")],
        person=None), [])

    print("\nwhat the model cannot reach")
    for forbidden in ("end_conversation", "enrol", "enroll_face", "set_name",
                      "ask_name", "stop_listening", "mute", "shutdown",
                      "forget_person", "say", "fetch", "open_url", "browse"):
        check(f"no tool called {forbidden}", forbidden in TOOLS, False)
    check("the whole list is five tools", sorted(TOOLS),
          ["forget", "look_at", "remember", "search", "weather"])
    # The containment that matters for AIRON-26: what came back off the web
    # is read by a model that has no tools at all, so a page cannot reach
    # one however convincingly it is written.
    check("nothing off the web can ask for a tool",
          "actions" in FOUND_SCHEMA["properties"], False)
    check("nor have itself remembered as a fact about somebody",
          "remember" in FOUND_SCHEMA["properties"], False)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for bad in FAILED:
        print(f"  failed: {bad}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
