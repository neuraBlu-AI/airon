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

from airon.brain.tools import TOOLS, Action, Toolbox         # noqa: E402
from airon.world.weather import Forecast                    # noqa: E402
from airon.core.events import Person, WorldState             # noqa: E402
from airon.face.expression import FaceAnimator               # noqa: E402
from airon.memory.service import MemoryService               # noqa: E402
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
                      "forget_person", "say"):
        check(f"no tool called {forbidden}", forbidden in TOOLS, False)
    check("the whole list is four tools", sorted(TOOLS),
          ["forget", "look_at", "remember", "weather"])

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for bad in FAILED:
        print(f"  failed: {bad}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
