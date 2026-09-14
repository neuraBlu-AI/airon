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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.brain.tools import (FOUND_NOTHING, NO_SEARCH,         # noqa: E402
                               NO_TICKET, TOOLS, Action, Toolbox)
from airon.world.project import Filed, SAME_TICKET, _alike   # noqa: E402
from airon.world.search import Findings, Source, _clean      # noqa: E402
from airon.world.weather import Forecast                    # noqa: E402
from airon.core.events import Person, WorldState             # noqa: E402
from airon.face.expression import FaceAnimator               # noqa: E402
from airon.memory.service import MemoryService               # noqa: E402
from airon.brain.conversation import (Conversation,          # noqa: E402
                                      FOUND_SCHEMA, _Spoken)
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

    print("\nits own project (AIRON-32)")

    class StubTracker:
        """A Project with the tracker answered locally, so no key is needed
        and nothing is ever really filed."""

        def __init__(self, ready=True):
            self.ready = ready
            self.last_error = ""
            self.filed = []
            self.requested = 0.0

        def tracker_ready(self):
            return self.ready

        def look_up(self, question=""):
            return Findings(query=question or "aiRon's own project",
                            sources=(Source(title="Recent commits", url="",
                                            text="AIRON-31 ask whether it is "
                                                 "substituting"),), fetched=1.0)

        def file_ticket(self, title, detail=""):
            if not title:
                self.last_error = "nothing to file"
                return None
            if any(_alike(f.title, title) >= SAME_TICKET for f in self.filed):
                self.last_error = f"already filed as {self.filed[0].identifier}"
                return None
            filed = Filed(identifier=f"AIRON-{90 + len(self.filed)}",
                          title=title, url="https://example.invalid")
            self.filed.append(filed)
            return filed

    box = Toolbox(project=StubTracker(), lang="de")
    said = box.run([Action("project", "woran arbeitest du gerade?")], person="person_001")
    check("what it knows about itself goes to the model, not out loud",
          (said[0].speak, bool(said[0].found)), ("", True))

    said = box.run([Action("report", "Mikrofon fällt aus",
                           "Das Array liefert nach einer Weile keine Samples mehr.")],
                   person="person_001")
    check("a ticket is filed and named under its number",
          said[0].speak, "Ich habe das als AIRON-90 notiert.")
    said = box.run([Action("report", "Mikrofon fällt immer wieder aus", "nochmal")],
                   person="person_001")
    check("and the same fault is not filed twice", said[0].speak, NO_TICKET["de"])

    twice = Toolbox(project=StubTracker(), lang="de")
    twice.run([Action("report", "eins", "a"), Action("report", "zwei", "b")],
              person="person_001")
    check("one ticket per turn, however many the model asks for",
          [f.title for f in twice.project.filed], ["eins"])

    check("without a tracker it says so rather than pretending",
          Toolbox(project=StubTracker(ready=False), lang="de").run(
              [Action("report", "x", "y")], person="person_001")[0].speak,
          NO_TICKET["de"])

    # The boundary this ticket was scoped to. Reading the repository needs no
    # credential; writing to it is not a thing aiRon can do at all, and the
    # way to check that is that there is no tool for it.
    for forbidden in ("commit", "push", "edit", "write_file", "patch",
                      "open_pr", "merge", "run", "shell", "deploy"):
        check(f"no tool called {forbidden}", forbidden in TOOLS, False)
    check("nothing in the project module can write to the repository",
          any(word in Path("airon/world/project.py").read_text()
              for word in ('"commit"', '"push"', '"checkout"', 'subprocess.Popen',
                           'shell=True')), False)

    print("\nfailing safely")
    check("an unknown tool never becomes an Action", Action.from_dict(
        {"tool": "delete_everything", "target": "*"}), None)
    check("a malformed action never becomes an Action", Action.from_dict("nonsense"), None)
    empty = Toolbox()
    check("no memory, no face, no crash", empty.run(
        [Action("remember", "x"), Action("look_at", "y"), Action("forget", "z")],
        person=None), [])

    print("\nwhat aiRon is told about now (AIRON-27)")
    blocks = Conversation(lang="de").system("André")
    year = str(datetime.now().year)
    check("the current date reaches the model at all",
          year in blocks[1]["text"], True)
    # The one worth a test rather than a read. A line that changes every
    # minute, put in the block that never changes, is a personality cache
    # that is never once hit - and nothing about the robot looks wrong when
    # that happens, so nothing would ever catch it except this.
    check("and never from the cached block",
          "cache_control" in blocks[0] and year not in blocks[0]["text"], True)
    check("aiRon is told it knows the time, not that it might",
          "never say you cannot know" in blocks[0]["text"].lower(), True)

    print("\nwhat the model cannot reach")
    for forbidden in ("end_conversation", "enrol", "enroll_face", "set_name",
                      "ask_name", "stop_listening", "mute", "shutdown",
                      "forget_person", "say", "fetch", "open_url", "browse"):
        check(f"no tool called {forbidden}", forbidden in TOOLS, False)
    check("the whole list is seven tools", sorted(TOOLS),
          ["forget", "look_at", "project", "remember", "report", "search",
           "weather"])
    # The containment that matters for AIRON-26: what came back off the web
    # is read by a model that has no tools at all, so a page cannot reach
    # one however convincingly it is written.
    check("nothing off the web can ask for a tool",
          "actions" in FOUND_SCHEMA["properties"], False)
    check("nor have itself remembered as a fact about somebody",
          "remember" in FOUND_SCHEMA["properties"], False)

    print("\nadmitting the search missed (AIRON-31)")
    fields = list(FOUND_SCHEMA["properties"])
    check("a searched answer has to say whether it found anything",
          "answered" in FOUND_SCHEMA["required"], True)
    # The order is the mechanism, not tidiness: the reply is spoken sentence
    # by sentence as it streams, so a verdict that arrives after the words is
    # a verdict that arrives after they have been heard.
    check("and says so before it says anything else",
          fields.index("answered") < fields.index("say"), True)
    running = _Spoken()
    for piece in ('{"emotion":"curious", "answered"', ': fal', 'se, "say":"x"}'):
        running.feed(piece)
    check("the verdict is readable while the reply is still arriving",
          running.answered, False)
    check("and half of one is never mistaken for a verdict",
          _Spoken().feed('{"answered": fal') or _Spoken().answered, None)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for bad in FAILED:
        print(f"  failed: {bad}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
