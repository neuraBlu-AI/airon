"""
What the model may ask aiRon to do, as opposed to say.

AIRON-8. Until now the model returned words, a face and some memories, and
that was the whole of what it could do. Spec section 7 wants it asking for
actions - "look at Pierre" - rather than reaching for the machinery itself,
and the mechanism is worth building while the actions are harmless and the
worst outcome is a robot looking the wrong way.

Four tools, and the list is short on purpose:

    look_at    point attention at somebody in the room
    remember   keep something because the person asked, not because the
               model judged it worth keeping
    forget     retract one thing, without deleting the person
    weather    tomorrow's forecast, here or somewhere named; said and shown

Only `weather` reaches the network, to one host with coordinates from .env,
and it says what it fetched rather than letting the model say it - the
reasoning is in airon/world/weather.py. Nothing here is safety-critical.
Section 20 says safety-critical behaviour must never depend solely on a
language model, and that does not stop being true because the model is
calling a function instead of talking.

The harder constraint is the second one. brain_service owns turn-taking,
cooldowns, the answer deadline and the name-asking flow, and it owns them
because they have to work when the model does not. So there is deliberately
no tool that ends a conversation, enrols a face, changes a name, or silences
aiRon: a tool that could do any of those would move that authority into the
model through the back door, which is the failure this design exists to
avoid. The way to check that is to read TOOLS - if it is not there, the model
cannot reach it.

Requests arrive as data and are executed here, rather than through the API's
tool-calling loop. That is a latency decision: none of these return
anything the model needs to see - the weather tool says its own result -
and a second round trip would double the wait for a robot whose whole
conversation is already about four seconds.
When a tool does need to hand something back - a battery level, a search
result - that is the point to reach for the SDK's tool runner instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.log import log
from ..world.weather import NOWHERE

#: Everything the model is allowed to ask for. The enum in the reply schema is
#: built from this, so adding a name here is the only way to add a tool - and
#: is a deliberate act, reviewable in a diff.
TOOLS = ("look_at", "remember", "forget", "weather")

#: What an explicitly requested memory is worth. Higher than the model's own
#: judgement calls, which sit near 0.4: somebody saying "remember this" is the
#: strongest signal available that a thing matters, and it should outlive the
#: small talk around it.
ASKED_IMPORTANCE = 0.8

#: Said when the forecast cannot be had. Deliberately an admission rather
#: than a guess: the point of fetching is that aiRon knows, and when it does
#: not know it should say so.
NO_WEATHER = {
    "en": "I could not reach the weather just now.",
    "de": "Ich komme gerade nicht an das Wetter heran.",
}


@dataclass(frozen=True)
class Outcome:
    """What a tool did. `speak` is a line for aiRon to say out loud, and only
    the weather tool has one: it exists so a fact reaches the person as the
    fact that was fetched, rather than as the model's memory of asking."""

    log: str
    speak: str = ""


@dataclass(frozen=True)
class Action:
    """One thing the model asked for. Data, not a callable - so the layer can
    be exercised without a model, which is AIRON-8's second requirement."""

    tool: str
    target: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "Action | None":
        if not isinstance(raw, dict):
            return None
        tool = str(raw.get("tool", "")).strip()
        if tool not in TOOLS:
            return None
        return cls(tool=tool, target=" ".join(str(raw.get("target", "")).split()))


class Toolbox:
    """
    Runs what the model asked for, and refuses everything else.

    Every method returns a short line saying what happened, which is what
    ends up in the log. "Did nothing, and why" is a real answer here: aiRon
    looking at somebody who left, or forgetting a thing it never knew, should
    be visible rather than silent.
    """

    def __init__(self, *, memory=None, face=None, store=None, weather=None,
                 lang: str = "en"):
        self.memory = memory
        self.face = face
        self.store = store        # StateStore: who is currently visible
        self.weather = weather
        self.lang = lang

    def run(self, actions: list[Action], *, person: str | None) -> list[Outcome]:
        """Execute in order, and never raise: a bad tool call is a robot that
        did not do something, not a robot that fell over mid-sentence."""
        done = []
        for action in actions:
            try:
                outcome = getattr(self, f"_{action.tool}")(action.target, person)
            except Exception as exc:                    # a tool, not the turn
                outcome = Outcome(f"{action.tool} failed: "
                                  f"{type(exc).__name__}: {exc}")
            if isinstance(outcome, str):
                outcome = Outcome(outcome)
            if outcome and outcome.log:
                log(f"[tool] {outcome.log}")
                done.append(outcome)
        return done

    # ------------------------------------------------------------ the tools

    def _look_at(self, target: str, person: str | None) -> str:
        """Point the face at somebody who is actually in the room.

        Resolved against what the camera can see right now, not against the
        gallery: being asked to look at somebody who is not here should do
        nothing, rather than aim the eyes at a remembered coordinate.
        """
        if self.face is None or self.store is None:
            return ""
        world = self.store.get()
        wanted = target.strip().lower()
        match = next((p for p in world.people
                      if (p.name or "").lower() == wanted or p.id == target), None)
        if match is None and wanted in ("them", "the speaker", "whoever spoke"):
            match = next((p for p in world.people if p.id == person), None)
        if match is None:
            return f"not looking at {target!r} - nobody here by that name"
        self.face.command.attention_person = match.id
        return f"looking at {match.name or match.id}"

    def _remember(self, target: str, person: str | None) -> str:
        """Keep something because it was asked for, not because it was judged."""
        if self.memory is None or not target:
            return ""
        memory = self.memory.remember(target, person_id=person, kind="fact",
                                      importance=ASKED_IMPORTANCE)
        if memory is None:
            return f"could not remember {target!r}"
        return f"remembered on request: {memory.text!r}"

    def _weather(self, target: str, person: str | None) -> Outcome:
        """Tomorrow's forecast: fetched, shown, and read out.

        The spoken line is built from the response rather than left to the
        model, which is the whole reason this tool can be allowed to exist -
        see airon/world/weather.py. If it cannot be fetched, aiRon says so
        rather than guessing.
        """
        if self.weather is None:
            return Outcome("")
        if not self.weather.configured():
            return Outcome("no weather: nowhere configured - set "
                           "AIRON_WEATHER_PLACE in .env",
                           NO_WEATHER.get(self.lang, NO_WEATHER["en"]))
        forecast = self.weather.tomorrow(target)
        if forecast is None:
            # A place nobody can find is a different answer from a network
            # that will not answer, and saying so is the difference between
            # aiRon looking broken and aiRon looking like it misheard - which
            # is usually what actually happened.
            if target and "no such place" in self.weather.last_error:
                return Outcome(f"no such place: {target!r}",
                               NOWHERE.get(self.lang, NOWHERE["en"]).format(
                                   place=target))
            return Outcome(f"no weather: {self.weather.last_error}",
                           NO_WEATHER.get(self.lang, NO_WEATHER["en"]))
        return Outcome(f"tomorrow in {forecast.place}: {forecast.condition}, "
                       f"{forecast.low}-{forecast.high}C",
                       forecast.sentence(self.lang))

    def _forget(self, target: str, person: str | None) -> str:
        """Retract one thing. Refusing is the safe outcome - see
        MemoryStore.forget_memory, which would rather drop nothing than drop
        the wrong thing and leave the person believing it is gone."""
        if self.memory is None or not target:
            return ""
        memory = self.memory.forget_memory(target, person_id=person)
        if memory is None:
            return f"nothing close enough to {target!r} to forget"
        return f"forgot on request: {memory.text!r}"
