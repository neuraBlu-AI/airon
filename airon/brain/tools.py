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
    search     look something up on the web, and answer from what came back
    project    what aiRon has been working on, what is open, what broke
    report     put one ticket in the tracker, and say that it did

The last two reach the network and the first three do not. `weather` goes to
one host with coordinates from .env and says what it fetched rather than
letting the model say it; `search` cannot do that, because no template turns
a web page into a spoken sentence, so it hands what it found back to the
model under a prompt that treats it as a quotation. The reasoning for each is
in airon/world/weather.py and airon/world/search.py. Nothing here is
safety-critical.
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
tool-calling loop. That is a latency decision: three of these return nothing
the model needs to see - the weather tool says its own result - and a second
round trip would double the wait for a robot whose whole conversation is
already about four seconds.

`search` is the one that does need to hand something back, which AIRON-8
anticipated and pointed at the SDK's tool runner for. It is not built that
way, and the reason is that the tool runner goes quiet while it works: the
model asks, the loop fetches, the model answers, and the person in front of
the robot hears nothing for the whole of it. Keeping requests as data lets
aiRon say "let me look" out of the first call, search while those words are
in the air, and answer out of a second - the same two round trips, with the
robot talking through the first one. What comes back rides on Outcome.found,
and brain_service is what asks the model to turn it into speech.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.log import log
from ..world.project import KEY_ENV as TRACKER_ENV
from ..world.search import KEY_ENV, Findings
from ..world.weather import NOWHERE

#: Everything the model is allowed to ask for. The enum in the reply schema is
#: built from this, so adding a name here is the only way to add a tool - and
#: is a deliberate act, reviewable in a diff.
TOOLS = ("look_at", "remember", "forget", "weather", "search",
         "project", "report")

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

#: Said when a search could not be run at all - no key, no network, no
#: answer from Tavily. Distinct from having searched and found nothing,
#: below, because they are different things to be told by a robot that has
#: just said it would go and look.
NO_SEARCH = {
    "en": "I could not look that up just now.",
    "de": "Das konnte ich gerade nicht nachschlagen.",
}

#: Said when the search ran and came back empty. Nothing reaches the model
#: in this case: there is nothing to answer from, and a model asked to
#: answer from nothing is a model inventing an answer.
FOUND_NOTHING = {
    "en": "I looked, but I could not find anything about that.",
    "de": "Ich habe nachgesehen, aber dazu finde ich nichts.",
}

#: Said when the tracker cannot be reached, or would not take the ticket.
#: An admission rather than a shrug: somebody who has just been told their
#: complaint was written down should not find out later that it was not.
NO_TICKET = {
    "en": "I could not write that down in the tracker just now.",
    "de": "Ich konnte das gerade nicht ins Ticket-System eintragen.",
}

#: Said when one was filed. Built here rather than left to the model for the
#: same reason the forecast is: the person should hear what was actually
#: written down, under the number it was written down as.
FILED = {
    "en": "I have written that down as {ticket}.",
    "de": "Ich habe das als {ticket} notiert.",
}

#: One ticket per turn, for the same reason as one search: a model that has
#: misread the room should cost one row in the tracker, not nine.
REPORTS_PER_TURN = 1

#: One search per turn. The model asking twice in one reply is a model that
#: has misread the room, and the cost of humouring it is paid by whoever is
#: standing there waiting - so the second request is dropped rather than run.
SEARCHES_PER_TURN = 1


@dataclass(frozen=True)
class Outcome:
    """What a tool did.

    `speak` is a line for aiRon to say out loud, and only the weather tool
    has one: it exists so a fact reaches the person as the fact that was
    fetched, rather than as the model's memory of asking.

    `found` is the other shape, and only the search tool has it: what came
    back cannot be said as it stands, because it is web prose rather than a
    forecast with two numbers in it. It travels back to brain_service, which
    asks the model to turn it into one spoken sentence. The two are
    exclusive by construction - a tool either knows how to say its result or
    hands it to somebody who does.
    """

    log: str
    speak: str = ""
    found: Findings | None = None


@dataclass(frozen=True)
class Action:
    """One thing the model asked for. Data, not a callable - so the layer can
    be exercised without a model, which is AIRON-8's second requirement."""

    tool: str
    target: str = ""
    #: A second string, for the one tool that needs more than a name. `report`
    #: uses it for the sentence under the title; everything else leaves it
    #: empty, which is why it is not called `title` and `body`.
    detail: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "Action | None":
        if not isinstance(raw, dict):
            return None
        tool = str(raw.get("tool", "")).strip()
        if tool not in TOOLS:
            return None
        return cls(tool=tool,
                   target=" ".join(str(raw.get("target", "")).split()),
                   detail=" ".join(str(raw.get("detail", "")).split()))


class Toolbox:
    """
    Runs what the model asked for, and refuses everything else.

    Every method returns a short line saying what happened, which is what
    ends up in the log. "Did nothing, and why" is a real answer here: aiRon
    looking at somebody who left, or forgetting a thing it never knew, should
    be visible rather than silent.
    """

    def __init__(self, *, memory=None, face=None, store=None, weather=None,
                 search=None, project=None, lang: str = "en"):
        self.memory = memory
        self.face = face
        self.store = store        # StateStore: who is currently visible
        self.weather = weather
        self.search = search
        self.project = project
        self.lang = lang

    def run(self, actions: list[Action], *, person: str | None) -> list[Outcome]:
        """Execute in order, and never raise: a bad tool call is a robot that
        did not do something, not a robot that fell over mid-sentence."""
        done = []
        searches = reports = 0
        for action in actions:
            if action.tool == "search":
                searches += 1
                if searches > SEARCHES_PER_TURN:
                    log(f"[tool] ignoring a second search this turn: "
                        f"{action.target!r}")
                    continue
            if action.tool == "report":
                reports += 1
                if reports > REPORTS_PER_TURN:
                    log(f"[tool] ignoring a second ticket this turn: "
                        f"{action.target!r}")
                    continue
            try:
                outcome = getattr(self, f"_{action.tool}")(action, person)
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

    def _look_at(self, action: "Action", person: str | None) -> str:
        """Point the face at somebody who is actually in the room.

        Resolved against what the camera can see right now, not against the
        gallery: being asked to look at somebody who is not here should do
        nothing, rather than aim the eyes at a remembered coordinate.
        """
        if self.face is None or self.store is None:
            return ""
        world = self.store.get()
        wanted = action.target.strip().lower()
        match = next((p for p in world.people
                      if (p.name or "").lower() == wanted or p.id == action.target), None)
        if match is None and wanted in ("them", "the speaker", "whoever spoke"):
            match = next((p for p in world.people if p.id == person), None)
        if match is None:
            return f"not looking at {action.target!r} - nobody here by that name"
        self.face.command.attention_person = match.id
        return f"looking at {match.name or match.id}"

    def _remember(self, action: "Action", person: str | None) -> str:
        """Keep something because it was asked for, not because it was judged."""
        if self.memory is None or not action.target:
            return ""
        memory = self.memory.remember(action.target, person_id=person, kind="fact",
                                      importance=ASKED_IMPORTANCE)
        if memory is None:
            return f"could not remember {action.target!r}"
        return f"remembered on request: {memory.text!r}"

    def _weather(self, action: "Action", person: str | None) -> Outcome:
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
        forecast = self.weather.tomorrow(action.target)
        if forecast is None:
            # A place nobody can find is a different answer from a network
            # that will not answer, and saying so is the difference between
            # aiRon looking broken and aiRon looking like it misheard - which
            # is usually what actually happened.
            if action.target and "no such place" in self.weather.last_error:
                return Outcome(f"no such place: {action.target!r}",
                               NOWHERE.get(self.lang, NOWHERE["en"]).format(
                                   place=action.target))
            return Outcome(f"no weather: {self.weather.last_error}",
                           NO_WEATHER.get(self.lang, NO_WEATHER["en"]))
        return Outcome(f"tomorrow in {forecast.place}: {forecast.condition}, "
                       f"{forecast.low}-{forecast.high}C",
                       forecast.sentence(self.lang))

    def _search(self, action: "Action", person: str | None) -> Outcome:
        """Look something up, and hand back what was found rather than say it.

        The one tool whose result the model has to see. Everything else here
        either does something silently or reads a template out loud; a search
        returns paragraphs written by strangers, and turning those into one
        sentence a robot says is a judgement, which means a second call.

        Three outcomes, said differently on purpose. Not being able to search
        at all, searching and finding nothing, and finding something are three
        different things to be told by a robot that has just announced it was
        going to go and look - and only the third is worth waking the model
        for, because a model asked to answer from nothing invents an answer.
        """
        if self.search is None:
            return Outcome("")
        if not self.search.configured():
            return Outcome(f"no search: no {KEY_ENV} in .env",
                           NO_SEARCH.get(self.lang, NO_SEARCH["en"]))
        findings = self.search.look_up(action.target)
        if findings is None:
            return Outcome(f"no search: {self.search.last_error}",
                           NO_SEARCH.get(self.lang, NO_SEARCH["en"]))
        if not findings:
            return Outcome(f"searched for {action.target!r} and found nothing",
                           FOUND_NOTHING.get(self.lang, FOUND_NOTHING["en"]))
        return Outcome(f"searched for {findings.query!r}: "
                       f"{len(findings.sources)} results"
                       f"{', with a summary' if findings.answer else ''}",
                       found=findings)

    def _project(self, action: "Action", person: str | None) -> Outcome:
        """What aiRon has been working on, what is open, and what has broken.

        Findings, like a search, and for the same reason: there is no cheap
        way to search a repository for the answer to a spoken question, so
        the project's actual state goes to the model and the model finds the
        answer in it - or admits it cannot, which AIRON-31 made it able to do.

        Everything here is read. The repository is read from the local clone
        with no credentials at all, the tracker read-only over its own key.
        Nothing in this tool can change a line of aiRon.
        """
        if self.project is None:
            return Outcome("")
        findings = self.project.look_up(action.target)
        if not findings:
            return Outcome("nothing to say about the project - no repository "
                           f"and no tracker ({self.project.last_error})",
                           FOUND_NOTHING.get(self.lang, FOUND_NOTHING["en"]))
        return Outcome(f"looked itself up: {len(findings.sources)} things known"
                       f"{' (tracker off)' if not self.project.tracker_ready() else ''}",
                       found=findings)

    def _report(self, action: "Action", person: str | None) -> Outcome:
        """Put one ticket in the tracker, and say so under its number.

        The only tool aiRon has that writes anywhere outside its own memory,
        which is why what it says is a template rather than the model's
        account of what it did: somebody told their complaint was written
        down should hear the number it was written down as, and should find
        out immediately when it was not.

        The refusals live in Project.file_ticket - no title, no key, the same
        thing twice, too many in one run - because they are about the tracker
        rather than about the conversation. This is only where they are said.
        """
        if self.project is None:
            return Outcome("")
        if not self.project.tracker_ready():
            return Outcome(f"no ticket: no {TRACKER_ENV} in .env",
                           NO_TICKET.get(self.lang, NO_TICKET["en"]))
        filed = self.project.file_ticket(action.target, action.detail)
        if filed is None:
            return Outcome(f"no ticket: {self.project.last_error}",
                           NO_TICKET.get(self.lang, NO_TICKET["en"]))
        return Outcome(f"filed {filed.identifier}: {filed.title!r}",
                       FILED.get(self.lang, FILED["en"]).format(
                           ticket=filed.identifier))

    def _forget(self, action: "Action", person: str | None) -> str:
        """Retract one thing. Refusing is the safe outcome - see
        MemoryStore.forget_memory, which would rather drop nothing than drop
        the wrong thing and leave the person believing it is gone."""
        if self.memory is None or not action.target:
            return ""
        memory = self.memory.forget_memory(action.target, person_id=person)
        if memory is None:
            return f"nothing close enough to {action.target!r} to forget"
        return f"forgot on request: {memory.text!r}"
