"""
Looking something up, and the second thing aiRon reaches the network for.

AIRON-26. The weather tool already argued its way past AIRON-8's rule that
tools stay off the network, and the argument was that it goes to one host
with one parameter and reads the answer out of a template. This tool cannot
make that argument. The model composes the query, the reply is prose written
by strangers, and no template can turn a paragraph about the Bundesliga into
a sentence a robot says out loud. So the reasoning has to be different:

- It never runs inside a turn. Like every action, it runs after aiRon has
  finished speaking - and because it is slower than the others, the model is
  told to say "let me look" first, so the wait is a robot looking something
  up rather than a robot that has frozen.
- One host, one path, one method. The key lives in .env and is sent in a
  header; nothing here ever logs it, and a query is a query - there is no
  parameter the model can set that changes where the request goes.
- What comes back is quoted, never obeyed. Search results are text written
  by whoever owned the page, and some of the web is written specifically to
  be read by a model. The containment is in two places: the prompt that
  turns findings into speech says they are quotes, and the schema for that
  second call has no actions in it at all. A page can therefore change what
  aiRon says, which is the unavoidable cost of reading the web out loud; it
  cannot make aiRon do anything, which is the part that would matter.
- What comes back is capped before it is read. A budget in characters, a
  handful of results, and one search per turn - because "search for X" from
  a model that has misread the room should cost one round trip, not twenty.

Tavily rather than a general web fetch, because it returns extracted text
and its own one-line answer instead of HTML, and because the alternative -
aiRon following links - is a much larger thing to have let a robot do.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..core.log import log

SEARCH_URL = "https://api.tavily.com/search"

#: The key, which is read here and never printed. Startup lists the names it
#: loaded from .env and never the values, and nothing in this file logs the
#: header it builds.
KEY_ENV = "TAVILY_API_KEY"

#: Longer than the weather's four seconds, because a search genuinely takes
#: longer, and shorter than the model's twelve, because this is spent before
#: the model has even been asked the question. Somebody is standing there
#: having just been told "let me look".
TIMEOUT_S = 6.0

#: How many results to keep. Tavily ranks them, and the fifth-best hit has
#: never yet been the one that answered a spoken question - it just spends
#: the budget below and the seconds the model takes to read it.
MAX_RESULTS = 4

#: How much of each result to keep, and how much in total, in characters.
#: Both matter: one long page must not crowd out three short ones, and the
#: whole lot is read by a model somebody is waiting for.
MAX_RESULT_CHARS = 600
MAX_TOTAL_CHARS = 2000

#: A query is a few words. This is not a guard against a malicious model so
#: much as against a confused one - a transcript pasted into the query field
#: is a bad search as well as a large request.
MAX_QUERY_CHARS = 200

#: The same question twice in a conversation is one round trip. Short,
#: because unlike tomorrow's weather, "what is the score" changes.
CACHE_S = 10 * 60.0

#: Control characters and the zero-width tricks used to hide text from a
#: human reader while leaving it visible to a model. Stripped before any of
#: this reaches a prompt.
#: Written with escapes rather than the characters themselves, because the
#: whole point of these is that you cannot see them in a source file either.
UNPRINTABLE = re.compile("[\x00-\x08\x0b-\x1f\x7f\u200b-\u200f\u202a-\u202e\ufeff]")


def _clean(text: str, limit: int) -> str:
    """One field from the web, made safe to put in front of a model.

    Safe here means only: no control characters, no invisible direction
    marks, no runaway length. It does not mean trustworthy - nothing can
    make it that, which is why the prompt treats it as a quotation and the
    schema it feeds has no actions in it.
    """
    return " ".join(UNPRINTABLE.sub(" ", str(text or "")).split())[:limit]


@dataclass(frozen=True)
class Source:
    """One page the answer might be in."""

    title: str
    url: str
    text: str


@dataclass(frozen=True)
class Findings:
    """What one search returned, trimmed to what a spoken answer can use."""

    query: str
    #: Tavily's own one-line answer, when it offers one. Kept separate from
    #: the sources because it is not one: it is another model's summary of
    #: them, and it is sometimes wrong about them. Asked who won Euro 2024 it
    #: returned "Spanien gewann ... Sie besiegten Frankreich im Finale",
    #: while the pages underneath it said England, which is what happened.
    #: So it is offered last and labelled for what it is - see as_prompt.
    answer: str = ""
    sources: tuple[Source, ...] = field(default_factory=tuple)
    fetched: float = 0.0

    def __bool__(self) -> bool:
        """Whether there is anything here worth answering from."""
        return bool(self.answer or self.sources)

    def as_prompt(self) -> str:
        """The findings as the model will see them: quoted, labelled, bounded.

        The pages first and the search engine's own summary last, which is
        the opposite of how it arrives and is deliberate. That summary is
        another model's reading of these same pages, it is one sentence long,
        and it is the most tempting thing in the document to simply repeat -
        so when it is wrong, it is wrong in exactly the way that reaches the
        person as a fact said out loud by a robot. It is kept because it is
        often the whole answer and costs nothing; it is put here, and named
        as what it is, so that it loses an argument with the pages.
        """
        parts = [f"[{n}] {source.title}\n{source.text}"
                 for n, source in enumerate(self.sources, 1)]
        if self.answer:
            parts.append("A one-line summary written by the search engine "
                         "itself, not taken from any of the pages above. It "
                         "is sometimes wrong where they are right:\n"
                         f"{self.answer}")
        return "\n\n".join(parts)


class Search:
    """
    Whatever aiRon last looked up, for whoever wants to answer from it.

    One instance, held by the toolbox. `look_up` runs on the thread that
    talks - after the talking, never during it - and returns None on every
    failure with the reason in `last_error`, because "aiRon could not find
    out" is a normal thing for a robot to be and not a reason to fall over
    mid-conversation.
    """

    def __init__(self, *, key: str | None = None):
        self._key = key if key is not None else os.environ.get(KEY_ENV, "").strip()
        self.last_error = ""
        #: The most recent findings, for anything that wants to show them.
        self.latest: Findings | None = None
        #: When somebody last asked, distinct from when the network was last
        #: touched: asking again inside the cache window is still asking.
        self.requested = 0.0
        self._cache: dict[str, Findings] = {}
        self._lock = threading.Lock()

    def configured(self) -> bool:
        return bool(self._key)

    def _post(self, query: str) -> dict:
        body = json.dumps({
            "query": query,
            "max_results": MAX_RESULTS,
            # "basic" rather than "advanced": advanced costs a second or two
            # more for deeper extraction, and the questions somebody asks a
            # robot out loud are answered by the summary line far more often
            # than by the fourth paragraph of the third result.
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            SEARCH_URL, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key}"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))

    def look_up(self, query: str, *, force: bool = False) -> Findings | None:
        """Search the web for one question.

        Returns None when there is no key, no network, or nothing usable came
        back - and a Findings that is falsy when the search itself succeeded
        and found nothing, which is a different answer and said differently.
        """
        self.requested = time.monotonic()
        asked = _clean(query, MAX_QUERY_CHARS)
        if not asked:
            self.last_error = "nothing to search for"
            return None
        if not self._key:
            self.last_error = f"no {KEY_ENV} in the environment"
            return None

        key = asked.lower()
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None and not force and time.monotonic() - cached.fetched < CACHE_S:
            with self._lock:
                self.latest = cached
            return cached

        started = time.monotonic()
        try:
            data = self._post(asked)
        except urllib.error.HTTPError as exc:
            # The status is worth having on its own: 401 is a key that needs
            # replacing and 432 is a plan that has run out, and both look
            # identical from in front of the robot.
            self.last_error = f"HTTP {exc.code} from Tavily"
            log(f"[search] could not search: {self.last_error}")
            return None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            log(f"[search] could not search: {self.last_error}")
            return None

        findings = self._read(asked, data)
        with self._lock:
            self._cache[key] = findings
            self.latest = findings
        log(f"[search] {asked!r}: {len(findings.sources)} results in "
            f"{time.monotonic() - started:.1f}s")
        return findings

    def _read(self, query: str, data: dict) -> Findings:
        """Tavily's response, trimmed to a character budget.

        Deliberately tolerant of shape: a missing field means one less thing
        to answer from, not a failed search. The only shape that matters is
        that results are a list of things with text in them.
        """
        answer = _clean(data.get("answer", ""), MAX_RESULT_CHARS)
        budget = MAX_TOTAL_CHARS - len(answer)
        sources = []
        for raw in (data.get("results") or [])[:MAX_RESULTS]:
            if not isinstance(raw, dict) or budget <= 0:
                break
            text = _clean(raw.get("content", ""), min(MAX_RESULT_CHARS, budget))
            if not text:
                continue
            budget -= len(text)
            sources.append(Source(title=_clean(raw.get("title", ""), 120),
                                  url=_clean(raw.get("url", ""), 200),
                                  text=text))
        return Findings(query=query, answer=answer, sources=tuple(sources),
                        fetched=time.monotonic())
