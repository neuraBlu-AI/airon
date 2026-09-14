"""
What aiRon can find out about itself: its own repository, its own tickets,
and what has gone wrong with it lately.

AIRON-32. The ticket asked for access to coding tools and GitHub "to improve
own code", and the boundary drawn for this first pass is read, plus filing a
ticket. aiRon can say what it has been working on, what is still open, and
what broke; it cannot change a line of itself. That split is not timidity,
it is the same rule the rest of the toolbox is built on - brain/tools.py
says of itself that there is deliberately no tool that enrols a face or
changes a name, because a tool that could would move that authority into the
model through the back door. Editing its own source is further past that
line than anything currently on the far side of it, and the same brain now
reads web pages written by strangers (AIRON-26), so a page that can reach a
commit is a page that can reach the robot.

Three sources, and they are deliberately unequal:

    git      the local repository. No credentials, no network, no account.
             Every command is a fixed argument list run without a shell, and
             nothing the model says ever reaches one - see `_git`.
    trouble  what has gone wrong in this run, from core/log.py's ring buffer.
             This is the half that makes "noticing its own errors" mean
             something rather than being a phrase in a ticket.
    Plane    the tracker, over its REST API, if a key is configured. Reading
             is optional and filing needs the same key; without it aiRon says
             it cannot reach the tracker, rather than guessing what is open.

Nothing here touches GitHub, and that is deliberate too. The `gh` CLI on this
machine is authenticated with `repo` scope, which is write access to every
repository the account can see. Handing a subprocess holding that to a robot
in order to read a commit message is a trade nobody would make twice. The
local clone has the commits in it already.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..core.log import log, recent_trouble
from .search import Findings, Source, _clean

#: Where the tracker lives, and what it needs to be reachable. The workspace
#: is the slug out of a Plane URL - app.plane.so/<workspace>/projects/... -
#: and the project is the one aiRon's tickets are in.
PLANE_API = "https://api.plane.so/api/v1"
KEY_ENV = "PLANE_API_KEY"
WORKSPACE_ENV = "PLANE_WORKSPACE"
PROJECT_ENV = "PLANE_PROJECT_ID"

#: Short, because this runs on the thread that talks.
TIMEOUT_S = 6.0

#: A git command that has not answered in this long has hung on something,
#: and a robot standing silent is worse than a robot that does not know.
GIT_TIMEOUT_S = 3.0

#: How much of the repository is worth saying out loud. A spoken answer uses
#: one or two of these; the rest is there so the model can pick.
COMMITS = 8
TICKETS = 12

#: The tracker does not change between two questions in one conversation.
CACHE_S = 5 * 60.0

#: Filed tickets are capped per run rather than per turn alone. One turn
#: cannot file two, and a conversation that keeps circling the same fault
#: should leave one ticket behind rather than nine.
REPORTS_PER_RUN = 5

#: How close two titles have to be before the second is the same ticket. A
#: crude ratio on words, which is enough: the failure being guarded against
#: is aiRon filing "microphone drops out" four times in four minutes, not a
#: subtle duplicate somebody would have to think about.
SAME_TICKET = 0.6


def _words(text: str) -> set[str]:
    return {word for word in text.lower().split() if len(word) > 3}


def _alike(one: str, two: str) -> float:
    first, second = _words(one), _words(two)
    if not first or not second:
        return 0.0
    return len(first & second) / min(len(first), len(second))


@dataclass(frozen=True)
class Filed:
    """A ticket aiRon put in the tracker."""

    identifier: str
    title: str
    url: str


class Project:
    """
    aiRon's own repository and tracker, as far as it is allowed to see them.

    `look_up` gathers the lot into the same Findings the web search returns,
    which is not a convenience so much as the point: the brain already knows
    how to turn findings into one spoken sentence, and it already refuses to
    answer from findings that do not contain the answer (AIRON-31). Asking
    "what have you been working on" should go through the same discipline as
    asking who won a football match, and it does.
    """

    def __init__(self, *, root: Path | None = None, key: str | None = None,
                 workspace: str | None = None, project_id: str | None = None):
        self.root = root or Path(__file__).resolve().parent.parent.parent
        self._key = key if key is not None else os.environ.get(KEY_ENV, "").strip()
        self.workspace = (workspace if workspace is not None
                          else os.environ.get(WORKSPACE_ENV, "").strip())
        self.project_id = (project_id if project_id is not None
                           else os.environ.get(PROJECT_ENV, "").strip())
        self.last_error = ""
        #: What was filed this run, so the same fault is not filed twice and
        #: a loop cannot fill the tracker.
        self.filed: list[Filed] = []
        self.requested = 0.0
        self._tickets: tuple[float, list[str]] | None = None
        self._states: dict[str, str] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- the repo

    def tracker_ready(self) -> bool:
        return bool(self._key and self.workspace and self.project_id)

    def _git(self, *args: str) -> str:
        """One git command, with a fixed argument list and no shell.

        Every caller below passes literals. Nothing the model says reaches
        this function, and that is the property worth keeping rather than
        the sandboxing that would be needed if it did.
        """
        try:
            done = subprocess.run(("git", *args), cwd=self.root, timeout=GIT_TIMEOUT_S,
                                  capture_output=True, text=True, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            self.last_error = f"git: {type(exc).__name__}"
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    def branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD")

    def commits(self, count: int = COMMITS) -> list[str]:
        """Recent commit subjects, newest first, with how long ago."""
        raw = self._git("log", f"-{count}", "--date=relative",
                        "--pretty=format:%ad - %s")
        return [line for line in raw.splitlines() if line.strip()]

    def uncommitted(self) -> list[str]:
        """Files changed but not committed - what aiRon is in the middle of."""
        raw = self._git("status", "--short")
        return [line.strip() for line in raw.splitlines() if line.strip()][:10]

    # ---------------------------------------------------------- the tracker

    def _plane(self, path: str, body: dict | None = None) -> dict | list | None:
        url = f"{PLANE_API}/workspaces/{self.workspace}/projects/{self.project_id}/{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url, data=data, method="POST" if data else "GET",
            headers={"Content-Type": "application/json", "x-api-key": self._key})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # The status on its own is worth having: 401 is a key to replace
            # and 404 is the wrong workspace slug, and from in front of the
            # robot both are "it did not work".
            self.last_error = f"HTTP {exc.code} from Plane"
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
        log(f"[project] tracker unreachable: {self.last_error}")
        return None

    def _state_names(self) -> dict[str, str]:
        """State id to group, so a ticket can be called open or done."""
        if self._states is not None:
            return self._states
        found = self._plane("states/")
        states = {}
        if isinstance(found, dict):
            for state in found.get("results") or []:
                states[state.get("id", "")] = state.get("group", "")
        self._states = states
        return states

    def tickets(self) -> list[str] | None:
        """Open tickets, one line each.

        Three answers, not two. A list is what is open, an empty list means
        nothing is, and None means the tracker could not be reached - and
        that distinction is the whole of it, because collapsing the third
        into the second has aiRon saying "nothing is open" about a tracker
        it never spoke to. That is the same fault AIRON-31 exists for, one
        layer further down.
        """
        if not self.tracker_ready():
            self.last_error = f"no tracker - set {KEY_ENV} in .env"
            return None
        with self._lock:
            cached = self._tickets
        if cached is not None and time.monotonic() - cached[0] < CACHE_S:
            return cached[1]

        found = self._plane("issues/")
        if not isinstance(found, dict):
            return None
        groups = self._state_names()
        open_ones = []
        for issue in (found.get("results") or []):
            group = groups.get(issue.get("state", ""), "")
            if group in ("completed", "cancelled"):
                continue
            name = _clean(issue.get("name", ""), 120)
            if name:
                open_ones.append(f"AIRON-{issue.get('sequence_id', '?')}: {name}")
        open_ones = open_ones[:TICKETS]
        with self._lock:
            self._tickets = (time.monotonic(), open_ones)
        return open_ones

    def file_ticket(self, title: str, detail: str = "") -> Filed | None:
        """Put one ticket in the tracker. The only thing here that writes.

        Refuses more often than it agrees, on purpose. It will not file
        without a title, will not file the same thing twice in a run, and
        will not file at all past a handful - because the failure mode of a
        robot that can create tickets is not one bad ticket, it is sixty.
        """
        title = _clean(title, 120)
        if not title:
            self.last_error = "nothing to file"
            return None
        if not self.tracker_ready():
            self.last_error = f"no tracker - set {KEY_ENV} in .env"
            return None
        if len(self.filed) >= REPORTS_PER_RUN:
            self.last_error = f"already filed {len(self.filed)} this run"
            return None
        for already in self.filed:
            if _alike(already.title, title) >= SAME_TICKET:
                self.last_error = f"already filed as {already.identifier}"
                return None

        created = self._plane("issues/", {
            "name": f"[aiRon] {title}",
            "description_html": self._body(title, detail),
        })
        if not isinstance(created, dict) or not created.get("id"):
            return None
        filed = Filed(identifier=f"AIRON-{created.get('sequence_id', '?')}",
                      title=title,
                      url=f"https://app.plane.so/{self.workspace}/projects/"
                          f"{self.project_id}/issues/{created['id']}")
        self.filed.append(filed)
        log(f"[project] filed {filed.identifier}: {title!r}")
        return filed

    def _body(self, title: str, detail: str) -> str:
        """What the ticket says, assembled here rather than by the model.

        The model supplies a title and a sentence. Everything else - who
        filed it, when, off which commit, and what the log had just said -
        is added here, because that is the part that makes the ticket worth
        having and the part a model standing in a kitchen cannot know.
        """
        parts = [f"<p>{_clean(detail, 600) or title}</p>",
                 "<p><b>Filed by aiRon</b>, from a conversation. Nothing here "
                 "has been verified by a person.</p>",
                 f"<p>Branch <code>{self.branch() or 'unknown'}</code>, "
                 f"at <code>{self._git('rev-parse', '--short', 'HEAD') or '?'}</code>.</p>"]
        trouble = recent_trouble(6)
        if trouble:
            lines = "".join(f"<li><code>{_clean(line, 200)}</code></li>"
                            for line in trouble)
            parts.append(f"<p>What the log said around then:</p><ul>{lines}</ul>")
        return "".join(parts)

    # ----------------------------------------------------------- both, once

    def look_up(self, question: str = "") -> Findings:
        """Everything aiRon may know about itself, as findings to answer from.

        Deliberately the same bundle whatever is asked. There is no cheap way
        to search a repository for the answer to a spoken question, and the
        honest alternative - handing the model the project's actual state and
        letting it find the answer in there, or admit it cannot - is the one
        the brain already knows how to do safely.
        """
        self.requested = time.monotonic()
        sources = []

        branch, commits = self.branch(), self.commits()
        if branch or commits:
            here = f"aiRon is on branch {branch or 'unknown'}."
            changed = self.uncommitted()
            if changed:
                here += (f" {len(changed)} file(s) changed but not committed: "
                         + ", ".join(line.split()[-1] for line in changed[:5]) + ".")
            sources.append(Source(title="Where the code is now", url="", text=here))
        if commits:
            sources.append(Source(
                title="Recent commits, newest first", url="",
                text=" | ".join(_clean(line, 160) for line in commits)))

        tickets = self.tickets()
        if tickets:
            sources.append(Source(title="Tickets still open", url="",
                                  text=" | ".join(tickets)))
        elif tickets == []:
            sources.append(Source(title="Tickets still open", url="",
                                  text="Nothing is open."))
        else:
            # Said rather than left out, so the model can pass it on. A robot
            # that cannot reach its tracker should say so if asked what is
            # open, not answer from the half of itself that still works.
            sources.append(Source(
                title="The tracker", url="",
                text=f"Could not be reached: {self.last_error}. "
                     "aiRon does not know what is open."))

        trouble = recent_trouble()
        if trouble:
            sources.append(Source(
                title="What has gone wrong since aiRon started", url="",
                text=" | ".join(_clean(line, 160) for line in trouble)))

        return Findings(query=_clean(question, 200) or "aiRon's own project",
                        sources=tuple(sources), fetched=time.monotonic())
