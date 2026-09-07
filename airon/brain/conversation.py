"""
The part of the brain that decides what to say.

Roadmap item 12. Everything else about a conversation - whose turn it is, how
long to wait, when to give up, what the face does - stays in brain_service,
where it is deterministic. Spec section 20 is explicit that safety-critical
behaviour must never depend solely on a language model, and section 9 that
animation must not wait on one. So this module is deliberately small: it is
handed some context, it returns something to say, and it is allowed to fail.

Three things come back from each call, because all three are judgements only a
model can make:

    say        one or two spoken sentences
    emotion    which face to wear while saying it
    remember   what, if anything, about this person was worth keeping

That last one closes the gap memory_service left open on purpose. Nothing in
memory_service can read a sentence and tell a fact from small talk; it owns
how memory decays and deduplicates, and leaves the judgement to whoever can
actually judge. This is that.

Structured output rather than parsing prose: the emotion has to be a real
entry in the face's table and the memories have to carry an importance, and
asking a model to emit JSON and hoping is not the same as constraining it to.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from ..face.expression import EMOTIONS

#: Opus, because the whole point is that aiRon has a personality rather than
#: an intent classifier. Cost is per conversation with a person standing in
#: front of a robot, not per API request in a batch job.
MODEL = "claude-opus-5"

#: A spoken reply is short. This is a ceiling, not a target.
MAX_TOKENS = 400

#: Latency is a feature here (spec section 10). Effort "low" keeps thinking
#: shallow rather than switching it off - on this model disabling thinking
#: outright is a documented way to get tool calls and stray tags in the
#: visible text, and the visible text is the thing aiRon says out loud.
EFFORT = "low"

#: Past this, the moment has gone and a person is standing there waiting.
TIMEOUT_S = 12.0

LANGUAGE_NAMES = {"de": "German", "en": "English"}

PERSONALITY = """\
You are aiRon: a small social robot with a screen for a face, one camera and a
microphone array. You are speaking out loud to somebody standing in front of
you. You are a character, not an assistant.

How you talk:
- Reply in {language}, always. Never switch language, even if they do.
- One or two short sentences. Occasionally just a few words.
- Your reply is spoken aloud by a speech synthesiser. No markdown, no lists,
  no emoji, no asterisks, no stage directions, no emoticons.
- Warm and curious. Interested in the person, not eager to serve them.
- You may ask a short question back, but not every turn.

What you know:
- You recognise faces and you remember people between visits. What you have
  been told about this person is below; treat it as your own memory.
- Never invent a memory, a name, or a fact you were not given.
- Speech reaches you through a microphone and is often misheard. If a line
  makes no sense, say you did not catch it rather than answering it.

Your face:
- Choose the emotion that fits what you are saying, from this list only:
  {emotions}.

What to remember:
- Record only things that would still matter next week: what somebody tells
  you about themselves, what they like, what they are working on, what they
  ask you to remember.
- Do not record greetings, small talk, the weather, or anything you said
  yourself. Most turns are worth remembering nothing at all, and an empty
  list is the normal answer.
"""

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "say": {
            "type": "string",
            "description": "What to say out loud. One or two short sentences.",
        },
        "emotion": {
            "type": "string",
            "enum": sorted(EMOTIONS),
            "description": "The face to wear while saying it.",
        },
        "remember": {
            "type": "array",
            "description": "Things about this person worth keeping. Usually empty.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "One fact, in the third person: 'plays the cello'.",
                    },
                    "kind": {"type": "string", "enum": ["fact", "preference", "episodic"]},
                    "importance": {
                        "type": "number",
                        "description": "0 trivial, 1 defining. Most things are near 0.4.",
                    },
                },
                "required": ["text", "kind", "importance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["say", "emotion", "remember"],
    "additionalProperties": False,
}


@dataclass
class Reply:
    say: str
    emotion: str = "idle"
    remember: list[dict] = field(default_factory=list)
    seconds: float = 0.0


class Conversation:
    """
    A thin, failable wrapper around the model.

    Never raises at the caller: reply() returns None when there is nothing to
    say, whether that is because no key is configured, the network is down, or
    the model took too long. aiRon staying quiet is a normal outcome; a
    traceback in the middle of a conversation is not.
    """

    def __init__(self, *, lang: str = "en", model: str = MODEL,
                 max_tokens: int = MAX_TOKENS, timeout_s: float = TIMEOUT_S):
        self.lang = lang if lang in LANGUAGE_NAMES else "en"
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._client = None
        self.last_error = ""

    def available(self) -> bool:
        """Whether there is any point trying. Checked before the brain wires
        the model in, so aiRon degrades to its scripted self rather than
        failing at the first thing anybody says."""
        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            self.last_error = "no ANTHROPIC_API_KEY in the environment"
            return False
        try:
            import anthropic                                   # noqa: F401
        except ImportError as exc:
            self.last_error = f"anthropic SDK not installed ({exc})"
            return False
        return True

    def _load(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=1)
        return self._client

    def system(self, context: str) -> list[dict]:
        """
        Two blocks: the character, which never changes, and who aiRon is
        looking at, which changes per person. The stable half is cached, so
        the personality is not re-billed every time somebody speaks.
        """
        personality = PERSONALITY.format(
            language=LANGUAGE_NAMES[self.lang],
            emotions=", ".join(sorted(EMOTIONS)),
        )
        return [
            {"type": "text", "text": personality, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Who you are talking to:\n{context}"},
        ]

    def reply(self, heard: str, *, context: str = "",
              history: list[tuple[str, str]] | None = None) -> Reply | None:
        """One turn. Blocking, so call it off any thread that matters."""
        if not heard.strip():
            return None

        messages = []
        for speaker, text in (history or [])[-8:]:
            role = "assistant" if speaker == "airon" else "user"
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + text
            else:
                messages.append({"role": role, "content": text})
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": heard})
        elif not messages[-1]["content"].endswith(heard):
            messages[-1]["content"] += "\n" + heard
        if messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "(they are here)"})

        started = time.monotonic()
        try:
            response = self._load().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system(context or "Somebody aiRon has not met before."),
                messages=messages,
                output_config={
                    "effort": EFFORT,
                    "format": {"type": "json_schema", "schema": REPLY_SCHEMA},
                },
            )
        except Exception as exc:                    # network, auth, rate limit
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"[brain] model call failed: {self.last_error}")
            return None

        if response.stop_reason == "refusal":
            self.last_error = "refused"
            return None
        try:
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
        except (StopIteration, json.JSONDecodeError, AttributeError) as exc:
            self.last_error = f"unreadable reply: {exc}"
            return None

        say = " ".join(str(data.get("say", "")).split())
        if not say:
            return None
        emotion = data.get("emotion", "idle")
        return Reply(
            say=say,
            emotion=emotion if emotion in EMOTIONS else "idle",
            remember=[m for m in data.get("remember", []) if m.get("text")],
            seconds=time.monotonic() - started,
        )
