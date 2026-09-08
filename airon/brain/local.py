"""
The same brain, run on the robot instead of in the cloud.

AIRON-9. This exists to be judged rather than to be adopted: the point of
that ticket is a measured comparison, and a comparison you can only read in
a table is not one that settles anything about a robot. It has to be talked
to. So this speaks the same interface as Conversation - same callbacks, same
Reply, same schema - and is chosen with `--brain local`, which makes the two
swappable mid-afternoon rather than mid-refactor.

What is already measured, Gemma 3 4B at Q4_K_M, fully on the GPU:

    free text                 1.75 s a turn, 14.8 tok/s
    with the reply schema     4.37 s a turn,  6.9 tok/s
    the cloud, for comparison 2.80 s a turn

The schema is the expensive half, not the model and not the hardware. Every
token generated is checked against a grammar, and Gemma 3 has a vocabulary of
262,144 tokens to check against, so constraining the output costs more than
producing it. Without the constraint this model is faster than the cloud.
With it, it is slower - and aiRon's brain needs the structure, because the
emotion has to be a real face and the memories have to carry an importance.

What local buys is not speed. It is that the conversation never leaves the
house, and that aiRon keeps talking when the network does not - spec section
17's elderly-care direction being the case where the first of those stops
being a nicety.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..core.log import log
from ..face.expression import EMOTIONS
from .conversation import (LANGUAGE_NAMES, MAX_TOKENS, PERSONALITY, REPLY_SCHEMA,
                           Reply, _Spoken, _tidy)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "llm"

#: The default local model. Chosen for German rather than for benchmarks:
#: small models are markedly weaker in German than in English, and this robot
#: speaks German. Overridable with AIRON_LLM_LOCAL.
MODEL = "gemma-3-4b-it-Q4_K_M.gguf"

MODEL_ENV = "AIRON_LLM_LOCAL"

#: How much of the conversation the model is shown. Shorter than the cloud's,
#: because prefill is not free here - it is arithmetic on this robot's own GPU,
#: competing with the face for the same memory bandwidth.
HISTORY_TURNS = 6

#: Longer than the cloud's twelve seconds. A local turn measured 4.4 s and the
#: whole point of the exercise is to hear what it sounds like, so cutting it
#: off at the cloud's deadline would measure the deadline.
TIMEOUT_S = 30.0


def configured_model() -> Path:
    """Which weights to run: $AIRON_LLM_LOCAL if set, else MODEL."""
    import os

    named = os.environ.get(MODEL_ENV, "").strip()
    if not named:
        return MODEL_DIR / MODEL
    path = Path(named)
    return path if path.is_absolute() else MODEL_DIR / path


class LocalConversation:
    """
    Conversation's twin, backed by llama.cpp on the Jetson's GPU.

    Deliberately the same shape as the cloud one rather than a better shape:
    two things that are meant to be compared should differ in one place only.
    """

    def __init__(self, *, lang: str = "en", model_path: Path | None = None,
                 max_tokens: int | None = None, timeout_s: float | None = None):
        self.lang = lang if lang in LANGUAGE_NAMES else "en"
        self.model_path = Path(model_path) if model_path else configured_model()
        self.max_tokens = max_tokens or MAX_TOKENS
        self.timeout_s = timeout_s or TIMEOUT_S
        self.model = self.model_path.name
        self.last_error = ""
        self._llm = None

    @property
    def summary(self) -> str:
        return (f"{self.model} on the GPU, up to {self.max_tokens} tokens, "
                f"{self.timeout_s:g}s to answer")

    def available(self) -> bool:
        """Whether there is any point trying, checked before the brain wires
        it in, so aiRon degrades to its scripted self rather than failing at
        the first thing anybody says."""
        if not self.model_path.exists():
            self.last_error = f"no local model at {self.model_path}"
            return False
        try:
            import llama_cpp                                   # noqa: F401
        except ImportError as exc:
            self.last_error = f"llama-cpp-python not installed ({exc})"
            return False
        return True

    def system(self, context: str) -> str:
        """One string rather than the cloud's cached blocks - there is no
        cache to write to here, the prefill is simply done again."""
        personality = PERSONALITY.format(
            language=LANGUAGE_NAMES[self.lang],
            emotions=", ".join(sorted(EMOTIONS)),
        )
        return f"{personality}\n\nWho you are talking to:\n{context}"

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama

            started = time.monotonic()
            # 2048 rather than the model's maximum: the personality is about
            # 450 tokens, six turns of history a few hundred more, and a
            # spoken reply is capped at 400. The rest would be KV cache this
            # robot cannot spare - it has 8 GB for everything, and loading
            # these weights at all takes it to within a few hundred MB of the
            # limit with a little swap already in use.
            self._llm = Llama(model_path=str(self.model_path), n_gpu_layers=-1,
                              n_ctx=2048, verbose=False)
            log(f"[brain] {self.model} loaded in {time.monotonic() - started:.1f}s")
        return self._llm

    def warm(self) -> float:
        """Load the weights and build the kernels now, rather than in front of
        the first person to say hello. Returns the seconds it cost."""
        started = time.monotonic()
        self._load().create_chat_completion(
            messages=[{"role": "user", "content": "Hallo"}], max_tokens=1)
        return time.monotonic() - started

    def reply(self, heard: str, *, context: str = "",
              history: list[tuple[str, str]] | None = None,
              on_emotion=None, on_sentence=None) -> Reply | None:
        """One turn. Same contract as Conversation.reply, including that it
        never raises and that None means aiRon says nothing."""
        if not heard.strip():
            return None

        messages = []
        for speaker, text in (history or [])[-HISTORY_TURNS:]:
            role = "assistant" if speaker == "airon" else "user"
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n" + text
            else:
                messages.append({"role": role, "content": text})
        # Gemma's template insists on user/assistant/user/... strictly, and
        # raises rather than coping. A conversation can easily start with
        # aiRon - it greets people before they say anything - so any leading
        # turns of its own are dropped rather than sent.
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": heard})
        elif not messages[-1]["content"].endswith(heard):
            messages[-1]["content"] += "\n" + heard
        # Gemma has no system role at all. The personality is folded into the
        # first thing the person said, which is where a chat template would
        # have put it anyway.
        system = self.system(context or "Somebody aiRon has not met before.")
        messages[0]["content"] = f"{system}\n\n---\n\n{messages[0]['content']}"

        # Load before the clock starts. The weights take about 26 s to reach
        # the GPU the first time, which is not part of answering a question
        # and must not be charged to the timeout - the first turn of the day
        # otherwise times out before the model has said anything. warm()
        # exists so that this has already happened by the time anybody speaks.
        self._load()

        started = time.monotonic()
        spoken = _Spoken()
        first_words = 0.0
        keep_speaking = True

        def offer(sentence: str) -> None:
            nonlocal first_words, keep_speaking
            if on_sentence is None or not keep_speaking or not sentence:
                return
            if not first_words:
                first_words = time.monotonic() - started
            keep_speaking = on_sentence(sentence) is not False

        try:
            stream = self._load().create_chat_completion(
                messages=messages,
                response_format={"type": "json_object", "schema": REPLY_SCHEMA},
                max_tokens=self.max_tokens, temperature=0.7, stream=True)
            told = False
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if not delta:
                    continue
                ready = spoken.feed(delta)
                if not told and spoken.emotion in EMOTIONS:
                    told = True
                    if on_emotion is not None:
                        on_emotion(spoken.emotion)
                for sentence in ready:
                    offer(sentence)
                # There is no server to time this out, so it is timed out here.
                # A local model that has started rambling will happily fill
                # max_tokens while somebody stands waiting.
                if time.monotonic() - started > self.timeout_s:
                    self.last_error = f"gave up after {self.timeout_s:g}s"
                    log(f"[brain] local model too slow: {self.last_error}")
                    break
            offer(spoken.rest())
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            log(f"[brain] local model failed: {self.last_error}")
            if not first_words:
                return None

        try:
            data = json.loads(spoken.raw)
        except json.JSONDecodeError:
            # The grammar should make this impossible, but a stream cut short
            # by the timeout above ends mid-document. Whatever was already
            # spoken still happened and is still worth remembering.
            data = {"say": spoken.say, "emotion": spoken.emotion, "remember": []}

        say = _tidy(str(data.get("say", "")))
        if not say:
            return None
        emotion = data.get("emotion", "idle")
        remember = [{**item, "text": _tidy(str(item["text"]))}
                    for item in data.get("remember", []) if item.get("text")]
        return Reply(
            say=say,
            emotion=emotion if emotion in EMOTIONS else "idle",
            remember=remember,
            seconds=time.monotonic() - started,
            first_words=first_words or (time.monotonic() - started),
        )
