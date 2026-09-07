"""
speech_service's other half: turning an utterance into words.

Whisper, exported to ONNX and run through sherpa-onnx. That combination was
chosen because onnxruntime was already on this Jetson and PyTorch is not:
nothing here needs the GPU, which stays free for the LLM later. Measured on
this machine with the multilingual small model quantised to int8, a one to
three second utterance decodes in 1.7 s on average and 2.8 s at worst, after
a one-off 4.2 s load. Four threads; six is no faster, the CPU is already
saturated at four.

Two things learned from running it against this microphone, both of which the
code depends on:

The language is pinned, never detected. Whisper's language detector needs more
audio than a name gives it - asked to identify the single word "Anna" it
answered Japanese, twice, and transcribed the Japanese it expected. Told which
language to use, it wrote "Anna" both times. Names are exactly the short
utterances aiRon cares most about, so detection is the wrong trade here.

Silence is not silent. Pointed at an empty room Whisper does not return an
empty string; it returns "(birds chirping)", "*schreit*", "[speaking in
foreign language]", or one of a handful of stock phrases learned from
subtitle training data. The VAD keeps most of that away, but not all, so
anything that is only an annotation is discarded rather than handed to the
brain as something a human said.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np

from ..core import EventBus, EventType

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "asr"

#: Multilingual, so German and English both work without swapping models.
#: Change this and re-run tools/fetch_speech_models.py.
#:
#: Measured German, twelve sentences taken from a real session, each scaled to
#: 0.05 RMS - the level a person across the room actually produces here - and
#: mixed with noise at three seeds per point. Word error rate, and how many of
#: the 36 came back exactly right:
#:
#:      SNR      base            small
#:      30 dB    19.5%  17/36    11.3%  23/36
#:      20 dB    23.0%  15/36    11.9%  22/36
#:      15 dB    21.3%  13/36    17.0%  20/36
#:      10 dB    35.7%   3/36    23.5%  15/36
#:       5 dB    62.5%   0/36    49.7%   6/36
#:
#: small wins everywhere and wins by more as the room gets worse, which is the
#: opposite of a marginal call: at 10 dB base gets three utterances out of
#: thirty-six right and small gets fifteen. It costs 1.2 s per utterance.
#:
#: An earlier note here said small "transcribes unusual names noticeably
#: better". That does not hold and has been removed. Neither model gets André,
#: Pierre or aiRon right at any level tested, and small is the one that turns
#: Elizabeth into "Elitabeth". Names need a way to be corrected, not a bigger
#: model.
#:
#: The same measurement run on Piper's raw output, which sits near full scale,
#: shows the two models level. That condition is not worth optimising for -
#: nobody speaks into this microphone at full scale - and it is why the first
#: version of this comparison reached the wrong conclusion.
WHISPER = "small"

#: Transcripts that are nothing but a bracketed annotation - "(laughs)",
#: "[speaking in foreign language]", "*schreit*". All observed from this
#: microphone within the first minute of use.
ANNOTATION = re.compile(r"^\s*[(\[*][^)\]*]*[)\]*]\s*$")

#: Whisper's stock hallucinations on near-silence, a well known artefact of
#: training on subtitles. Compared casefolded and stripped of punctuation.
FILLER = {
    "thank you", "thanks for watching", "thanks for watching!",
    "you", "bye", "bye bye", "oh", "hmm", "uh", "um",
    "vielen dank", "danke", "untertitel von stephanie geiges",
    "untertitelung des zdf", "untertitel im auftrag des zdf",
}


def is_speech(text: str) -> bool:
    """Did a human actually say this, or did the model fill a silence?"""
    stripped = text.strip()
    if len(stripped) < 2 or ANNOTATION.match(stripped):
        return False
    bare = re.sub(r"[^\w\s]", "", stripped, flags=re.UNICODE).strip().casefold()
    return bool(bare) and bare not in FILLER


class Listener:
    """
    Drains audio_service's utterance queue and publishes what was said.

    Its own thread, because a decode takes about a second and neither the
    microphone nor the face may wait for it.
    """

    def __init__(self, bus: EventBus, audio, *, lang: str = "en",
                 model_dir: Path | None = None, threads: int = 4):
        self.bus = bus
        self.audio = audio
        self.lang = lang
        self.threads = threads
        self.model_dir = (model_dir or MODEL_DIR) / f"sherpa-onnx-whisper-{WHISPER}"

        self._recognizer = None
        self._thread = threading.Thread(target=self._run, name="listener", daemon=True)
        self._stop = threading.Event()
        self.last_text = ""
        #: True while an utterance is being turned into words. The brain waits
        #: on this: a decode takes a second or two, so the second half of a
        #: sentence arrives well after the person stopped saying it, and
        #: answering before it lands is answering half a question.
        self.decoding = False

    def available(self) -> bool:
        return all((self.model_dir / f).exists() for f in (
            f"{WHISPER}-encoder.int8.onnx", f"{WHISPER}-decoder.int8.onnx",
            f"{WHISPER}-tokens.txt"))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def transcribe(self, audio: np.ndarray) -> str:
        """One utterance to text. Blocking; called on the listener thread."""
        recognizer = self._load()
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        recognizer.decode_stream(stream)
        return stream.result.text.strip()

    # --------------------------------------------------------------- inner

    def _load(self):
        if self._recognizer is None:
            import sherpa_onnx

            started = time.monotonic()
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(self.model_dir / f"{WHISPER}-encoder.int8.onnx"),
                decoder=str(self.model_dir / f"{WHISPER}-decoder.int8.onnx"),
                tokens=str(self.model_dir / f"{WHISPER}-tokens.txt"),
                num_threads=self.threads, language=self.lang, task="transcribe")
            # The first decode costs about four times the rest, so spend it now
            # rather than on the first thing anybody says.
            warm = self._recognizer.create_stream()
            warm.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
            self._recognizer.decode_stream(warm)
            print(f"[listener] whisper-{WHISPER} ready in "
                  f"{time.monotonic() - started:.1f}s, listening in {self.lang}")
        return self._recognizer

    def _run(self) -> None:
        self._load()
        while not self._stop.is_set():
            # Flagged busy *before* the utterance leaves the queue, so the
            # two signals overlap. Taking it first would leave an instant when
            # the queue is empty and nothing is decoding, and the brain reading
            # exactly then concludes the person has finished talking - which is
            # how half a sentence gets answered on its own.
            if self.audio.utterances.empty():
                if self._stop.wait(0.05):
                    break
                continue
            self.decoding = True
            try:
                audio = self.audio.utterances.get_nowait()
            except Exception:
                self.decoding = False
                continue
            try:
                text = self.transcribe(audio)
            except Exception as exc:
                print(f"[listener] decode failed: {str(exc)[:120]}")
                continue
            finally:
                self.decoding = False
            if not is_speech(text):
                continue
            self.last_text = text
            self.bus.publish(EventType.HEARD, text=text, lang=self.lang,
                             seconds=round(len(audio) / 16000, 2))
