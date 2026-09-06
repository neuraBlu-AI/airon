"""
speech_service - aiRon's voice.

Piper (ONNX, CPU) synthesises German and English locally: no network, no API
key, and nothing competing with the LLM for the GPU later. Measured on this
Jetson: about 0.25-0.3x realtime, so a sentence is ready in well under a second,
after a one-off ~3 s model load per voice.

Synthesis and playback run on their own thread. The face never waits for either
- it reads `level` whenever it paints, which is spec section 9's requirement
that animation stay independent of everything slow.
"""

from __future__ import annotations

import io
import queue
import re
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import EventBus, EventType

VOICE_DIR = Path(__file__).resolve().parent.parent.parent / "voices"

#: language -> Piper voice. Both are "medium" quality: the "high" voices are
#: noticeably slower to synthesise for very little gain through small speakers.
VOICES = {
    "de": "de_DE-thorsten-medium",
    "en": "en_US-lessac-medium",
}

#: pw-play buffers before the first sample is audible, so the mouth would lead
#: the voice without this. Tuned by ear; adjust if lips drift from speech.
PLAYBACK_LATENCY_S = 0.12

#: Mouth envelope resolution. ~23 ms at 22.05 kHz - finer than the eye can see.
ENVELOPE_FRAME = 512

_UMLAUTS = set("äöüß")
_GERMAN_WORDS = {
    "ich", "du", "der", "die", "das", "und", "ist", "nicht", "mit", "wie",
    "hallo", "guten", "danke", "bitte", "wir", "sie", "ein", "eine", "auf",
    "für", "was", "wer", "wo", "wann", "warum", "heute", "morgen", "ja", "nein",
}


def detect_language(text: str, default: str = "en") -> str:
    """
    Pick a voice for a line of text.

    A deliberately small heuristic, not a language identifier: umlauts are
    decisive, otherwise two or more common German words tip it. Anything the
    brain generates should pass an explicit language and skip this entirely.
    """
    lowered = text.lower()
    if any(ch in _UMLAUTS for ch in lowered):
        return "de"
    words = set(re.findall(r"[a-zäöüß]+", lowered))
    return "de" if len(words & _GERMAN_WORDS) >= 2 else default


@dataclass
class Utterance:
    text: str
    lang: str


class SpeechService:
    def __init__(self, bus: EventBus, default_lang: str = "en",
                 voice_dir: Path | None = None, volume: float = 0.9):
        self.bus = bus
        self.default_lang = default_lang
        self.voice_dir = voice_dir or VOICE_DIR
        self.volume = volume

        self._voices: dict[str, object] = {}
        self._queue: queue.Queue[Utterance | None] = queue.Queue()
        self._lock = threading.Lock()
        self._speaking = False
        self._level = 0.0

        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._stop = threading.Event()

    # -------------------------------------------------------------- public

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking

    @property
    def level(self) -> float:
        """How open the mouth should be right now, 0..1, from the audio itself."""
        with self._lock:
            return self._level

    def start(self) -> None:
        self._thread.start()

    def say(self, text: str, lang: str | None = None) -> None:
        """Queue a line. Returns immediately; nothing here blocks the caller."""
        text = text.strip()
        if text:
            self._queue.put(Utterance(text, lang or detect_language(text, self.default_lang)))

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def available(self) -> list[str]:
        return [lang for lang, name in VOICES.items()
                if (self.voice_dir / f"{name}.onnx").exists()]

    # --------------------------------------------------------------- inner

    def _voice(self, lang: str):
        """Load on first use - each voice costs about 3 s and 61 MB."""
        if lang not in self._voices:
            from piper import PiperVoice

            model = self.voice_dir / f"{VOICES[lang]}.onnx"
            if not model.exists():
                raise FileNotFoundError(
                    f"missing voice {model.name}; run tools/fetch_voices.py")
            t0 = time.monotonic()
            self._voices[lang] = PiperVoice.load(model)
            print(f"[speech] loaded {VOICES[lang]} in {time.monotonic() - t0:.1f}s")
        return self._voices[lang]

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            try:
                self._speak(item)
            except Exception as exc:
                print(f"[speech] failed to say {item.text!r}: {str(exc)[:100]}")
                with self._lock:
                    self._speaking, self._level = False, 0.0

    def _speak(self, utterance: Utterance) -> None:
        lang = utterance.lang if utterance.lang in VOICES else self.default_lang
        voice = self._voice(lang)

        chunks = [c.audio_int16_array for c in voice.synthesize(utterance.text)]
        if not chunks:
            return
        audio = np.concatenate(chunks)
        rate = voice.config.sample_rate
        envelope = self._envelope(audio)

        self.bus.publish(EventType.SPEECH_STARTED, text=utterance.text, lang=lang)
        with self._lock:
            self._speaking = True

        proc = subprocess.Popen(
            ["pw-play", "--volume", str(self.volume), "-"],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # pw-play in this build cannot read raw PCM, so hand it a WAV stream.
        # Writing happens on its own thread: a big utterance would otherwise
        # block on the pipe buffer and freeze the mouth mid-sentence.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(audio.tobytes())
        writer = threading.Thread(
            target=self._feed, args=(proc, buffer.getvalue()), daemon=True)
        writer.start()

        started = time.monotonic()
        seconds_per_frame = ENVELOPE_FRAME / rate
        while proc.poll() is None and not self._stop.is_set():
            elapsed = time.monotonic() - started - PLAYBACK_LATENCY_S
            index = int(elapsed / seconds_per_frame)
            with self._lock:
                self._level = float(envelope[index]) if 0 <= index < len(envelope) else 0.0
            time.sleep(0.008)

        writer.join(timeout=1.0)
        with self._lock:
            self._speaking, self._level = False, 0.0
        self.bus.publish(EventType.SPEECH_FINISHED, text=utterance.text)

    @staticmethod
    def _feed(proc, payload: bytes) -> None:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass

    @staticmethod
    def _envelope(audio: np.ndarray) -> np.ndarray:
        """
        Per-frame loudness, normalised - this is what opens aiRon's mouth.

        Real phoneme alignments are available from Piper and would be better
        still, but amplitude already tracks syllables closely enough to read as
        speech, and it works for any voice without a phoneme-to-viseme map.
        """
        frames = len(audio) // ENVELOPE_FRAME
        if frames < 1:
            return np.zeros(1, dtype=np.float32)
        block = audio[:frames * ENVELOPE_FRAME].astype(np.float32).reshape(frames, -1)
        rms = np.sqrt((block ** 2).mean(axis=1))
        peak = np.percentile(rms, 95)
        if peak <= 0:
            return np.zeros(frames, dtype=np.float32)
        # Gamma < 1 lifts the quiet parts so consonants still move the mouth.
        return np.clip(rms / peak, 0.0, 1.0) ** 0.7
