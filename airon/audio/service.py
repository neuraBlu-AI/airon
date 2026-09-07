"""
audio_service - the microphone array, cut into utterances.

Spec section 10: microphone array -> VAD -> speech recognition. This module is
the first two arrows. It never transcribes anything; it hands whole utterances
to speech_service and gets out of the way.

Capture goes through arecord rather than a binding, for the same reason
playback goes through pw-play: it needs no native extension, it is already
proven on this machine, and the audio path stays inspectable from a shell when
something sounds wrong.

The ReSpeaker XVF3800 presents two capture channels at 16 kHz. They are not a
stereo pair. Measured over five seconds of the same room: channel 0 RMS 1765,
channel 1 RMS 262, correlation 0.94 - the same sound, one processed and one
much quieter. Channel 0 is the beamformed, echo-cancelled, noise-suppressed
output the array exists to produce, so that is the one aiRon listens to.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..core import EventBus, EventType

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "asr"
VAD_MODEL = MODEL_DIR / "silero_vad.onnx"

RATE = 16000
CHANNELS = 2          # what the XVF3800 offers; we keep channel 0
CHANNEL = 0

#: How long to wait before trying the microphone again after it drops out,
#: and how far to back off if it keeps failing. USB audio devices wedge -
#: a hub renumbers, the bus glitches, the XVF3800 returns EIO on every read
#: until something resets it. None of that should cost aiRon its hearing for
#: the rest of the run, and none of it should turn into a spin loop against
#: a device that is genuinely gone.
REOPEN_DELAY_S = 2.0
REOPEN_MAX_DELAY_S = 30.0

#: Silero's frame size at 16 kHz. Feeding it exactly this avoids any
#: re-buffering inside the detector.
WINDOW = 512

#: Utterance shape. A name is short, so speech has to be allowed to be short
#: too; the silence figure is what decides how long aiRon waits before deciding
#: you have finished, and half a second is about the pause people leave between
#: sentences without meaning to hand over.
MIN_SPEECH_S = 0.25
MIN_SILENCE_S = 0.5
MAX_SPEECH_S = 12.0

#: How long after aiRon stops talking before it trusts the microphone again.
#: The XVF3800 can cancel its own echo, but only when it is given the played
#: audio as a reference over its USB playback endpoint - and aiRon's voice
#: currently goes out of the display's speakers instead, where the array never
#: hears about it. So Phase 1 is half duplex: aiRon does not listen to itself.
SPEAKING_TAIL_S = 0.35


def find_capture_device(prefer: str = "respeaker") -> str:
    """
    The array's ALSA device, found by name.

    Card numbers move when USB devices are plugged in a different order, and a
    robot that goes deaf because a hub enumerated differently is not a robot
    anybody will trust.
    """
    try:
        listing = subprocess.run(["arecord", "-l"], capture_output=True,
                                 text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return "default"
    for line in listing.splitlines():
        match = re.match(r"card \d+: (\S+)", line)
        if match and (prefer in line.lower() or "array" in line.lower()):
            return f"plughw:{match.group(1)},0"
    return "default"


class AudioService:
    """
    Listens continuously and publishes one utterance at a time.

    Runs on its own thread. `utterances` is the queue speech_service drains;
    everything else here is for the face, which wants to know that somebody is
    talking long before it knows what they said.
    """

    def __init__(self, bus: EventBus, *, device: str | None = None,
                 vad_model: Path | None = None,
                 is_muted: Callable[[], bool] | None = None):
        self.bus = bus
        self.device = device or find_capture_device()
        self.vad_model = vad_model or VAD_MODEL
        self.is_muted = is_muted or (lambda: False)

        self.utterances: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._level = 0.0
        self._hearing_voice = False
        self._clipped = 0

        self._proc: subprocess.Popen | None = None
        self._muted_until = 0.0
        self._vad = None
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._stop = threading.Event()

    # -------------------------------------------------------------- public

    def available(self) -> bool:
        return self.vad_model.exists()

    @property
    def level(self) -> float:
        """Loudness right now, 0..1 - the face's "someone is talking" signal."""
        with self._lock:
            return self._level

    @property
    def hearing_voice(self) -> bool:
        with self._lock:
            return self._hearing_voice

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc                   # may be cleared by the audio thread
        if proc is not None:
            try:
                proc.terminate()            # unblock the read
            except OSError:
                pass
        self._thread.join(timeout=3.0)

    # --------------------------------------------------------------- inner

    def _build_vad(self):
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(self.vad_model)
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_speech_duration = MIN_SPEECH_S
        config.silero_vad.min_silence_duration = MIN_SILENCE_S
        config.silero_vad.max_speech_duration = MAX_SPEECH_S
        config.sample_rate = RATE
        config.num_threads = 1
        return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    def _open(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                ["arecord", "-D", self.device, "-f", "S16_LE", "-c", str(CHANNELS),
                 "-r", str(RATE), "-t", "raw", "-q"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            print(f"[audio] cannot open {self.device}: {exc}")
            return False
        return True

    def _why_it_stopped(self) -> str:
        """Whatever arecord said on its way out. Kept, rather than sent to
        /dev/null, because "read error: Input/output error" is the difference
        between a wedged USB device and a robot that is merely in a quiet
        room, and the two look identical from the far end of the pipe."""
        if self._proc is None:
            return ""
        # Make sure it is actually dead first: reading stderr of a process
        # that is still alive blocks until it decides to say something, and
        # this runs on the thread that carries every word aiRon hears.
        try:
            self._proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        except OSError:
            return ""
        try:
            err = (self._proc.stderr.read() or b"").decode("utf-8", "replace")
        except (OSError, ValueError):
            return ""
        lines = [l.strip() for l in err.splitlines()
                 if l.strip() and not l.startswith("Recording raw data")]
        return lines[-1] if lines else ""

    def _close(self) -> None:
        if self._proc is None:
            return
        for stream in (self._proc.stdout, self._proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._proc = None

    def _run(self) -> None:
        self._vad = self._build_vad()
        nbytes = WINDOW * CHANNELS * 2
        muted_until_quiet = False
        delay = REOPEN_DELAY_S
        announced = False

        while not self._stop.is_set():
            if self._proc is None:
                if not self._open():
                    return
                if not announced:
                    print(f"[audio] listening on {self.device}")
                    announced = True

            raw = self._proc.stdout.read(nbytes)
            if not raw or len(raw) < nbytes:
                # The device stopped. Say why, drop what the detector had
                # half-built, and try again - deafness should last seconds,
                # not the rest of the run.
                why = self._why_it_stopped()
                self._close()
                self._set_level(0.0)
                if self._stop.is_set():
                    break
                print(f"[audio] microphone stopped delivering samples"
                      f"{': ' + why if why else ''}; retrying in {delay:.0f}s")
                self._vad.reset()
                muted_until_quiet = False
                if self._stop.wait(delay):
                    break
                delay = min(delay * 2, REOPEN_MAX_DELAY_S)
                continue

            if delay != REOPEN_DELAY_S:
                print("[audio] microphone back")
                delay = REOPEN_DELAY_S

            block = np.frombuffer(raw, np.int16).reshape(-1, CHANNELS)
            mono = block[:, CHANNEL].astype(np.float32) / 32768.0
            if np.abs(mono).max() > 0.99:
                self._clipped += 1

            # Half duplex: keep draining the pipe, but hear nothing while aiRon
            # is talking, and throw away whatever the detector had half-built.
            # The tail matters as much as the mute - the speakers are still
            # ringing, and the room still echoing, after the last sample plays.
            if self.is_muted():
                self._muted_until = time.monotonic() + SPEAKING_TAIL_S
            if time.monotonic() < self._muted_until:
                muted_until_quiet = True
                self._set_level(0.0)
                continue
            if muted_until_quiet:
                muted_until_quiet = False
                self._vad.reset()

            self._set_level(float(np.sqrt((mono ** 2).mean())))
            self._vad.accept_waveform(mono)
            self._note_voice(self._vad.is_speech_detected())

            while not self._vad.empty():
                segment = np.asarray(self._vad.front.samples, dtype=np.float32)
                self._vad.pop()
                self._offer(segment)

        self._set_level(0.0)
        self._close()

    def _offer(self, segment: np.ndarray) -> None:
        """Queue an utterance, dropping the oldest if nothing is draining."""
        try:
            self.utterances.put_nowait(segment)
        except queue.Full:
            try:
                self.utterances.get_nowait()
                self.utterances.put_nowait(segment)
            except queue.Empty:
                pass

    def _note_voice(self, speaking: bool) -> None:
        with self._lock:
            changed = speaking != self._hearing_voice
            self._hearing_voice = speaking
        if changed:
            self.bus.publish(EventType.VOICE_STARTED if speaking
                             else EventType.VOICE_STOPPED)

    def _set_level(self, level: float) -> None:
        with self._lock:
            # Rise fast so the face reacts on the first syllable, fall slowly
            # so it does not flicker between words.
            self._level = max(level, self._level * 0.8)
