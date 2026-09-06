"""
vision_service - camera in, structured world state out.

Runs on its own thread at camera rate. It never touches the face renderer or
the brain directly: it writes a WorldState to the store and publishes events
on the bus, exactly as spec sections 6-8 require. Face recognition is not here
yet, so everyone gets a temporary id (spec section 12).
"""

from __future__ import annotations

import itertools
import threading
import time

from ..core import EventBus, EventType, Person, StateStore, WorldState
from .camera import OakCamera, sample_distance
from .tracker import FaceTracker

# How long a face may be missing before aiRon accepts that you have gone.
# Haar detection drops frames when you turn your head; leaving takes longer.
PRESENCE_GRACE_S = 1.2

# Eye contact needs hysteresis or a blink reads as looking away: enter the state
# only once it has held, and leave it only after a sustained absence.
LOOK_ENTER_S = 0.25
LOOK_EXIT_S = 1.0


class VisionService:
    def __init__(self, store: StateStore, bus: EventBus, *,
                 width: int = 640, height: int = 480, fps: int = 30,
                 want_depth: bool = True, force_depth: bool = False):
        self.store = store
        self.bus = bus
        self.camera = OakCamera(width, height, fps, want_depth=want_depth,
                                force_depth=force_depth)
        self.tracker = FaceTracker()

        self._ids = itertools.count(1)
        self._person: Person | None = None
        self._last_seen = 0.0
        self._was_looking = False
        self._look_true_at: float | None = None
        self._look_false_at: float | None = None

        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._stop = threading.Event()

        # Latest frame kept for tools/vision_bench.py. The face never reads it.
        self._lock = threading.Lock()
        self._latest = None
        self.fps = 0.0

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self.camera.close()

    @property
    def latest_frame(self):
        """(Frame, FaceObservation) or None - debug tooling only."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------ loop

    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.002)
                continue

            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            self.fps = self.fps * 0.9 + (0.1 / dt) if dt > 0 else self.fps

            obs = self.tracker.update(frame.color, dt)
            self._update_world(obs, frame, now)

            with self._lock:
                self._latest = (frame, obs)

    def _update_world(self, obs, frame, now: float) -> None:
        if obs.found:
            self._last_seen = now
            if self._person is None:
                self._person = Person(id=f"person_{next(self._ids):03d}")
                self.bus.publish(EventType.PERSON_ENTERED, person=self._person.id)
                self.bus.publish(EventType.UNKNOWN_PERSON_DETECTED, person=self._person.id)

            p = self._person
            p.bbox = obs.bbox
            p.attention_x = obs.attention_x
            p.attention_y = obs.attention_y
            p.head_roll = obs.head_roll
            p.eyes_open = obs.eyes_open
            p.mouth_curve = obs.mouth_curve
            p.mouth_open = obs.mouth_open
            p.last_seen = time.time()
            p.distance_m = sample_distance(frame.depth, obs.bbox, frame.color.shape)
            p.position = ("left" if obs.attention_x < -0.25
                          else "right" if obs.attention_x > 0.25 else "center")

            # Frontal enough that both eyes are visible, and roughly centred:
            # a rough proxy until real gaze estimation lands.
            raw_looking = obs.eyes_open > 0.5 and abs(obs.attention_x) < 0.6
            if raw_looking:
                self._look_true_at = self._look_true_at or now
                self._look_false_at = None
            else:
                self._look_false_at = self._look_false_at or now
                self._look_true_at = None

            if (not self._was_looking and self._look_true_at
                    and now - self._look_true_at >= LOOK_ENTER_S):
                self._was_looking = True
                self.bus.publish(EventType.PERSON_LOOKING_AT_AIRON, person=p.id)
            elif (self._was_looking and self._look_false_at
                    and now - self._look_false_at >= LOOK_EXIT_S):
                self._was_looking = False
            p.looking_at_airon = self._was_looking

        elif self._person is not None and now - self._last_seen > PRESENCE_GRACE_S:
            self.bus.publish(EventType.PERSON_LEFT, person=self._person.id)
            self._person = None
            self._was_looking = False
            self._look_true_at = self._look_false_at = None

        people = [self._person] if self._person is not None else []
        self.store.set(WorldState(people=people, timestamp=time.time()))
