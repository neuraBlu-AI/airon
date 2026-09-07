"""
vision_service - camera in, structured world state out.

Runs on its own thread at camera rate. It never touches the face renderer or
the brain directly: it writes a WorldState to the store and publishes events
on the bus, exactly as spec sections 6-8 require. Face recognition is not here
yet, so everyone gets a temporary id (spec section 12).
"""

from __future__ import annotations

import threading
import time

from ..core import EventBus, EventType, Person, StateStore, WorldState
from .camera import OakCamera, sample_distance
from .tracker import FaceTracker

# How long a face may be missing before aiRon accepts that you have gone. The
# camera's ObjectTracker now bridges detection gaps itself and reports LOST
# before REMOVED, so this no longer has to paper over a fragile detector.
PRESENCE_GRACE_S = 2.5

#: Track states that still count as "this person is here".
PRESENT = ("NEW", "TRACKED", "LOST")
#: States good enough to measure an expression from.
MEASURABLE = ("NEW", "TRACKED")

# Eye contact needs hysteresis or a blink reads as looking away: enter the state
# only once it has held, and leave it only after a sustained absence.
LOOK_ENTER_S = 0.25
LOOK_EXIT_S = 1.0

#: The OAK sometimes resets itself mid-run and comes back as its ROM
#: bootloader. read() returns None rather than raising, so without a watchdog
#: the vision thread spins forever on a device that is no longer there.
STALL_TIMEOUT_S = 5.0
RECONNECT_EVERY_S = 5.0


class VisionService:
    def __init__(self, store: StateStore, bus: EventBus, *,
                 width: int = 640, height: int = 480, fps: int = 30,
                 want_depth: bool = True, force_depth: bool = False,
                 depth_fps: int | None = None, stream_depth: bool = False):
        self.store = store
        self.bus = bus
        self.camera = OakCamera(width, height, fps, want_depth=want_depth,
                                force_depth=force_depth, depth_fps=depth_fps,
                                stream_depth=stream_depth)
        self.tracker = FaceTracker()

        self._track_id: int | None = None
        self._person: Person | None = None
        self._last_seen = 0.0
        self._was_looking = False
        self._look_true_at: float | None = None
        self._look_false_at: float | None = None

        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._stop = threading.Event()
        self.camera_ok = True
        self._dt = 1 / 30
        self._camera_args = dict(width=width, height=height, fps=fps,
                                 want_depth=want_depth, force_depth=force_depth,
                                 depth_fps=depth_fps, stream_depth=stream_depth)

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
        last_frame = time.monotonic()
        while not self._stop.is_set():
            frame = self.camera.read() if self.camera_ok else None
            if frame is None:
                if self.camera_ok and time.monotonic() - last_frame > STALL_TIMEOUT_S:
                    self._on_camera_lost()
                elif not self.camera_ok:
                    self._try_reconnect()
                time.sleep(0.002)
                continue

            last_frame = time.monotonic()
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            self.fps = self.fps * 0.9 + (0.1 / dt) if dt > 0 else self.fps

            self._dt = dt
            obs = self.tracker.update(frame.color, dt) if not self.camera.has_detector else self.tracker.obs
            self._update_world(obs, frame, now)

            with self._lock:
                self._latest = (frame, obs)

    def _on_camera_lost(self) -> None:
        self.camera_ok = False
        self.fps = 0.0
        self._next_retry = time.monotonic() + RECONNECT_EVERY_S
        print("[vision] camera stopped delivering frames - going blind, will retry")
        self.bus.publish(EventType.CAMERA_LOST)
        if self._person is not None:
            self.bus.publish(EventType.PERSON_LEFT, person=self._person.id)
            self._person = None
        self.store.set(WorldState(timestamp=time.time()))
        try:
            self.camera.close()
        except Exception:
            pass

    def _try_reconnect(self) -> None:
        """Rebuild the pipeline from scratch. A reset OAK comes back as its ROM
        bootloader, so this is a fresh boot, not a resumed connection."""
        if time.monotonic() < self._next_retry:
            return
        self._next_retry = time.monotonic() + RECONNECT_EVERY_S
        try:
            self.camera = OakCamera(**self._camera_args)
        except Exception as exc:
            print(f"[vision] reconnect failed: {str(exc)[:90]}")
            return
        self.camera_ok = True
        print(f"[vision] camera back: {self.camera.name}"
              f"{' with depth' if self.camera.has_depth else ' (no depth)'}")
        self.bus.publish(EventType.CAMERA_READY, depth=self.camera.has_depth)

    def _update_world(self, obs, frame, now: float) -> None:
        present = [t for t in frame.tracks if t.status in PRESENT]
        if present or not self.camera.has_detector:
            self._update_person(obs, frame, now, present)
        elif self._person is not None and now - self._last_seen > PRESENCE_GRACE_S:
            self.bus.publish(EventType.PERSON_LEFT, person=self._person.id)
            self._person = None
            self._track_id = None
            self._was_looking = False
            self._look_true_at = self._look_false_at = None

        people = [self._person] if self._person is not None else []
        self.store.set(WorldState(people=people, timestamp=time.time()))

    def _update_person(self, obs, frame, now: float, present: list) -> None:
        track = None
        if present:
            # Whoever is nearest has aiRon's attention; without depth, whoever
            # fills most of the frame.
            ranged = [t for t in present if t.distance_m]
            track = (min(ranged, key=lambda t: t.distance_m) if ranged
                     else max(present, key=lambda t: t.bbox[2] * t.bbox[3]))

        measurable = track is not None and track.status in MEASURABLE
        obs = self.tracker.update(frame.color, self._dt,
                                  bbox=track.bbox if measurable else None) \
            if (track is not None or not self.camera.has_detector) else obs
        if not obs.found and track is None:
            if self._person is not None and now - self._last_seen > PRESENCE_GRACE_S:
                self.bus.publish(EventType.PERSON_LEFT, person=self._person.id)
                self._person = None
                self._track_id = None
            return

        self._last_seen = now
        # The id comes from the camera's tracker, so it survives head turns and
        # brief occlusion - which is what stops aiRon greeting you every 20 s.
        track_id = track.id if track is not None else 0
        if self._person is None or self._track_id != track_id:
            if self._person is not None:
                self.bus.publish(EventType.PERSON_LEFT, person=self._person.id)
            self._person = Person(id=f"person_{track_id:03d}")
            self._track_id = track_id
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
        p.distance_m = (track.distance_m if track is not None and track.distance_m
                        else sample_distance(frame.depth, obs.bbox, frame.color.shape))
        p.position = ("left" if obs.attention_x < -0.25
                      else "right" if obs.attention_x > 0.25 else "center")

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
