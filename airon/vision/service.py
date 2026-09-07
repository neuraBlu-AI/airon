"""
vision_service - camera in, structured world state out.

Runs on its own thread at camera rate. It never touches the face renderer or
the brain directly: it writes a WorldState to the store and publishes events
on the bus, exactly as spec sections 6-8 require.

It also decides who it is looking at. Spec section 12: a face becomes an
embedding, the embedding is matched against the local gallery, and the person
keeps a persistent id if they are known - or a temporary guest id if they are
not, until somebody introduces them.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque

import numpy as np

from ..core import EventBus, EventType, Person, StateStore, WorldState
from ..identity import FaceRecognizer, Gallery, Identity
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

# Identity is decided by vote, not by one lucky frame. A single embedding is a
# coin flip against a bad blink or a half-turned head; three agreeing samples
# out of the last five settle it, which at four samples a second means aiRon
# knows your name inside about a second of seeing your face properly.
IDENTITY_VOTES = 5
IDENTITY_AGREE = 3

# How long to keep quiet before admitting we do not know who this is. Someone
# walking in while looking away can take a couple of seconds to present a face
# the gate will accept, and announcing a stranger before then is just wrong.
# Sampling continues afterwards: turn around and aiRon still catches up.
IDENTITY_TIMEOUT_S = 4.0

# Views of the current person, kept in case they introduce themselves. By the
# time somebody has finished saying "my name is Pierre", aiRon has been
# quietly collecting gated views of them for several seconds - so enrolment
# needs no second capture pass and nobody has to hold still for a camera.
ENROL_BUFFER = 24
ENROL_MIN_VIEWS = 5
#: A view that says nothing new is not worth a slot in that buffer.
ENROL_NOVELTY = 0.97


class VisionService:
    def __init__(self, store: StateStore, bus: EventBus, *,
                 width: int = 640, height: int = 480, fps: int = 30,
                 want_depth: bool = True, force_depth: bool = False,
                 depth_fps: int | None = None, stream_depth: bool = False,
                 recognise: bool = True, gallery: Gallery | None = None):
        self.store = store
        self.bus = bus
        self.camera = OakCamera(width, height, fps, want_depth=want_depth,
                                force_depth=force_depth, depth_fps=depth_fps,
                                stream_depth=stream_depth)
        self.tracker = FaceTracker()

        self.gallery = gallery if gallery is not None else Gallery()
        self._want_recognition = recognise
        self.recognizer = self._make_recognizer()

        self._votes: deque[str | None] = deque(maxlen=IDENTITY_VOTES)
        self._recent: deque = deque(maxlen=ENROL_BUFFER)
        self._identity_lock = threading.Lock()
        self._identified = False
        self._said_unknown = False
        self._identify_by = 0.0

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

    def _make_recognizer(self) -> FaceRecognizer | None:
        return (FaceRecognizer(self.camera)
                if self._want_recognition and self.camera.has_recognizer else None)

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
        self.recognizer = self._make_recognizer()
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

    # ------------------------------------------------------------- identity

    def _begin_identification(self, now: float) -> None:
        """Start over: a new track is a new question, even if it is you again."""
        self._votes.clear()
        with self._identity_lock:
            self._recent.clear()
        self._identified = False
        self._said_unknown = False
        self._identify_by = now + IDENTITY_TIMEOUT_S
        if self.recognizer is not None:
            self.recognizer.reset()

    def _identify(self, person: Person, frame, obs, now: float) -> None:
        if self.recognizer is None:
            if not self._said_unknown:
                self._said_unknown = True
                self.bus.publish(EventType.UNKNOWN_PERSON_DETECTED, person=person.id)
            return

        embedding = self.recognizer.update(frame.color, obs.bbox, now)
        if embedding is not None:
            self._remember_view(embedding)
            match = self.gallery.match(embedding)
            self._votes.append(match.identity.person_id if match.accepted else None)
            # Teach the gallery only once the vote has settled, and only about
            # the person it settled on. One confident-looking sample is exactly
            # how a gallery gets quietly poisoned with the wrong face.
            if (self._identified and person.recognized and match.accepted
                    and match.identity.person_id == person.id):
                self.gallery.reinforce(match.identity, embedding, match.score)

        if not self._identified:
            self._decide(person, now)

    def _remember_view(self, embedding) -> None:
        """Bank a distinct view of whoever is here, for a possible enrolment."""
        with self._identity_lock:
            if self._recent and float(np.max(np.stack(self._recent) @ embedding)) > ENROL_NOVELTY:
                return
            self._recent.append(embedding)

    def enrol_current(self, name: str) -> Identity | None:
        """
        Introduce whoever aiRon is looking at, under `name`.

        Called from the brain's thread while the vision thread keeps running,
        so the person and the view buffer are taken under a lock and the rest
        is done outside it. Returns None if nobody is there, or if too few
        usable views have been gathered to recognise them by later.
        """
        with self._identity_lock:
            views = list(self._recent)
            person = self._person
        if person is None or len(views) < ENROL_MIN_VIEWS:
            return None

        identity = self.gallery.enrol(name, np.stack(views))
        self.gallery.seen(identity)
        with self._identity_lock:
            was = person.id
            if self._person is person:      # still the same sighting
                person.id = identity.person_id
                person.name = identity.name
                person.recognized = True
                self._identified = True
                self._said_unknown = True
        self.bus.publish(EventType.PERSON_NAMED, person=identity.person_id,
                         name=identity.name, was=was, views=len(views))
        return identity

    def _decide(self, person: Person, now: float) -> None:
        votes = [v for v in self._votes if v is not None]
        if votes:
            winner, agreed = Counter(votes).most_common(1)[0]
            identity = self.gallery.identities.get(winner)
            if agreed >= IDENTITY_AGREE and identity is not None:
                was, person.id = person.id, identity.person_id
                person.name = identity.name
                person.recognized = True
                self._identified = True
                self.gallery.seen(identity)
                self.bus.publish(EventType.KNOWN_PERSON_DETECTED,
                                 person=identity.person_id, name=identity.name, was=was)
                return

        # Not known yet - say so once, then keep looking. Someone who walks in
        # facing away should still be greeted by name when they turn round.
        if not self._said_unknown and now >= self._identify_by:
            self._said_unknown = True
            self.bus.publish(EventType.UNKNOWN_PERSON_DETECTED, person=person.id)

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
            # A guest id until the gallery says otherwise (spec section 12).
            # Keeping the two shapes visibly different - guest_007 against
            # person_001 - means nothing downstream can mistake a session for
            # an identity.
            self._person = Person(id=f"guest_{track_id:03d}")
            self._track_id = track_id
            self._begin_identification(now)
            self.bus.publish(EventType.PERSON_ENTERED, person=self._person.id)

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

        # Only measure identity off a live detection. A LOST track is the
        # camera predicting where a face probably went, and embedding that
        # predicted box means embedding whatever is actually there instead.
        if measurable:
            self._identify(p, frame, obs, now)
