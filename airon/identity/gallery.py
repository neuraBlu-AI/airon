"""
Who aiRon knows.

Spec section 12: face embedding -> compare against a local identity database ->
persistent person id. This is that database.

It never leaves the Jetson and it stores no pictures. An identity is a name and
a handful of 256-float vectors, so "delete Pierre" really is the whole of
deleting Pierre - there is no image cache to also remember to clear. That is a
deliberately small privacy story, chosen because it is simple enough to explain
to the person being recognised.

Matching is cosine similarity, and an identity scores as its single best
vector rather than as the mean of them. One face measurably changes between
arm's length and across the room, and between window light and a lamp; keeping
several vectors per person and taking the best is what absorbs that, where a
mean would just blur the person into everyone else.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GALLERY_DIR = Path(__file__).resolve().parent.parent.parent / "known_faces"
INDEX_FILE = "gallery.json"

#: Dimensionality of face-reidentification-retail-0095's output.
EMBEDDING_DIM = 256

#: Below this cosine score, a face is nobody we know.
#:
#: Measured on this OAK-D Lite over 96 gated samples of a single face: mean
#: 0.90, 5th percentile 0.77. So 0.65 sits clear of a genuine match while
#: leaving room for bad light. It has NOT been tuned against a second enrolled
#: person, because there was only one person to enrol - run
#: tools/identity_bench.py once two people are in the gallery and move this if
#: their distributions overlap.
MATCH_THRESHOLD = 0.65

#: A match must also beat the runner-up by this much. If two people both score
#: 0.70 the gallery cannot actually tell them apart, and guessing is worse than
#: staying quiet: aiRon calling you by your brother's name is memorable in the
#: wrong way.
MATCH_MARGIN = 0.06

#: Online learning. A confident sighting can add its vector to the person it
#: matched, which is what lets aiRon keep up with a haircut or a new pair of
#: glasses over weeks. It is gated well above MATCH_THRESHOLD so a borderline
#: match can never teach the gallery something false, and only kept when the
#: vector says something the stored ones do not - otherwise sitting still for
#: a minute would fill the record with sixty copies of one pose.
LEARN_THRESHOLD = 0.80
LEARN_NOVELTY = 0.92
MAX_VECTORS = 32


@dataclass
class Identity:
    """One person aiRon has been introduced to."""

    person_id: str
    name: str
    vectors: np.ndarray                       # (N, 256) float32, each unit length
    created: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_seen: int = 0

    def similarity(self, embedding: np.ndarray) -> float:
        """Best cosine score this person can manage against `embedding`."""
        if self.vectors.size == 0:
            return -1.0
        return float(np.max(self.vectors @ embedding))


@dataclass
class Match:
    identity: Identity | None
    score: float = -1.0
    runner_up: float = -1.0

    @property
    def accepted(self) -> bool:
        """Confident enough to say a name out loud."""
        return (self.identity is not None
                and self.score >= MATCH_THRESHOLD
                and self.score - self.runner_up >= MATCH_MARGIN)


def normalise(vectors: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is a cosine."""
    v = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-9)


class Gallery:
    """
    The identity database on disk: one JSON index plus one .npy per person.

    Two plain file types rather than a database, because the entire content is
    a few kilobytes and being able to read the index in a text editor - and
    delete a person with rm - is worth more here than query power.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else GALLERY_DIR
        self._lock = threading.Lock()
        self.identities: dict[str, Identity] = {}
        self.load()

    # ------------------------------------------------------------------ io

    def load(self) -> None:
        index = self.path / INDEX_FILE
        if not index.exists():
            return
        try:
            records = json.loads(index.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[gallery] cannot read {index}: {exc}")
            return

        loaded: dict[str, Identity] = {}
        for record in records:
            person_id = record["person_id"]
            vectors_file = self.path / f"{person_id}.npy"
            try:
                vectors = normalise(np.load(vectors_file))
            except OSError as exc:
                print(f"[gallery] {person_id} has no vectors ({exc}); skipping")
                continue
            if vectors.shape[1] != EMBEDDING_DIM:
                print(f"[gallery] {person_id} has {vectors.shape[1]}-d vectors, "
                      f"expected {EMBEDDING_DIM}; skipping")
                continue
            loaded[person_id] = Identity(
                person_id=person_id, name=record["name"], vectors=vectors,
                created=record.get("created", time.time()),
                last_seen=record.get("last_seen", 0.0),
                times_seen=record.get("times_seen", 0),
            )
        with self._lock:
            self.identities = loaded

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            people = list(self.identities.values())
            for identity in people:
                np.save(self.path / f"{identity.person_id}.npy", identity.vectors)
            records = [{
                "person_id": p.person_id, "name": p.name, "created": p.created,
                "last_seen": p.last_seen, "times_seen": p.times_seen,
                "vectors": int(len(p.vectors)),
            } for p in people]
        (self.path / INDEX_FILE).write_text(json.dumps(records, indent=2) + "\n")

    # -------------------------------------------------------------- lookup

    def __len__(self) -> int:
        return len(self.identities)

    def names(self) -> list[str]:
        return sorted(p.name for p in self.identities.values())

    def by_name(self, name: str) -> Identity | None:
        wanted = name.strip().casefold()
        for identity in self.identities.values():
            if identity.name.casefold() == wanted:
                return identity
        return None

    def match(self, embedding: np.ndarray) -> Match:
        """Nearest identity, with the runner-up so the caller can judge the gap."""
        embedding = normalise(embedding)[0]
        with self._lock:
            scored = sorted(((p.similarity(embedding), p) for p in self.identities.values()),
                            key=lambda pair: pair[0], reverse=True)
        if not scored:
            return Match(None)
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else -1.0
        return Match(best, best_score, runner_up)

    # --------------------------------------------------------------- write

    def enrol(self, name: str, embeddings: np.ndarray) -> Identity:
        """
        Introduce someone, or add more views of someone already known.

        Re-enrolling an existing name merges rather than replaces, so you can
        top up a person in different light without starting them over.
        """
        vectors = _prune(normalise(embeddings))
        existing = self.by_name(name)
        if existing is not None:
            with self._lock:
                existing.vectors = _prune(np.vstack([existing.vectors, vectors]))
            self.save()
            return existing

        with self._lock:
            person_id = self._next_id()
            identity = Identity(person_id=person_id, name=name.strip(), vectors=vectors)
            self.identities[person_id] = identity
        self.save()
        return identity

    def reinforce(self, identity: Identity, embedding: np.ndarray, score: float) -> bool:
        """
        Let a confident sighting teach the gallery, if it has anything to teach.

        Returns whether the vector was kept, so callers can log it; a False
        here is the normal case and not a failure.
        """
        if score < LEARN_THRESHOLD:
            return False
        vector = normalise(embedding)
        with self._lock:
            if float(np.max(identity.vectors @ vector[0])) > LEARN_NOVELTY:
                return False
            identity.vectors = _prune(np.vstack([identity.vectors, vector]))
        self.save()
        return True

    def seen(self, identity: Identity) -> None:
        """Note that this person turned up. The brain will want this later."""
        with self._lock:
            identity.last_seen = time.time()
            identity.times_seen += 1

    def forget(self, name: str) -> bool:
        """Delete a person completely - index entry and vectors."""
        identity = self.by_name(name)
        if identity is None:
            return False
        with self._lock:
            self.identities.pop(identity.person_id, None)
        (self.path / f"{identity.person_id}.npy").unlink(missing_ok=True)
        self.save()
        return True

    def _next_id(self) -> str:
        used = set()
        for person_id in self.identities:
            _, _, digits = person_id.rpartition("_")
            if digits.isdigit():
                used.add(int(digits))
        n = next(i for i in range(1, len(used) + 2) if i not in used)
        return f"person_{n:03d}"


def _prune(vectors: np.ndarray, limit: int = MAX_VECTORS) -> np.ndarray:
    """
    Keep the most distinct `limit` vectors.

    Drops whichever vector is most nearly a duplicate of another, repeatedly,
    so what survives is a spread of poses and lighting rather than the most
    recent minute. Quadratic, on at most a few dozen rows.
    """
    vectors = normalise(vectors)
    while len(vectors) > limit:
        sims = vectors @ vectors.T
        np.fill_diagonal(sims, -1.0)
        vectors = np.delete(vectors, int(np.argmax(sims.max(axis=1))), axis=0)
    return vectors
