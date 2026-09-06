"""
The contract between perception and everything downstream.

Spec CLAUDE.md section 8: the vision subsystem must not hand raw frames to the
brain. It maintains a structured WorldState and emits discrete events, so the
brain reasons over meaning rather than pixels, and the face can react at 60 FPS
without waiting on anything slow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    """Discrete things worth telling the brain about (spec section 8)."""

    PERSON_ENTERED = "PERSON_ENTERED"
    PERSON_LEFT = "PERSON_LEFT"
    KNOWN_PERSON_DETECTED = "KNOWN_PERSON_DETECTED"
    UNKNOWN_PERSON_DETECTED = "UNKNOWN_PERSON_DETECTED"
    PERSON_LOOKING_AT_AIRON = "PERSON_LOOKING_AT_AIRON"
    PERSON_FELL = "PERSON_FELL"
    OBJECT_DETECTED = "OBJECT_DETECTED"
    GESTURE_DETECTED = "GESTURE_DETECTED"


@dataclass
class Event:
    type: EventType
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        detail = " ".join(f"{k}={v}" for k, v in self.payload.items())
        return f"{self.type.value} {detail}".strip()


@dataclass
class Person:
    """
    One person aiRon can currently see.

    Everything here is derived on the Jetson from the OAK-D stream. The
    expression channels are measurements, not judgements - turning them into
    an emotion is the brain's job, not the vision service's.
    """

    id: str
    name: str | None = None
    recognized: bool = False

    distance_m: float | None = None      # real metres, from the stereo pair
    position: str = "center"             # left | center | right
    looking_at_airon: bool = False

    # Normalised attention coordinates, -1..+1, ready for the face renderer.
    attention_x: float = 0.0
    attention_y: float = 0.0

    # Measured expression channels, 0..1 unless noted.
    eyes_open: float = 1.0
    mouth_curve: float = 0.0             # -1 frown .. 0 neutral .. +1 grin
    mouth_open: float = 0.0
    head_roll: float = 0.0               # radians

    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # in full camera pixels
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def smile(self) -> float:
        """Positive half of the mouth curve - "is this person smiling", 0..1."""
        return max(self.mouth_curve, 0.0)

    def to_dict(self) -> dict:
        """The shape spec section 8 asks the brain to consume."""
        return {
            "id": self.id,
            "name": self.name,
            "recognized": self.recognized,
            "distance": round(self.distance_m, 2) if self.distance_m else None,
            "position": self.position,
            "looking_at_airon": self.looking_at_airon,
        }


@dataclass
class WorldState:
    """What aiRon believes is in front of it right now."""

    people: list[Person] = field(default_factory=list)
    objects: list[dict] = field(default_factory=list)
    environment: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def primary(self) -> Person | None:
        """Whoever currently has aiRon's attention: the nearest face."""
        if not self.people:
            return None
        with_depth = [p for p in self.people if p.distance_m is not None]
        if with_depth:
            return min(with_depth, key=lambda p: p.distance_m)
        return max(self.people, key=lambda p: p.bbox[2] * p.bbox[3])

    def to_dict(self) -> dict:
        return {
            "people": [p.to_dict() for p in self.people],
            "objects": self.objects,
            "environment": self.environment,
            "timestamp": self.timestamp,
        }
