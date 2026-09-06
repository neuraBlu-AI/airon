"""
What aiRon's face is doing, and why.

Spec section 9: the brain sends a high-level emotional state, never animation
frames, and the face keeps moving at 60 FPS regardless of how slow the brain
is. So FaceCommand is the whole brain-facing API, and FaceAnimator owns every
value that actually moves.

There is a second input the spec does not mention: MIRROR mode, where the
measured expression of the person in front of the camera drives the face
directly. It is how the very first prototype worked, it is the fastest way to
see the perception pipeline is alive, and it is genuinely charming, so it
survives as a mode rather than as the architecture.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..core import WorldState


@dataclass(frozen=True)
class Emotion:
    """A pose, not an animation. The animator eases toward these numbers."""

    eye_open: float = 1.0        # vertical scale of the eye
    eye_width: float = 1.0
    brow_lift: float = 0.0       # +up, -down, at the inner end
    brow_angle: float = 0.0      # + is angry-inward, - is sad-outward
    mouth_curve: float = 0.0     # +1 grin, -1 frown
    mouth_open: float = 0.0
    energy: float = 1.0          # multiplies idle motion


EMOTIONS: dict[str, Emotion] = {
    "idle":      Emotion(),
    "curious":   Emotion(eye_open=1.10, brow_lift=0.35, brow_angle=-0.10, mouth_curve=0.15),
    "happy":     Emotion(eye_open=0.85, brow_lift=0.20, mouth_curve=0.85, energy=1.2),
    "excited":   Emotion(eye_open=1.20, eye_width=1.05, brow_lift=0.45,
                         mouth_curve=1.0, mouth_open=0.35, energy=1.6),
    "confused":  Emotion(eye_open=0.95, brow_lift=0.30, brow_angle=0.25, mouth_curve=-0.15),
    "thinking":  Emotion(eye_open=0.80, brow_lift=0.15, brow_angle=0.15, energy=0.7),
    "listening": Emotion(eye_open=1.05, brow_lift=0.25, mouth_curve=0.10, energy=0.9),
    "speaking":  Emotion(eye_open=1.0, mouth_curve=0.3, mouth_open=0.5, energy=1.1),
    "sleepy":    Emotion(eye_open=0.35, brow_lift=-0.20, mouth_curve=-0.10, energy=0.4),
    "concerned": Emotion(eye_open=1.0, brow_lift=0.10, brow_angle=-0.30, mouth_curve=-0.45),
}


@dataclass
class FaceCommand:
    """The entire brain -> face API (spec section 9)."""

    emotion: str = "idle"
    attention_x: float = 0.0
    attention_y: float = 0.0
    speaking: bool = False
    intensity: float = 0.7

    @property
    def pose(self) -> Emotion:
        return EMOTIONS.get(self.emotion, EMOTIONS["idle"])


def approach(current: float, target: float, rate: float, dt: float) -> float:
    return current + (target - current) * (1.0 - math.exp(-rate * dt))


class FaceAnimator:
    """
    Holds every animated value. Ticked at display rate, reads the world state
    and the current command, and never blocks on either.
    """

    def __init__(self, mirror: bool = True):
        self.mirror = mirror
        self.command = FaceCommand()

        self.gaze_x = 0.0
        self.gaze_y = 0.0
        self.lid = 1.0
        self.eye_open = 1.0
        self.eye_width = 1.0
        self.brow_lift = 0.0
        self.brow_angle = 0.0
        self.mouth_curve = 0.0
        self.mouth_open = 0.0
        self.roll = 0.0
        self.presence = 0.0        # 0 nobody, 1 someone is here
        self.awake = 0.0

        self._t = 0.0
        self._blink_at = 3.0
        self._blink_phase = -1.0
        self._saccade_at = 2.0
        self._saccade = (0.0, 0.0)

    def tick(self, dt: float, world: WorldState) -> None:
        self._t += dt
        person = world.primary
        pose = self.command.pose
        gain = self.command.intensity

        # --- attention -------------------------------------------------
        if person is not None:
            tx, ty = person.attention_x, person.attention_y
        else:
            tx, ty = self.command.attention_x, self.command.attention_y

        self._idle_motion(dt, pose.energy, person is not None)
        self.gaze_x = approach(self.gaze_x, tx + self._saccade[0], 8.0, dt)
        self.gaze_y = approach(self.gaze_y, ty + self._saccade[1], 8.0, dt)
        self.presence = approach(self.presence, 1.0 if person else 0.0, 3.0, dt)

        # --- expression -------------------------------------------------
        if self.mirror and person is not None:
            # Copy the human. Measurements are already smoothed by the tracker.
            self.eye_open = approach(self.eye_open, 1.0, 8.0, dt)
            self.eye_width = approach(self.eye_width, 1.0, 8.0, dt)
            self.lid = approach(self.lid, person.eyes_open, 26.0, dt)
            self.mouth_curve = approach(self.mouth_curve, person.mouth_curve, 7.0, dt)
            self.mouth_open = approach(self.mouth_open, person.mouth_open, 10.0, dt)
            self.roll = approach(self.roll, person.head_roll, 7.0, dt)
            # Brows join in: raised with a grin, drawn outward-down with a
            # frown, which is what makes a copied frown read as concern.
            self.brow_lift = approach(self.brow_lift, person.mouth_curve * 0.3, 5.0, dt)
            self.brow_angle = approach(self.brow_angle, min(person.mouth_curve, 0.0) * 0.4,
                                       5.0, dt)
        else:
            self.eye_open = approach(self.eye_open, pose.eye_open, 6.0, dt)
            self.eye_width = approach(self.eye_width, pose.eye_width, 6.0, dt)
            self.lid = approach(self.lid, self._blink_value(), 26.0, dt)
            self.mouth_curve = approach(self.mouth_curve, pose.mouth_curve * gain, 6.0, dt)
            self.mouth_open = approach(self.mouth_open, self._speech_mouth(pose), 12.0, dt)
            self.roll = approach(self.roll, 0.0, 4.0, dt)
            self.brow_lift = approach(self.brow_lift, pose.brow_lift * gain, 6.0, dt)
            self.brow_angle = approach(self.brow_angle, pose.brow_angle * gain, 6.0, dt)

        self.awake = approach(self.awake, 1.0, 1.5, dt)

    # ------------------------------------------------------------- idle life

    def _idle_motion(self, dt: float, energy: float, has_person: bool) -> None:
        """Micro-saccades and blinks. A face that holds perfectly still is dead."""
        if self._t >= self._saccade_at:
            reach = 0.05 if has_person else 0.22
            self._saccade = (random.uniform(-reach, reach), random.uniform(-reach * 0.6, reach * 0.6))
            self._saccade_at = self._t + random.uniform(0.8, 3.0) / max(energy, 0.2)
        if self._t >= self._blink_at:
            self._blink_phase = 0.0
            self._blink_at = self._t + random.uniform(2.5, 6.5) / max(energy, 0.2)
        if self._blink_phase >= 0.0:
            self._blink_phase += dt

    def _blink_value(self) -> float:
        """A blink is ~140 ms: shut fast, open a little slower."""
        if self._blink_phase < 0.0:
            return 1.0
        if self._blink_phase > 0.14:
            self._blink_phase = -1.0
            return 1.0
        half = 0.07
        return abs(self._blink_phase - half) / half

    def _speech_mouth(self, pose: Emotion) -> float:
        """Stand-in visemes until the speech service exists (spec section 10)."""
        if not self.command.speaking:
            return pose.mouth_open
        wobble = 0.5 + 0.5 * math.sin(self._t * 17.0) * math.sin(self._t * 6.3)
        return 0.15 + 0.55 * wobble
