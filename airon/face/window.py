"""
face_service - the window aiRon lives in.

A QTimer drives repaints at the display rate. It reads the latest WorldState
and never waits for vision, so a slow or stalled perception loop shows as a
face that keeps blinking and breathing, not a frozen screen.
"""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from ..core import StateStore
from .expression import EMOTIONS, FaceAnimator
from .renderer import paint_face

TARGET_FPS = 60

# Number keys cycle emotions by hand until brain_service exists.
EMOTION_KEYS = list(EMOTIONS)


class FaceWindow(QWidget):
    def __init__(self, store: StateStore, animator: FaceAnimator, vision=None,
                 fullscreen: bool = True):
        super().__init__()
        self.store = store
        self.animator = animator
        self.vision = vision
        self.show_debug = False

        self.setWindowTitle("aiRon")
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setCursor(Qt.BlankCursor)
        self.resize(1280, 800)
        if fullscreen:
            self.showFullScreen()
        else:
            self.show()

        self._last = time.monotonic()
        self._fps = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / TARGET_FPS))

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(now - self._last, 0.1)
        self._last = now
        if dt > 0:
            self._fps = self._fps * 0.9 + (0.1 / dt)
        self.animator.tick(dt, self.store.get())
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        paint_face(painter, self.width(), self.height(), self.animator)
        if self.show_debug:
            self._paint_debug(painter)
        painter.end()

    def _paint_debug(self, painter: QPainter) -> None:
        state = self.store.get()
        person = state.primary
        lines = [
            f"face {self._fps:4.1f} fps   vision {getattr(self.vision, 'fps', 0.0):4.1f} fps",
            f"mode {'MIRROR' if self.animator.mirror else self.animator.command.emotion}",
        ]
        if person is None:
            lines.append("nobody in view")
        else:
            distance = f"{person.distance_m:.2f} m" if person.distance_m else "no depth"
            who = f"{person.id} = {person.name}" if person.name else person.id
            lines += [
                f"{who}  {person.position}  {distance}",
                f"attention {person.attention_x:+.2f},{person.attention_y:+.2f}"
                f"  looking={person.looking_at_airon}",
                f"eyes {person.eyes_open:.2f}  curve {person.mouth_curve:+.2f}"
                f"  mouth {person.mouth_open:.2f}",
            ]

        painter.setPen(QColor(120, 200, 255, 210))
        painter.setFont(QFont("monospace", 11))
        for i, line in enumerate(lines):
            painter.drawText(18, 28 + i * 20, line)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif key == Qt.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif key == Qt.Key_D:
            self.show_debug = not self.show_debug
        elif key == Qt.Key_M:
            self.animator.mirror = not self.animator.mirror
        elif key == Qt.Key_C and self.vision is not None:
            self.vision.tracker.recalibrate()
        elif key == Qt.Key_S:
            self.animator.command.speaking = not self.animator.command.speaking
        elif Qt.Key_1 <= key <= Qt.Key_9:
            index = key - Qt.Key_1
            if index < len(EMOTION_KEYS):
                self.animator.mirror = False
                self.animator.command.emotion = EMOTION_KEYS[index]
