"""
Test overlays: what the camera sees, and what was said.

None of this is aiRon. Section 9 asks that the screen behave as a face rather
than a computer UI, so every panel here is off unless it is asked for - by
flag at startup, or by a key while running - and nothing in here can change
how the robot behaves. They exist because some failures are only legible while
watching: whether the camera has the person at all, and whether a strange
answer came from a strange transcript.

Removing the lot is deleting this file, two flags in app.py, and the overlay
list in window.py. That is the whole coupling, and it is deliberate.

Panels are duck-typed rather than sharing a base class: an object with
`enabled`, `key` and `paint(painter, width, height)` is a panel. The window
loops over them and knows nothing else about them, so adding a third does not
touch the window at all.

The preview is deliberately cheap. Camera frames arrive at 20 fps and the face
repaints at 60, so the QImage is rebuilt once per *frame* rather than once per
paint; without that check three quarters of the conversions would be of a
picture that had not changed. The face's frame rate is not negotiable
(section 9), and an inspection panel that costs it is worse than no panel.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPen

from ..core import EventType

#: Panels draw in this blue unless they have a reason not to - the same blue
#: the state overlay already uses, so the debug furniture reads as one thing.
CHROME = QColor(120, 200, 255, 200)


class CameraPreview:
    """
    The camera's own picture, small, in the top-right corner.

    Shows the frame as the camera delivered it, not mirrored to match the face
    - the question this answers is "what does the camera have", and flipping it
    to be friendlier would make it answer a different one. The tracked face is
    outlined so it is obvious whether vision has found somebody or is merely
    pointed at them.
    """

    key = Qt.Key_P

    def __init__(self, vision, *, enabled: bool = False,
                 width_frac: float = 0.24, margin: int = 18):
        self.vision = vision
        self.enabled = enabled
        self.width_frac = width_frac
        self.margin = margin
        self._image: QImage | None = None
        self._stamp: float | None = None
        self._bbox: tuple[int, int, int, int] | None = None
        self._size = (0, 0)

    def _refresh(self) -> None:
        """Rebuild the QImage, but only when the camera has produced a new one."""
        latest = getattr(self.vision, "latest_frame", None)
        if latest is None:
            return
        frame, obs = latest
        if frame.timestamp == self._stamp:
            return
        self._stamp = frame.timestamp

        color = frame.color
        if color is None or getattr(color, "size", 0) == 0:
            return
        height, width = color.shape[:2]
        self._size = (width, height)

        # Format_BGR888 matches what the camera hands back, so there is no
        # channel swap to pay for. copy() because the QImage would otherwise
        # point into a numpy buffer this function is about to stop referencing.
        buffer = np.ascontiguousarray(color)
        image = QImage(buffer.data, width, height,
                       buffer.strides[0], QImage.Format_BGR888)
        self._image = image.copy()
        self._bbox = obs.bbox if (obs is not None and obs.found) else None

    def paint(self, painter, width: int, height: int) -> None:
        self._refresh()
        if self._image is None or not self._size[0]:
            return

        source_w, source_h = self._size
        box_w = int(width * self.width_frac)
        box_h = int(box_w * source_h / source_w)
        x = width - box_w - self.margin
        y = self.margin
        target = QRect(x, y, box_w, box_h)

        painter.drawImage(target, self._image)
        painter.setPen(QPen(CHROME, 2))
        painter.drawRect(target)

        if self._bbox is not None:
            bx, by, bw, bh = self._bbox
            scale = box_w / source_w
            painter.setPen(QPen(QColor(130, 255, 170, 230), 2))
            painter.drawRect(int(x + bx * scale), int(y + by * scale),
                             max(1, int(bw * scale)), max(1, int(bh * scale)))

        painter.setPen(CHROME)
        painter.setFont(QFont("monospace", 10))
        painter.drawText(x, y + box_h + 16,
                         f"camera {getattr(self.vision, 'fps', 0.0):4.1f} fps"
                         f"{'' if self._bbox is not None else '   no face'}")


class Transcript:
    """
    The conversation as text, newest at the bottom.

    Reading what was actually transcribed is usually the fastest way to explain
    a reply that made no sense - several times the model was blamed for an
    answer that was a perfectly reasonable response to a mangled sentence.
    """

    key = Qt.Key_T

    HEARD = QColor(150, 220, 255, 235)
    SAID = QColor(255, 220, 150, 235)

    def __init__(self, bus, *, enabled: bool = False, turns: int = 8,
                 margin: int = 18, width_frac: float = 0.55):
        self.enabled = enabled
        self.margin = margin
        self.width_frac = width_frac
        self._log: deque[tuple[str, str]] = deque(maxlen=turns)
        bus.subscribe(self._on_event)

    def _on_event(self, event) -> None:
        # Called on whichever thread published - the listener's, the brain's.
        # deque.append is atomic and paint only ever takes a snapshot, so there
        # is nothing here worth a lock.
        if event.type is EventType.HEARD:
            self._log.append(("you", event.payload.get("text", "")))
        elif event.type is EventType.SPEECH_STARTED:
            self._log.append(("aiRon", event.payload.get("text", "")))

    @staticmethod
    def _wrap(text: str, metrics: QFontMetrics, limit: int) -> list[str]:
        """Greedy word wrap. Wrapped rather than elided because the end of a
        sentence is exactly the part worth reading - a truncated transcript is
        the failure being looked for."""
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and metrics.horizontalAdvance(candidate) > limit:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [""]

    def paint(self, painter, width: int, height: int) -> None:
        entries = list(self._log)
        if not entries:
            return

        font = QFont("monospace", 11)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        step = metrics.height() + 2
        limit = int(width * self.width_frac)

        # Laid out from the bottom up, so the newest line sits in the same
        # place regardless of how much has been said.
        rendered: list[tuple[QColor, str]] = []
        for speaker, text in entries:
            colour = self.SAID if speaker == "aiRon" else self.HEARD
            for i, line in enumerate(self._wrap(text, metrics, limit)):
                prefix = f"{speaker:>5}  " if i == 0 else " " * 7
                rendered.append((colour, prefix + line))

        y = height - self.margin
        for colour, line in reversed(rendered):
            if y < self.margin:
                break
            painter.setPen(colour)
            painter.drawText(self.margin, y, line)
            y -= step


def build(bus, vision, *, preview: bool = False, transcript: bool = False) -> list:
    """Every panel, each knowing whether it starts visible.

    Always constructed, never conditionally: a panel that only exists when its
    flag was passed cannot be switched on later with its key, and finding that
    out mid-test is annoying in exactly the situation these are for.
    """
    return [
        CameraPreview(vision, enabled=preview),
        Transcript(bus, enabled=transcript),
    ]
