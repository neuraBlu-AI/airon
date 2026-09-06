#!/usr/bin/env python3
"""
Render aiRon's face to a PNG contact sheet without a display.

Lets the renderer be reviewed - and regressions caught - without standing in
front of the camera. Runs on the offscreen Qt platform, so it works over SSH.

    python tools/render_face_preview.py [out.png]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt                                   # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter       # noqa: E402
from PySide6.QtWidgets import QApplication                      # noqa: E402

from airon.core import Person, StateStore, WorldState           # noqa: E402
from airon.face import FaceAnimator, paint_face                 # noqa: E402

CELL_W, CELL_H = 420, 300
COLUMNS = 4

# (caption, mirror-mode person or None, emotion, extra animator overrides)
SCENES = [
    ("idle, nobody there", None, "idle", {}),
    ("curious", None, "curious", {}),
    ("happy", None, "happy", {}),
    ("excited", None, "excited", {}),
    ("thinking", None, "thinking", {}),
    ("sleepy", None, "sleepy", {}),
    ("concerned", None, "concerned", {}),
    ("speaking", None, "speaking", {"mouth_open": 0.55}),
    ("mirror: looking left", "left", None, {}),
    ("mirror: grinning", "grin", None, {}),
    ("mirror: slight smile", "half", None, {}),
    ("mirror: mouth open", "open", None, {}),
    ("mirror: mid-blink", "blink", None, {}),
    ("mirror: slight frown", "halffrown", None, {}),
    ("mirror: frowning", "frown", None, {}),
    ("mirror: frown + open", "frownopen", None, {}),
]

PEOPLE = {
    "left":      dict(attention_x=-0.75, attention_y=0.1, mouth_curve=0.1),
    "grin":      dict(attention_x=0.0, mouth_curve=0.95),
    "half":      dict(attention_x=0.0, mouth_curve=0.45),
    "open":      dict(attention_x=0.2, mouth_curve=0.4, mouth_open=0.85),
    "blink":     dict(attention_x=0.0, eyes_open=0.08, head_roll=0.22),
    "halffrown": dict(attention_x=0.0, mouth_curve=-0.45),
    "frown":     dict(attention_x=0.0, mouth_curve=-0.95),
    "frownopen": dict(attention_x=-0.2, mouth_curve=-0.8, mouth_open=0.5),
}


def build_animator(kind, emotion, overrides):
    animator = FaceAnimator(mirror=kind is not None)
    world = WorldState()
    if kind is not None:
        world.people = [Person(id="person_001", **PEOPLE[kind])]
    else:
        animator.command.emotion = emotion

    # Settle the easing so the still frame shows the pose, not the approach.
    for _ in range(240):
        animator.tick(1 / 60, world)
    animator._saccade = (0.0, 0.0)
    if kind is not None:
        person = world.people[0]
        animator.gaze_x, animator.gaze_y = person.attention_x, person.attention_y
        animator.lid = person.eyes_open
        animator.brow_lift = person.mouth_curve * 0.3
        animator.brow_angle = min(person.mouth_curve, 0.0) * 0.4
        animator.mouth_curve = person.mouth_curve
        animator.mouth_open = person.mouth_open
        animator.roll = person.head_roll
    else:
        animator.lid = 1.0
    for key, value in overrides.items():
        setattr(animator, key, value)
    return animator


def main() -> int:
    app = QApplication(sys.argv[:1])
    rows = (len(SCENES) + COLUMNS - 1) // COLUMNS
    sheet = QImage(CELL_W * COLUMNS, CELL_H * rows, QImage.Format_RGB32)
    sheet.fill(QColor(0, 0, 0))

    painter = QPainter(sheet)
    for index, (caption, kind, emotion, overrides) in enumerate(SCENES):
        col, row = index % COLUMNS, index // COLUMNS
        painter.save()
        painter.translate(col * CELL_W, row * CELL_H)
        painter.setClipRect(0, 0, CELL_W, CELL_H)
        paint_face(painter, CELL_W, CELL_H, build_animator(kind, emotion, overrides))
        painter.setPen(QColor(140, 190, 230))
        painter.setFont(QFont("monospace", 10))
        painter.drawText(12, CELL_H - 12, caption)
        painter.setPen(QColor(40, 50, 65))
        painter.drawRect(0, 0, CELL_W - 1, CELL_H - 1)
        painter.restore()
    painter.end()

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "face_preview.png")
    sheet.save(str(out))
    print(f"wrote {out} ({sheet.width()}x{sheet.height()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
