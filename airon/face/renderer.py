"""
Drawing aiRon's face with QPainter.

Spec section 9: the display is the face, not a UI. So this paints the whole
surface - dark ground, two large eyes, brows, one mouth - with no chrome, no
panels and no text. Every dimension is derived from the smaller screen
dimension so the same code fills the 10.1" 1920x1200 panel and a small
development window identically.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QLinearGradient, QPainter,
                           QPainterPath, QPen, QRadialGradient)

GROUND_TOP = QColor(16, 21, 30)
GROUND_BOTTOM = QColor(6, 8, 13)
EYE = QColor(226, 243, 255)
IRIS = QColor(56, 168, 233)
PUPIL = QColor(8, 12, 18)
GLOW = QColor(56, 168, 233)
MOUTH = QColor(214, 235, 252)


def paint_face(painter: QPainter, w: float, h: float, a) -> None:
    painter.setRenderHint(QPainter.Antialiasing, True)

    ground = QLinearGradient(0, 0, 0, h)
    ground.setColorAt(0.0, GROUND_TOP)
    ground.setColorAt(1.0, GROUND_BOTTOM)
    painter.fillRect(QRectF(0, 0, w, h), QBrush(ground))

    unit = min(w, h)
    painter.save()
    painter.translate(w / 2.0, h / 2.0)
    # Copying head roll one-for-one looks like a glitch; a fraction reads as
    # sympathy. Same reason the gaze offsets below are damped.
    painter.rotate(-math.degrees(a.roll) * 0.55)

    eye_dx = unit * 0.245
    eye_y = -unit * 0.06
    lid = max(a.lid, 0.03)
    eye_w = unit * 0.165 * a.eye_width
    # Never let a blink erase the eye entirely - a closed eye is a drawn lid,
    # not an absence, or the face reads as switched off for those 140 ms.
    eye_h = max(unit * 0.215 * a.eye_open * lid, unit * 0.016)

    for side in (-1, 1):
        _eye(painter, side * eye_dx, eye_y, eye_w, eye_h, unit, a)
        _brow(painter, side, side * eye_dx, eye_y, eye_w, eye_h, unit, a)

    _mouth(painter, 0.0, unit * 0.235, unit, a)
    painter.restore()


def _eye(painter, x, y, ew, eh, unit, a) -> None:
    # Glow first, so the eye sits inside its own light.
    glow = QRadialGradient(QPointF(x, y), unit * 0.20)
    tint = QColor(GLOW)
    tint.setAlphaF(0.20 * a.presence + 0.05)
    glow.setColorAt(0.0, tint)
    glow.setColorAt(1.0, QColor(0, 0, 0, 0))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(glow))
    painter.drawEllipse(QPointF(x, y), unit * 0.20, unit * 0.20)

    body = QRectF(x - ew / 2.0, y - eh / 2.0, ew, eh)
    radius = min(ew, eh) / 2.0
    painter.setBrush(QBrush(EYE))
    painter.drawRoundedRect(body, radius, radius)

    # A shut eye is a line - no iris, no highlight, or it looks painted on.
    if eh < unit * 0.045:
        return

    look = QPointF(x + a.gaze_x * ew * 0.26, y + a.gaze_y * eh * 0.22)
    iris_r = min(ew, eh) * 0.34
    painter.setBrush(QBrush(IRIS))
    painter.drawEllipse(look, iris_r, iris_r)
    painter.setBrush(QBrush(PUPIL))
    painter.drawEllipse(look, iris_r * 0.55, iris_r * 0.55)

    spark = QColor(255, 255, 255, 235)
    painter.setBrush(QBrush(spark))
    painter.drawEllipse(QPointF(look.x() - iris_r * 0.34, look.y() - iris_r * 0.38),
                        iris_r * 0.22, iris_r * 0.22)


def _brow(painter, side, x, y, ew, eh, unit, a) -> None:
    if abs(a.brow_lift) < 0.02 and abs(a.brow_angle) < 0.02:
        return

    gap = unit * 0.055 + eh / 2.0
    lift = a.brow_lift * unit * 0.035
    tilt = a.brow_angle * unit * 0.045
    inner_x = x + side * -ew * 0.5      # toward the centre of the face
    outer_x = x + side * ew * 0.5

    pen = QPen(EYE, unit * 0.018, Qt.SolidLine, Qt.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    path = QPainterPath(QPointF(inner_x, y - gap - lift + tilt))
    path.quadTo(QPointF(x, y - gap - lift * 1.35),
                QPointF(outer_x, y - gap - lift * 0.5 - tilt * 0.4))
    painter.drawPath(path)
    painter.setPen(Qt.NoPen)


def _mouth(painter, x, y, unit, a) -> None:
    half = unit * 0.105 * (1.0 + 0.25 * max(a.mouth_curve, 0.0))
    thickness = unit * 0.020

    # A smile bows DOWN between its corners: the corners are the high points,
    # and y grows downward. Positive curve therefore pushes the control point
    # below the corner line, negative pulls it above into a frown.
    bow = a.mouth_curve * unit * 0.085

    if a.mouth_open > 0.06:
        # Two arcs sharing corners. The lower one is offset from the upper one
        # rather than from the corner line, so the opening never turns inside
        # out when a wide smile bows the top edge further down than the bottom.
        height = unit * 0.030 + a.mouth_open * unit * 0.115
        path = QPainterPath(QPointF(x - half, y))
        path.quadTo(QPointF(x, y + bow), QPointF(x + half, y))
        path.quadTo(QPointF(x, y + bow + height), QPointF(x - half, y))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(MOUTH))
        painter.drawPath(path)
        return

    painter.setPen(QPen(MOUTH, thickness, Qt.SolidLine, Qt.RoundCap))
    painter.setBrush(Qt.NoBrush)
    path = QPainterPath(QPointF(x - half, y))
    path.quadTo(QPointF(x, y + bow * 2.0), QPointF(x + half, y))
    painter.drawPath(path)
    painter.setPen(Qt.NoPen)
