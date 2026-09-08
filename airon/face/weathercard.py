"""
Tomorrow's weather, on aiRon's face.

Unlike the panels in overlay.py, this one is not test furniture: it is part
of what the robot shows a person, so it comes up on its own when somebody
asks about the weather and goes away again shortly after. It is here rather
than in that file for exactly that reason - overlay.py says of itself that
nothing in it can change how the robot behaves, and this changes what the
robot looks like.

It follows the same duck-typed contract the window already loops over
(`enabled`, `key`, `paint`), because that mechanism was the right shape
already and a second one would earn nothing.

The face keeps its own colours: the wash is thin, and the card sits at the
top rather than over the eyes. A screen that turns into a weather app is not
a face any more (section 9), and the point of showing a forecast on a robot
is that the robot is telling you, not that the robot has become a widget.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont

#: How long the card stays up after somebody asks, and how long it takes to
#: fade once it is done. Long enough to read twice; short enough that a face
#: is a face again by the time the conversation has moved on.
SHOW_S = 25.0
FADE_S = 1.5

#: What each kind of day looks like. Chosen to read at a glance from across a
#: room rather than to be meteorologically precise - the words carry the
#: detail, the colour carries the mood.
LOOKS = {
    "clear":  (QColor(255, 196, 76), "☀"),
    "cloudy": (QColor(150, 168, 190), "☁"),
    "rain":   (QColor(90, 150, 220), "☂"),
    "snow":   (QColor(210, 230, 245), "❄"),
    "storm":  (QColor(170, 130, 230), "⚡"),
}

WORDS = {
    "clear":  {"en": "clear", "de": "klar"},
    "cloudy": {"en": "cloudy", "de": "bewölkt"},
    "rain":   {"en": "rain", "de": "Regen"},
    "snow":   {"en": "snow", "de": "Schnee"},
    "storm":  {"en": "storms", "de": "Gewitter"},
}

TOMORROW = {"en": "Tomorrow", "de": "Morgen"}


class WeatherCard:
    """Reads the shared Weather object; owns nothing but how it looks."""

    key = None                      # not a debug panel, so no key toggles it

    def __init__(self, weather, *, lang: str = "en"):
        self.weather = weather
        self.lang = lang if lang in TOMORROW else "en"

    @property
    def enabled(self) -> bool:
        """Up while it is worth showing. The window checks this every frame,
        so appearing and leaving needs no timer of its own."""
        return (self.weather is not None
                and self.weather.latest is not None
                and time.monotonic() - self.weather.requested < SHOW_S + FADE_S)

    def _alpha(self) -> float:
        left = SHOW_S + FADE_S - (time.monotonic() - self.weather.requested)
        return max(0.0, min(1.0, left / FADE_S))

    def paint(self, painter, width: int, height: int) -> None:
        forecast = self.weather.latest
        if forecast is None:
            return
        colour, glyph = LOOKS.get(forecast.condition, LOOKS["cloudy"])
        fade = self._alpha()

        # A thin wash of the day's colour over the whole screen, and a band
        # across the top holding the words. The wash is what makes a rainy
        # tomorrow feel different from a clear one from the far side of a
        # room, where the text is unreadable anyway.
        wash = QColor(colour)
        wash.setAlphaF(0.10 * fade)
        painter.fillRect(0, 0, width, height, wash)

        band = max(56, int(height * 0.11))
        backing = QColor(12, 16, 24)
        backing.setAlphaF(0.72 * fade)
        painter.fillRect(0, 0, width, band, backing)

        stripe = QColor(colour)
        stripe.setAlphaF(0.95 * fade)
        painter.fillRect(0, band - 3, width, 3, stripe)

        painter.setPen(stripe)
        painter.setFont(QFont("sans-serif", max(20, band // 3)))
        # The place is named only when it is not this one: a card that says
        # the name of the room you are standing in is noise.
        where = f" {forecast.place}" if forecast.elsewhere else ""
        painter.drawText(QRectF(0, 0, width, band), Qt.AlignCenter,
                         f"{glyph}   {TOMORROW[self.lang]}{where}: "
                         f"{WORDS[forecast.condition][self.lang]}   "
                         f"{forecast.low}° – {forecast.high}°")
