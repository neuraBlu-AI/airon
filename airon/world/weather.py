"""
Tomorrow's forecast, and the only thing aiRon reaches the network for
besides the model.

AIRON-8 deliberately kept tools off the network. This one is the exception,
and it is worth saying why it is allowed rather than pretending the rule was
never there. The rule exists because a tool that blocks on a socket blocks a
conversation, and because a model that can call out can be talked into
calling out. Both are handled here rather than waived:

- It never runs inside a turn. Actions execute after aiRon has finished
  speaking, so a slow or dead network costs a spinning icon, not a pause in
  the middle of a sentence.
- The model chooses a place name and nothing else. One host, one path, and
  a name that is resolved by the same service before it is used; asking
  about New York is the point, asking about anything that is not a place
  fails to geocode and aiRon says it does not know where that is.
- What is spoken comes from the response, not from the model. aiRon reads
  the fetched numbers out of a template; it is never asked to say what the
  weather will be, because a robot that invents tomorrow's forecast is worse
  than one that says it does not know.

Open-Meteo needs no API key and no account, which is the reason for choosing
it over the better-known services: nothing about this feature requires a
credential to exist, so nothing about it can leak one.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from ..core.log import log

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

#: Where the robot is. Either is enough; coordinates win if both are set.
PLACE_ENV = "AIRON_WEATHER_PLACE"
LATLON_ENV = "AIRON_WEATHER_LATLON"

#: Short, because this runs on the thread that talks. A forecast that takes
#: longer than this to arrive is not worth the robot standing still for.
TIMEOUT_S = 4.0

#: Tomorrow's weather does not change every minute, and asking twice in a
#: conversation should not be two round trips.
CACHE_S = 30 * 60.0

#: WMO weather codes, collapsed to the handful of conditions worth drawing a
#: different screen for. The full table has 28 entries and distinguishes
#: "light freezing drizzle" from "dense freezing drizzle", which is not a
#: distinction a face can usefully show.
CONDITIONS = {
    "clear":   ({0, 1}, {"en": "clear", "de": "klar"}),
    "cloudy":  ({2, 3, 45, 48}, {"en": "cloudy", "de": "bewölkt"}),
    "rain":    ({51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82},
                {"en": "rainy", "de": "regnerisch"}),
    "snow":    ({71, 73, 75, 77, 85, 86}, {"en": "snowy", "de": "verschneit"}),
    "storm":   ({95, 96, 99}, {"en": "stormy", "de": "stürmisch"}),
}

#: Two shapes, because naming the place aiRon is standing in sounds like a
#: weather report rather than an answer. "Morgen wird es regnerisch" is what
#: a person in the room would say; "Morgen wird es in New York regnerisch" is
#: what they would say about somewhere else.
SENTENCE = {
    "en": "Tomorrow looks {condition}, between {low} and {high} degrees.",
    "de": "Morgen wird es {condition}, zwischen {low} und {high} Grad.",
}
SENTENCE_ELSEWHERE = {
    "en": "Tomorrow in {place} looks {condition}, between {low} and {high} degrees.",
    "de": "Morgen wird es in {place} {condition}, zwischen {low} und {high} Grad.",
}

#: Said when a place cannot be found at all.
NOWHERE = {
    "en": "I could not find a place called {place}.",
    "de": "Einen Ort namens {place} finde ich nicht.",
}


@dataclass(frozen=True)
class Forecast:
    """Tomorrow, in the only detail a robot with a face can use."""

    condition: str          # a key of CONDITIONS
    high: int
    low: int
    place: str
    fetched: float
    #: Somewhere other than where the robot is standing. Changes how it is
    #: said and whether the face names the place.
    elsewhere: bool = False

    def sentence(self, lang: str) -> str:
        lang = lang if lang in SENTENCE else "en"
        words = CONDITIONS[self.condition][1]
        shape = SENTENCE_ELSEWHERE[lang] if self.elsewhere else SENTENCE[lang]
        return shape.format(condition=words.get(lang, words["en"]),
                            low=self.low, high=self.high, place=self.place)


def _condition(code: int) -> str:
    for name, (codes, _) in CONDITIONS.items():
        if code in codes:
            return name
    return "cloudy"


class Weather:
    """
    Where tomorrow's forecast lives, for whoever wants to draw or say it.

    One instance, shared by the toolbox that fetches and the overlay that
    paints. `latest` is written on the talker thread and read on the Qt
    thread; it is replaced rather than mutated, so a reader sees either the
    old forecast or the new one and never half of either.
    """

    def __init__(self, *, place: str | None = None, latlon: str | None = None):
        self.place = place if place is not None else os.environ.get(PLACE_ENV, "").strip()
        self.latlon = latlon if latlon is not None else os.environ.get(LATLON_ENV, "").strip()
        #: The one the face is showing: whatever was asked for most recently,
        #: here or elsewhere.
        self.latest: Forecast | None = None
        #: Per place, so "und in New York?" straight after "wie wird morgen?"
        #: is one round trip rather than two, and asking about home again
        #: does not throw away what was just fetched.
        self._cache: dict[str, Forecast] = {}
        self.last_error = ""
        #: When somebody last asked. Distinct from Forecast.fetched, which is
        #: when the network was last touched: asking twice inside the cache
        #: window is still asking, and the face should show it again.
        self.requested = 0.0
        self._lock = threading.Lock()
        self._coords: tuple[float, float, str] | None = None
        self._places: dict[str, tuple[float, float, str] | None] = {}

    def configured(self) -> bool:
        return bool(self.place or self.latlon)

    def _get(self, url: str, params: dict) -> dict:
        query = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{url}?{query}", timeout=TIMEOUT_S) as response:
            return json.loads(response.read().decode("utf-8"))

    def geocode(self, place: str) -> tuple[float, float, str] | None:
        """Turn a name somebody said out loud into coordinates.

        Cached by what was asked rather than by what came back, because the
        cache exists to save a round trip for a repeated question and the
        question is what repeats. "New York" and "new york" are the same
        question; "New York" and "New York City" are two, which costs one
        extra lookup and is not worth being clever about.
        """
        key = place.strip().lower()
        if key in self._places:
            return self._places[key]
        found = self._get(GEOCODE_URL, {"name": place, "count": 1})
        results = found.get("results") or []
        if not results:
            self._places[key] = None
            return None
        first = results[0]
        here = (first["latitude"], first["longitude"], first.get("name", place))
        self._places[key] = here
        return here

    def coordinates(self) -> tuple[float, float, str] | None:
        """Where the robot itself is. Resolved once and kept: it does not move
        between rooms yet, and when it does this is the line that will need
        to know."""
        if self._coords is not None:
            return self._coords
        if self.latlon:
            try:
                lat, lon = (float(part) for part in self.latlon.split(",", 1))
            except ValueError:
                self.last_error = f"{LATLON_ENV}={self.latlon!r} is not 'lat,lon'"
                return None
            self._coords = (lat, lon, self.place or f"{lat:.2f},{lon:.2f}")
            return self._coords
        if not self.place:
            self.last_error = f"nowhere to ask about - set {PLACE_ENV} in .env"
            return None
        self._coords = self.geocode(self.place)
        if self._coords is None:
            self.last_error = f"no such place as {self.place!r}"
        return self._coords

    def tomorrow(self, place: str = "", *, force: bool = False) -> Forecast | None:
        """The forecast, for here or for somewhere named, from the cache when
        it is fresh enough.

        Returns None and sets last_error on any failure - no network, no
        configuration, a place nobody can find, a shape from the API nobody
        expected. Every caller treats that as "aiRon does not know", which is
        a thing it is allowed to be.

        `tomorrow` means tomorrow where the weather is, not where the robot
        is: the request asks for local days, so asking about New York in the
        evening gets New York's tomorrow rather than a slice of tonight.
        """
        self.requested = time.monotonic()
        asked = place.strip()
        key = asked.lower() or "@home"

        with self._lock:
            cached = self._cache.get(key)
        if cached and not force and time.monotonic() - cached.fetched < CACHE_S:
            with self._lock:
                self.latest = cached
            return cached

        try:
            if asked:
                coords = self.geocode(asked)
                if coords is None:
                    self.last_error = f"no such place as {asked!r}"
                    return None
            else:
                coords = self.coordinates()
                if coords is None:
                    return None
            lat, lon, place = coords
            data = self._get(FORECAST_URL, {
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "forecast_days": 2,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            })
            daily = data["daily"]
            # Index 1 is tomorrow: index 0 is today, and the question asked is
            # always about tomorrow.
            forecast = Forecast(
                condition=_condition(int(daily["weather_code"][1])),
                high=round(float(daily["temperature_2m_max"][1])),
                low=round(float(daily["temperature_2m_min"][1])),
                place=place,
                fetched=time.monotonic(),
                elsewhere=bool(asked),
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            log(f"[weather] could not fetch: {self.last_error}")
            return None

        with self._lock:
            self.latest = forecast
        return forecast
