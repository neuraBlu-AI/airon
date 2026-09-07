"""
Plumbing between services: a pub/sub event bus and a latest-value state store.

Deliberately tiny. Spec section 20 warns against premature complexity, and
Phase 1 is one process with a couple of threads - this is enough to keep the
module boundaries honest, and can be swapped for ROS 2 topics in Phase 2
without touching either side.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from .events import Event, EventType, WorldState


class EventBus:
    """Synchronous fan-out. Handlers must be quick; they run on the publisher's thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._subscribers.append(handler)

    def publish(self, event_type: EventType, **payload) -> Event:
        event = Event(event_type, payload)
        with self._lock:
            handlers = list(self._subscribers)
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:                      # one bad subscriber
                print(f"[bus] handler {handler!r} raised: {exc}")   # must not
        return event                                      # stop the others


class StateStore:
    """
    The latest WorldState, readable from any thread.

    The face reads this at 60 FPS while vision writes at camera rate; neither
    waits for the other. That separation is spec section 9's core requirement.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = WorldState()

    def set(self, state: WorldState) -> None:
        with self._lock:
            self._state = state

    def get(self) -> WorldState:
        with self._lock:
            return self._state
