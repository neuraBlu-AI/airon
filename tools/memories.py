#!/usr/bin/env python3
"""
What aiRon remembers, and how to make it forget.

    python tools/memories.py              everyone it knows
    python tools/memories.py André        one person, in full
    python tools/memories.py --forget André     delete their memories

Strength is importance decayed by time since anything last touched the memory.
Recalling a memory touches it, so what keeps proving useful stays strong and
what never comes up again fades below the floor and is dropped on the next
write. A memory listed at 0.09 is nearly gone.

To delete somebody completely - memories and face both - use
tools/enroll_face.py --forget, which does the two together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.memory import MemoryStore                       # noqa: E402
from airon.memory.service import LANGUAGES, describe_gap   # noqa: E402
from airon.memory.store import FORGET_BELOW                # noqa: E402


def show(person, now: float) -> None:
    language = LANGUAGES.get(person.language, person.language) or "unknown"
    print(f"{person.person_id}  {person.name or '(no name)'}")
    print(f"  met {person.visits}x, first "
          f"{time.strftime('%Y-%m-%d', time.localtime(person.first_met))}, "
          f"last {describe_gap(now - person.last_met)}, speaks {language}")
    if not person.memories:
        print("  (nothing remembered yet)")
        return
    for memory in sorted(person.memories, key=lambda m: m.strength(now), reverse=True):
        fading = "  <- fading" if memory.strength(now) < FORGET_BELOW * 2 else ""
        print(f"    {memory.strength(now):.2f}  [{memory.kind}] {memory.text}"
              f"  (recalled {memory.recalled}x){fading}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="memories")
    parser.add_argument("name", nargs="?", help="show one person")
    parser.add_argument("--forget", metavar="NAME", help="delete a person's memories")
    args = parser.parse_args(argv)

    store = MemoryStore()
    now = time.time()

    if args.forget:
        person = store.by_name(args.forget)
        if person is None:
            print(f"aiRon remembers nobody called {args.forget}.", file=sys.stderr)
            return 1
        count = len(person.memories)
        store.forget_person(person.person_id)
        print(f"Forgot {person.name}: {count} "
              f"{'memory' if count == 1 else 'memories'}, {person.visits} visits.")
        print("Their face is still enrolled - tools/enroll_face.py --forget "
              f"{person.name} removes that too.")
        return 0

    if not store.people:
        print("aiRon does not remember anybody yet.")
        return 0

    people = sorted(store.people.values(), key=lambda p: p.last_met, reverse=True)
    if args.name:
        person = store.by_name(args.name)
        if person is None:
            print(f"aiRon remembers nobody called {args.name}.", file=sys.stderr)
            return 1
        people = [person]

    for person in people:
        show(person, now)
        print()
    if store.world:
        print("about the world:")
        for memory in sorted(store.world, key=lambda m: m.strength(now), reverse=True):
            print(f"    {memory.strength(now):.2f}  [{memory.kind}] {memory.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
