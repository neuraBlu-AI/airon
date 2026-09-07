#!/usr/bin/env python3
"""
Introduce someone to aiRon.

    python tools/enroll_face.py Pierre        capture and remember a face
    python tools/enroll_face.py --list        who aiRon currently knows
    python tools/enroll_face.py --forget Pierre     delete a person entirely

Enrolment collects a spread of views rather than a burst of one, because a
gallery of twenty near-identical frames recognises you only in the pose you
happened to be sitting in. Samples that repeat what is already captured are
discarded, so turning your head slowly is what actually fills the bar.

Re-running for a name already known adds to that person instead of replacing
them - the way to make aiRon reliable in evening light is to enrol again in
the evening.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.identity import FaceRecognizer, Gallery     # noqa: E402
from airon.memory import MemoryStore                   # noqa: E402
from airon.vision import OakCamera                     # noqa: E402

#: A sample must differ this much from everything already captured to be kept.
#: Loose enough that a slow head turn keeps feeding it, tight enough that
#: sitting perfectly still stops it.
NOVELTY = 0.97

TARGET_SAMPLES = 24
TIMEOUT_S = 60.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="enroll_face", description=__doc__.split("\n")[1])
    parser.add_argument("name", nargs="?", help="who this is")
    parser.add_argument("--list", action="store_true", help="list known people")
    parser.add_argument("--forget", metavar="NAME", help="delete a person")
    parser.add_argument("--samples", type=int, default=TARGET_SAMPLES,
                        help=f"views to collect (default {TARGET_SAMPLES})")
    parser.add_argument("--no-preview", action="store_true",
                        help="skip the camera window (for an SSH session)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    gallery = Gallery()

    if args.list:
        if not gallery.identities:
            print("aiRon does not know anybody yet.")
            return 0
        for identity in sorted(gallery.identities.values(), key=lambda p: p.name):
            seen = (time.strftime("%Y-%m-%d %H:%M", time.localtime(identity.last_seen))
                    if identity.last_seen else "never")
            print(f"{identity.person_id}  {identity.name:<16} "
                  f"{len(identity.vectors):2d} views  seen {identity.times_seen}x  last {seen}")
        return 0

    if args.forget:
        # Face and memories together. "Delete Pierre" has to mean all of
        # Pierre, or the privacy story the gallery tells is not true.
        identity = gallery.by_name(args.forget)
        if identity is None:
            print(f"aiRon does not know anyone called {args.forget}.", file=sys.stderr)
            return 1
        gallery.forget(args.forget)
        memories = MemoryStore()
        person = memories.people.get(identity.person_id)
        remembered = len(person.memories) if person else 0
        memories.forget_person(identity.person_id)
        print(f"aiRon has forgotten {args.forget}: "
              f"{len(identity.vectors)} face views and {remembered} "
              f"{'memory' if remembered == 1 else 'memories'}.")
        return 0

    if not args.name:
        print("give a name to enrol, or --list / --forget", file=sys.stderr)
        return 2

    return capture(gallery, args)


def capture(gallery: Gallery, args) -> int:
    # Depth is irrelevant to enrolment and costs the colour sensor frame rate,
    # so this runs the camera lean - more frames means more distinct views.
    camera = OakCamera(want_depth=False, fps=30)
    if not camera.has_recognizer:
        print("this camera has no recognition stages - run tools/fetch_models.py",
              file=sys.stderr)
        camera.close()
        return 1

    recognizer = FaceRecognizer(camera, interval_s=0.10)
    preview = not args.no_preview
    collected: list[np.ndarray] = []
    tracks: list = []
    deadline = time.monotonic() + TIMEOUT_S

    print(f"Look at the camera. Turn your head slowly - left, right, up, down.\n"
          f"Collecting {args.samples} distinct views of {args.name}, "
          f"or giving up in {TIMEOUT_S:.0f}s.")

    import cv2
    try:
        while len(collected) < args.samples and time.monotonic() < deadline:
            frame = camera.read()
            if frame is None:
                time.sleep(0.002)
                continue
            if frame.tracks:
                tracks = [t for t in frame.tracks if t.status in ("NEW", "TRACKED")]

            now = time.monotonic()
            if tracks:
                track = max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])
                embedding = recognizer.update(frame.color, track.bbox, now)
                if embedding is not None and _is_new(embedding, collected):
                    collected.append(embedding)
                    print(f"  {len(collected):2d}/{args.samples}", flush=True)
            else:
                track = None

            if preview:
                preview = _show(cv2, frame, track, recognizer, collected, args)
                if preview == "quit":
                    print("\ncancelled - nothing saved.")
                    return 1
    except KeyboardInterrupt:
        print("\ncancelled - nothing saved.")
        return 1
    finally:
        camera.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    if len(collected) < 5:
        print(f"\nonly {len(collected)} usable views - too few to recognise anyone by.\n"
              f"The face has to be fully inside the frame, so sit back a little and "
              f"try again. (rejected {recognizer.rejected}: {recognizer.last_reject})",
              file=sys.stderr)
        return 1

    identity = gallery.enrol(args.name, np.stack(collected))
    print(f"\n{identity.name} is {identity.person_id}, "
          f"{len(identity.vectors)} views stored in {gallery.path}")
    return 0


def _is_new(embedding: np.ndarray, collected: list[np.ndarray]) -> bool:
    """Reject a view that says nothing the collected ones do not already say."""
    if not collected:
        return True
    return float(np.max(np.stack(collected) @ embedding)) < NOVELTY


def _show(cv2, frame, track, recognizer, collected, args):
    """Live preview with a progress bar. Returns False if it cannot draw."""
    view = frame.color.copy()
    height, width = view.shape[:2]
    if track is not None:
        x, y, w, h = track.bbox
        cv2.rectangle(view, (x, y), (x + w, y + h), (90, 220, 120), 2)

    done = len(collected) / max(args.samples, 1)
    cv2.rectangle(view, (12, height - 34), (width - 12, height - 14), (60, 60, 60), -1)
    cv2.rectangle(view, (12, height - 34),
                  (12 + int((width - 24) * done), height - 14), (120, 230, 140), -1)
    note = f"{args.name}: {len(collected)}/{args.samples}"
    if recognizer.last_reject and len(collected) < 2:
        note += f"   (skipping: {recognizer.last_reject})"
    cv2.putText(view, note, (14, height - 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (235, 235, 235), 1, cv2.LINE_AA)
    try:
        cv2.imshow("aiRon - enrolling", view)
    except cv2.error:
        print("[enrol] no display; continuing without a preview")
        return False
    return "quit" if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27) else True


if __name__ == "__main__":
    raise SystemExit(main())
