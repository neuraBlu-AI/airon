#!/usr/bin/env python3
"""
Is face recognition actually working, and is the threshold right?

    python tools/identity_bench.py           what the gallery can tell apart
    python tools/identity_bench.py --live    who the camera thinks you are

The static report is the one that matters for tuning. MATCH_THRESHOLD in
airon/identity/gallery.py was set from samples of a single person, which can
only ever show how alike one face is to itself. It says nothing about how
alike two different faces look to this network - and that is the number that
decides whether aiRon calls you by your brother's name. Enrol a second person,
run this, and move the threshold if the two distributions are not clearly
apart.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from airon.identity import FaceRecognizer, Gallery          # noqa: E402
from airon.identity.gallery import MATCH_MARGIN, MATCH_THRESHOLD   # noqa: E402
from airon.vision import OakCamera                          # noqa: E402


def describe(label: str, values: np.ndarray) -> None:
    if values.size == 0:
        print(f"  {label:22s} (none)")
        return
    p1, p5, p50, p95, p99 = np.percentile(values, [1, 5, 50, 95, 99])
    print(f"  {label:22s} n={values.size:6d}  min {values.min():+.3f}  "
          f"p5 {p5:+.3f}  median {p50:+.3f}  p95 {p95:+.3f}  max {values.max():+.3f}")


def report(gallery: Gallery) -> int:
    if not gallery.identities:
        print("aiRon does not know anybody yet - run tools/enroll_face.py NAME")
        return 1

    people = sorted(gallery.identities.values(), key=lambda p: p.name)
    print(f"{len(people)} known, {sum(len(p.vectors) for p in people)} vectors, "
          f"in {gallery.path}\n")

    same, different = [], []
    for i, person in enumerate(people):
        sims = person.vectors @ person.vectors.T
        iu = np.triu_indices(len(person.vectors), 1)
        own = sims[iu]
        same.append(own)
        print(f"{person.person_id}  {person.name}  ({len(person.vectors)} views)")
        describe("against itself", own)
        for other in people[i + 1:]:
            across = (person.vectors @ other.vectors.T).ravel()
            different.append(across)
            describe(f"vs {other.name}", across)
        print()

    same = np.concatenate(same) if same else np.array([])
    different = np.concatenate(different) if different else np.array([])
    print("overall")
    describe("same person", same)
    describe("different people", different)
    print(f"\nthreshold in use: {MATCH_THRESHOLD:.2f} (margin {MATCH_MARGIN:.2f})")

    if different.size == 0:
        print("Only one person is enrolled, so nothing here can tell you whether the\n"
              "threshold separates people. Enrol somebody else and run this again.")
        return 0

    # A threshold is only meaningful if it sits in the gap between the two.
    floor, ceiling = np.percentile(same, 1), different.max()
    misses = float((same < MATCH_THRESHOLD).mean())
    confusions = float((different >= MATCH_THRESHOLD).mean())
    print(f"at {MATCH_THRESHOLD:.2f}: {misses:.1%} of same-person pairs fall below it "
          f"(a miss), {confusions:.1%} of different-person pairs reach it (a mix-up)")
    if ceiling < floor:
        print(f"clean gap: anything in {ceiling:.2f}..{floor:.2f} separates these people. "
              f"Midpoint {(ceiling + floor) / 2:.2f}.")
    else:
        print(f"the distributions OVERLAP ({ceiling:.2f} vs {floor:.2f}). More enrolled "
              f"views per person is the first thing to try.")
    return 0


def live(gallery: Gallery) -> int:
    import cv2

    camera = OakCamera(want_depth=False, fps=30)
    if not camera.has_recognizer:
        print("this camera has no recognition stages - run tools/fetch_models.py",
              file=sys.stderr)
        camera.close()
        return 1
    recognizer = FaceRecognizer(camera, interval_s=0.15)
    print("q to quit")

    label, colour, tracks = "...", (200, 200, 200), []
    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.002)
                continue
            if frame.tracks:
                tracks = [t for t in frame.tracks if t.status in ("NEW", "TRACKED")]

            view = frame.color.copy()
            if tracks:
                track = max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])
                embedding = recognizer.update(frame.color, track.bbox, time.monotonic())
                if embedding is not None:
                    match = gallery.match(embedding)
                    if match.identity is None:
                        label, colour = "gallery is empty", (120, 120, 240)
                    elif match.accepted:
                        label = f"{match.identity.name}  {match.score:.2f}"
                        colour = (120, 230, 140)
                    else:
                        label = (f"unknown  (best {match.identity.name} {match.score:.2f}, "
                                 f"runner-up {match.runner_up:.2f})")
                        colour = (120, 180, 240)
                x, y, w, h = track.bbox
                cv2.rectangle(view, (x, y), (x + w, y + h), colour, 2)
            else:
                label, colour = "no face", (200, 200, 200)

            note = f"{label}   [{recognizer.sampled} sampled, {recognizer.rejected} rejected"
            note += f": {recognizer.last_reject}]" if recognizer.last_reject else "]"
            cv2.putText(view, note, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        colour, 1, cv2.LINE_AA)
            cv2.imshow("aiRon identity bench", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="identity_bench")
    parser.add_argument("--live", action="store_true",
                        help="show who the camera thinks you are, right now")
    args = parser.parse_args(argv)
    gallery = Gallery()
    return live(gallery) if args.live else report(gallery)


if __name__ == "__main__":
    raise SystemExit(main())
