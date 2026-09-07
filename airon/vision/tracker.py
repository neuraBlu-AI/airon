"""
Expression measurement on the Jetson CPU.

Finding the face is the camera's job now - a MobileNet-SSD detector and an
ObjectTracker run on the MyriadX and hand over a box with a stable id. This
module reads what the face inside that box is doing: eyes open or shut, and the
curve and opening of the mouth. Haar cascades are still used for the eyes,
which is cheap and confined to a known face region.

If no detector blob is installed, update() falls back to finding the face here
too, so aiRon still works - just less robustly to head pose.

This module only measures. It does not decide what an expression means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def approach(current: float, target: float, rate: float, dt: float) -> float:
    """Exponential glide - frame-rate independent smoothing."""
    return current + (target - current) * (1.0 - math.exp(-rate * dt))


#: How far the mouth must depart from its resting curvature to read as a full
#: expression. Frowns move the mouth line less than grins do, so they get a
#: smaller span - otherwise a real frown registers as a tenth of one.
SPAN_SMILE = 0.09
SPAN_FROWN = 0.06

#: While the mouth is this far from neutral something is being expressed, and
#: the baseline must stop following it - otherwise holding any expression
#: quietly redefines "neutral" and the expression fades while you hold it.
NEUTRAL_DEADBAND = 0.035


@dataclass
class FaceObservation:
    """One frame's worth of measurement. All channels 0..1 unless noted."""

    found: bool = False
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # full-resolution pixels
    attention_x: float = 0.0     # -1 left .. +1 right
    attention_y: float = 0.0     # -1 up   .. +1 down
    head_roll: float = 0.0       # radians
    eyes_open: float = 1.0
    mouth_curve: float = 0.0     # -1 frown .. 0 neutral .. +1 grin
    mouth_open: float = 0.0

    # Debug only - consumed by tools/vision_bench.py, ignored by the face.
    debug: dict = field(default_factory=dict)


class FaceTracker:
    DETECT_W = 320   # detect on a downscaled grey frame; plenty for a face

    def __init__(self) -> None:
        base = Path(cv2.data.haarcascades)
        self.face_cc = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        self.eye_cc = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))
        self.smile_cc = cv2.CascadeClassifier(str(base / "haarcascade_smile.xml"))
        if self.face_cc.empty():
            raise RuntimeError("OpenCV Haar cascades missing from this install")

        self.obs = FaceObservation()
        self._blink_frames = 0

        # Nobody's resting mouth is flat and everyone's differs, so track the
        # resting curvature and read expression as a departure from it. aiRon
        # calibrates itself to whoever sits down in front of it.
        self.curv = 0.0
        self.curv_base = 0.0
        self.open_raw = 0.0
        self._calibrating = 0

    def recalibrate(self) -> None:
        self._calibrating = 0
        self.curv_base = self.curv

    def update(self, frame: np.ndarray, dt: float,
               bbox: tuple[int, int, int, int] | None = None) -> FaceObservation:
        """
        Measure the face in `bbox` (full-resolution pixels, from the camera's
        own tracker). With no bbox, fall back to detecting one here.
        """
        h, w = frame.shape[:2]
        scale = self.DETECT_W / float(w)
        small = cv2.resize(frame, (self.DETECT_W, int(h * scale)))
        grey = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        grey_full = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        obs = self.obs
        obs.debug = {}

        if bbox is None:
            faces = self.face_cc.detectMultiScale(grey, 1.15, 5, minSize=(48, 48))
            if not len(faces):
                obs.found = False
                obs.eyes_open = approach(obs.eyes_open, 1.0, 10.0, dt)
                obs.mouth_curve = approach(obs.mouth_curve, 0.0, 6.0, dt)
                obs.mouth_open = approach(obs.mouth_open, 0.0, 6.0, dt)
                return obs
            # Biggest face wins - whoever is closest is who aiRon is talking to.
            box_small = max(faces, key=lambda f: f[2] * f[3])
            obs.bbox = tuple((np.array(box_small) / scale).astype(int))
        else:
            obs.bbox = tuple(int(v) for v in bbox)
            box_small = (np.array(bbox) * scale).astype(int)

        fx, fy, fw, fh = (int(v) for v in box_small)
        fx, fy = max(fx, 0), max(fy, 0)
        fw = max(min(fw, grey.shape[1] - fx), 1)
        fh = max(min(fh, grey.shape[0] - fy), 1)
        obs.found = True
        obs.debug["face"] = obs.bbox

        gx = (fx + fw / 2.0) / grey.shape[1]
        gy = (fy + fh / 2.0) / grey.shape[0]
        obs.attention_x = approach(obs.attention_x, float(np.clip((gx - 0.5) * 2.4, -1, 1)), 9.0, dt)
        obs.attention_y = approach(obs.attention_y, float(np.clip((gy - 0.5) * 2.4, -1, 1)), 9.0, dt)

        roi = grey[fy:fy + fh, fx:fx + fw]
        roll, lids = self._read_eyes(roi, (fx, fy), scale)
        obs.head_roll = approach(obs.head_roll, roll, 7.0, dt)
        obs.eyes_open = approach(obs.eyes_open, lids, 26.0, dt)   # blinks must be snappy

        curve, mouth = self._read_mouth(grey_full, (fx, fy, fw, fh), scale, dt)
        obs.mouth_curve = approach(obs.mouth_curve, curve, 6.0, dt)
        obs.mouth_open = approach(obs.mouth_open, mouth, 10.0, dt)
        return obs

    def _read_eyes(self, roi, origin, scale):
        """Eye pair gives head roll; losing it for a moment is a blink."""
        upper = roi[: int(roi.shape[0] * 0.60)]
        eyes = self.eye_cc.detectMultiScale(upper, 1.12, 6, minSize=(14, 14))
        ox, oy = origin
        self.obs.debug["eyes"] = [
            tuple((np.array([ox + ex, oy + ey, ew, eh]) / scale).astype(int))
            for ex, ey, ew, eh in eyes[:4]
        ]

        if len(eyes) >= 2:
            self._blink_frames = 0
            pair = sorted(sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2],
                          key=lambda e: e[0])
            (ax, ay, aw, ah), (bx, by, bw, bh) = pair
            roll = math.atan2((by + bh / 2) - (ay + ah / 2), (bx + bw / 2) - (ax + aw / 2))
            return float(np.clip(roll, -0.45, 0.45)), 1.0

        # Wait a beat so one dropped detection does not read as a blink.
        self._blink_frames += 1
        return self.obs.head_roll, 0.05 if self._blink_frames >= 2 else 1.0

    def _read_mouth(self, grey_full, face_box, scale, dt):
        """
        Mouth geometry from the pixels, at full camera resolution.

        The mouth is the darkest horizontal band in the lower face. Track its
        centre of darkness per column and fit a parabola: corners lifted above
        the centre is a smile, corners dropped below it is a frown, and the
        vertical spread of that dark mass is how far open it is. The Haar smile
        cascade only fires on a wide grin and returns a bare yes/no, so it is
        kept as a confirming vote on the positive side, not a driver.

        Returns a signed curve, -1 frown .. +1 grin.
        """
        fx, fy, fw, fh = (np.array(face_box) / scale).astype(int)
        H, W = grey_full.shape[:2]
        y0, y1 = max(fy + int(fh * 0.58), 0), min(fy + int(fh * 0.97), H)
        x0, x1 = max(fx + int(fw * 0.20), 0), min(fx + int(fw * 0.80), W)
        band = grey_full[y0:y1, x0:x1]
        if band.shape[0] < 10 or band.shape[1] < 16:
            return 0.0, 0.0

        band = cv2.GaussianBlur(band, (5, 5), 0)
        dark = 255.0 - band.astype(np.float32)
        dark -= dark.min()
        dark[dark < np.percentile(dark, 66)] = 0.0    # keep the darkest third

        mass = dark.sum(axis=0)
        if mass.max() <= 0:
            return 0.0, 0.0
        good = mass > mass.max() * 0.18
        if good.sum() < 10:
            return 0.0, 0.0

        rows = np.arange(band.shape[0], dtype=np.float32)[:, None]
        line = (dark * rows).sum(axis=0) / np.maximum(mass, 1e-6)

        xs = np.linspace(-1.0, 1.0, band.shape[1], dtype=np.float32)[good]
        ys = line[good] / band.shape[0]
        try:
            a = float(np.polyfit(xs, ys, 2)[0])
        except (np.linalg.LinAlgError, ValueError):
            return self.obs.mouth_curve, self.obs.mouth_open

        # Rows count downward, so corners-up is a NEGATIVE quadratic term.
        self.curv = float(np.clip(-a, -0.6, 0.6))

        # Learn the resting mouth only while the face is actually resting. An
        # earlier version relaxed the baseline quickly downward, which chased
        # and erased every frown within a couple of seconds.
        self._calibrating += 1
        deviation = self.curv - self.curv_base
        if self._calibrating < 20:
            self.curv_base = self.curv
        else:
            rate = 0.25 if abs(deviation) < NEUTRAL_DEADBAND else 0.01
            self.curv_base = approach(self.curv_base, self.curv, rate, dt)

        span = SPAN_SMILE if deviation >= 0 else SPAN_FROWN
        curve = float(np.clip(deviation / span, -1.0, 1.0))

        spread = (dark * (rows - line[None, :]) ** 2).sum() / max(dark.sum(), 1e-6)
        self.open_raw = float(math.sqrt(max(spread, 0.0)) / band.shape[0])
        mouth = float(np.clip((self.open_raw - 0.16) / 0.13, 0.0, 1.0))

        # The cascade only knows grins, so it may push the curve up, never down.
        lower = band[int(band.shape[0] * 0.1):]
        if lower.size and len(self.smile_cc.detectMultiScale(lower, 1.6, 18, minSize=(24, 14))):
            curve = min(1.0, max(curve, 0.0) + 0.35)

        cols = np.arange(band.shape[1])[good]
        self.obs.debug["mouth_curve"] = np.stack(
            [x0 + cols, y0 + line[good]], axis=1).astype(np.int32)
        self.obs.debug["curv"] = (self.curv, self.curv_base, self.open_raw)
        return curve, mouth
