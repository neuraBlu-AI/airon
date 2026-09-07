"""
Turning a tracked face into a 256-float signature.

Three things happen per sample, and only the middle one is this module's work:

    detection box  ->  landmarks-regression-retail-0009    5 points
    5 points       ->  similarity transform                <- here
    aligned 128px  ->  face-reidentification-retail-0095   256 floats

Both networks run on the camera's own NN cores, so the Jetson does a warp and
nothing else. Alignment is what makes the embeddings comparable at all: the
reid network was trained on faces warped so that the eyes, nose tip and mouth
corners land on fixed pixels, and handing it a raw detection box instead throws
away most of its accuracy.

Measured on this OAK-D Lite, 140 samples of one face over 40 s:

    every sample                       mean 0.62 cosine, 5th pct 0.13
    only samples passing the gate      mean 0.90 cosine, 5th pct 0.77

The rejected samples were faces clipped by the edge of the frame, where the
warp pulls in black nothing and the network dutifully embeds the black. A face
half out of shot scores no better against itself than a stranger does, so the
gate is not a refinement - without it recognition simply does not work.

Requests are pipelined rather than awaited: a crop goes out on one frame and
the answer is collected on a later one. The vision loop therefore never blocks
on the ~40 ms round trip, which is the same rule the rest of the perception
path follows.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

#: Where the five landmarks must end up in the 128x128 crop, as fractions of
#: it: right eye, left eye, nose tip, right mouth corner, left mouth corner.
#: These are the reference points OpenVINO's own demo uses for this exact pair
#: of models, so they are part of the reid network's contract, not a taste.
REFERENCE = np.float32([
    (0.31556875, 0.46157410),
    (0.68262291, 0.46157410),
    (0.50026249, 0.64050530),
    (0.34947187, 0.82469190),
    (0.65343127, 0.82469190),
])

#: Smallest region of the original frame we will accept warping up to 128 px.
#: Below this the network is being fed interpolation rather than face. It is a
#: resolution floor derived from the warp - not a tuned distance limit, since
#: every sample measured so far was of someone sitting close.
MIN_SOURCE_PX = 64

#: A request that never comes back would wedge the state machine, and the OAK
#: is known to reset itself out from under a running pipeline.
REQUEST_TIMEOUT_S = 1.0


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Least-squares similarity transform (rotation, uniform scale, translation).

    The Umeyama solution, computed directly. cv2.estimateAffinePartial2D looks
    like the obvious call and is the wrong one here: its RANSAC and LMEDS
    estimators need redundancy, and with exactly five correspondences LMEDS
    fits a degenerate two-point subset and returns a wildly rotated transform.
    Measured cost of that mistake: same-face similarity 0.69 instead of 0.80 on
    identical inputs, before the quality gate was in the picture at all.
    """
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_d, target_d = source - source_mean, target - target_mean

    u, singular, vt = np.linalg.svd((target_d.T @ source_d) / len(source))
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:      # keep it a rotation,
        correction[1, 1] = -1.0                       # never a reflection
    rotation = u @ correction @ vt
    variance = (source_d ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(singular) @ correction) / max(variance, 1e-9))

    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = target_mean - scale * rotation @ source_mean
    return matrix


class FaceRecognizer:
    """
    Samples a tracked face every so often and returns embeddings for it.

    `camera` is anything offering the four second-stage methods on OakCamera.
    update() is called every frame and returns an embedding on the few frames
    where one has just come back, None on all the others.
    """

    #: Sampling cadence. Recognition settles after a handful of agreeing
    #: samples, so a few per second reaches a name in about a second while
    #: leaving the camera's NN cores almost entirely idle.
    INTERVAL_S = 0.25

    def __init__(self, camera, interval_s: float = INTERVAL_S):
        self.camera = camera
        self.interval_s = interval_s

        self._pending: tuple[np.ndarray, tuple[int, int, int, int]] | None = None
        self._awaiting_embedding = False
        self._deadline = 0.0
        self._next_sample = 0.0

        # Counters for tools/identity_bench.py; nothing depends on them.
        self.sampled = 0
        self.rejected = 0
        self.last_reject = ""

    def reset(self) -> None:
        """Forget anything in flight - a new person, or a rebuilt camera."""
        self._pending = None
        self._awaiting_embedding = False
        self._next_sample = 0.0

    def update(self, frame: np.ndarray, bbox: tuple[int, int, int, int],
               now: float) -> np.ndarray | None:
        """One step of the pipeline. Never blocks."""
        if (self._pending is not None or self._awaiting_embedding) and now > self._deadline:
            self.last_reject = "timeout"
            self.reset()

        if self._awaiting_embedding:
            embedding = self.camera.poll_embedding()
            if embedding is None:
                return None
            self._awaiting_embedding = False
            self.sampled += 1
            return embedding

        if self._pending is not None:
            landmarks = self.camera.poll_landmarks()
            if landmarks is None:
                return None
            source, box = self._pending
            self._pending = None
            self._send_aligned(source, box, landmarks, now)
            return None

        if now >= self._next_sample:
            self._request(frame, bbox, now)
        return None

    # ------------------------------------------------------------- internals

    def _request(self, frame: np.ndarray, bbox, now: float) -> None:
        height, width = frame.shape[:2]
        x, y, w, h = (int(v) for v in bbox)
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, width - x), min(h, height - y)
        if w < 24 or h < 24:
            return
        crop = frame[y:y + h, x:x + w]
        if not self.camera.send_face_crop(crop):
            return
        # The frame is kept, not the crop: the warp reads from the full frame,
        # so alignment can reach the forehead and chin that the detection box
        # cuts off, at full resolution.
        self._pending = (frame, (x, y, w, h))
        self._deadline = now + REQUEST_TIMEOUT_S
        self._next_sample = now + self.interval_s

    def _send_aligned(self, frame: np.ndarray, box, landmarks: np.ndarray,
                      now: float) -> None:
        height, width = frame.shape[:2]
        x, y, w, h = box
        points = np.column_stack([x + landmarks[:, 0] * w,
                                  y + landmarks[:, 1] * h]).astype(np.float32)
        matrix = similarity_transform(points, REFERENCE * 128.0)

        reason = self._reject_reason(matrix, width, height)
        if reason:
            self.rejected += 1
            self.last_reject = reason
            return

        aligned = cv2.warpAffine(frame, matrix, (128, 128), flags=cv2.INTER_LINEAR)
        if self.camera.send_aligned_face(aligned):
            self._awaiting_embedding = True
            self._deadline = now + REQUEST_TIMEOUT_S

    @staticmethod
    def _reject_reason(matrix: np.ndarray, width: int, height: int) -> str:
        """
        Decide whether the 128x128 the network would see is all real pixels.

        Rather than guess from the detection box, invert the warp and ask where
        the four corners of the output actually come from. If any of them lies
        outside the frame, part of what the network embeds is padding.
        """
        corners = np.float32([[0, 0], [128, 0], [128, 128], [0, 128]])
        inverse = cv2.invertAffineTransform(matrix)
        source = np.c_[corners, np.ones(4, dtype=np.float32)] @ inverse.T

        if (source[:, 0] < 0).any() or (source[:, 0] > width).any() \
                or (source[:, 1] < 0).any() or (source[:, 1] > height).any():
            return "clipped"
        if float(np.hypot(*(source[1] - source[0]))) < MIN_SOURCE_PX:
            return "too small"
        return ""
