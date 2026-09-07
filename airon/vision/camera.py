"""
aiRon's eyes: the Luxonis OAK-D Lite AF over USB 3, via DepthAI v3.

The colour sensor gives the frame we detect faces in; the stereo pair gives
real distance in metres. Depth is aligned onto the colour sensor and scaled to
match it, so a face box found in the RGB image indexes straight into the depth
image with no coordinate mapping.

Both DepthAI APIs are supported, but they are not equivalent on this hardware:
depthai 3.10 cannot get frames out of the OAK-D Lite's OV7251 mono sensors, so
depth is only offered on v2. See requirements.txt for the evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Depth costs frame rate, not bus speed. Measured on this OAK-D Lite: colour 20
# fps plus aligned depth 10 fps ran for three minutes without a wobble on a USB
# 2 link, while colour 30 plus mono 30 crashed the device inside a minute even
# at SuperSpeed. So the rates below are the default whenever depth is on, on
# any link, and --fps / --depth-fps override them.
#
# That first measurement noted the chip "plateaued near 50 C", and that part has
# not held up. Soaked for five and a half minutes at these exact rates it went
# 61 C to 69 C and was still climbing when the test ended - no plateau at all.
# Nothing dropped a frame in that window, so this is not a known failure point,
# but it is the reason temperature is now reported rather than assumed: a
# session that had been running far longer than any test did go blind
# repeatedly, and there was no thermal record to look at afterwards.
#
# (An earlier version gated depth on a SuperSpeed link. That was wrong: it came
# from depth failing under depthai v3, where the mono sensors never delivered a
# frame on any bus, which looked like a bandwidth or power ceiling and was not.)
DEPTH_COLOR_FPS = 20
DEPTH_MONO_FPS = 10


#: Face detector run on the camera's own NN cores. MobileNet-SSD based, 300x300
#: input, one class. Far more tolerant of head pose than a Haar cascade, and it
#: costs the Jetson nothing - the MyriadX was otherwise idle.
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
SHAVES = 6
FACE_BLOB = f"face-detection-retail-0004_openvino_2022.1_{SHAVES}shave.blob"
FACE_LABEL = 1

#: Second stage, for face recognition (spec section 12). Landmarks locate five
#: points inside a detected face; reid turns an aligned crop of it into a
#: 256-float signature. Both are Intel zoo models designed to follow
#: face-detection-retail-0004, and both run on the camera beside it.
#:
#: The MyriadX has 16 shaves and the detector already holds 6. Four each here
#: leaves two spare, and costs nothing that matters: the reid network needs
#: about 40 ms a sample at this width, against a sampling cadence of four a
#: second. Compiling either for 6 would buy latency aiRon has no use for, at
#: the price of having nowhere to put the next network.
LANDMARK_SHAVES = REID_SHAVES = 4
LANDMARK_BLOB = f"landmarks-regression-retail-0009_openvino_2022.1_{LANDMARK_SHAVES}shave.blob"
REID_BLOB = f"face-reidentification-retail-0095_openvino_2022.1_{REID_SHAVES}shave.blob"
LANDMARK_SIZE = 48
REID_SIZE = 128

#: Detections below this confidence never become tracklets.
CONFIDENCE = 0.5

#: Spatial coordinates are the median depth inside a shrunken detection box;
#: half the box avoids sampling the background around a head.
BOX_SCALE = 0.5
DEPTH_MIN_MM, DEPTH_MAX_MM = 200, 7000


@dataclass
class Track:
    """One tracked face, as the camera itself reports it."""

    id: int                      # persistent across frames, assigned on device
    status: str                  # NEW | TRACKED | LOST | REMOVED
    bbox: tuple[int, int, int, int]          # in colour-frame pixels
    distance_m: float | None = None          # from spatial coordinates
    spatial_mm: tuple[float, float, float] | None = None


@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray | None = None          # uint16, millimetres, aligned to colour
    tracks: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


class OakCamera:
    """
    Colour, and optionally aligned depth, from the OAK-D Lite.

    Speaks DepthAI v2 (ColorCamera + MonoCamera + XLinkOut + Device(pipeline))
    and v3 (Camera.build + requestOutput + pipeline.start). v2 is preferred and
    pinned, because it is the only one of the two that can drive this device's
    stereo pair.
    """

    #: Colour frames that must arrive before a pipeline is considered alive.
    VERIFY_COLOR_FRAMES = 5
    VERIFY_TIMEOUT_S = 6.0

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 want_depth: bool = True, force_depth: bool = False,
                 depth_fps: int | None = None, stream_depth: bool = False):
        import depthai as dai

        self.dai = dai
        self.v3 = not hasattr(dai.node, "XLinkOut")
        self.width, self.height = width, height
        self.name = "OAK"
        self.usb_speed = "unknown"
        self.stream_depth = stream_depth
        self.has_detector = False
        self.has_recognizer = False
        self._track_q = None
        self._lm_in = self._lm_out = None
        self._reid_in = self._reid_out = None

        want_depth = self._gate_depth(want_depth, force_depth)
        # Running the stereo pair is what costs headroom, so back the colour
        # sensor off too rather than letting the device fall over mid-session.
        self.fps = min(fps, DEPTH_COLOR_FPS) if want_depth else fps
        self.depth_fps = depth_fps or DEPTH_MONO_FPS

        self._build(want_depth)
        if want_depth and not self._verify():
            self.close()
            raise RuntimeError(
                "the OAK accepted the pipeline but streamed nothing. Re-run with "
                "--no-depth to drop stereo, which is the usual culprit."
            )

    def _gate_depth(self, want_depth: bool, force_depth: bool) -> bool:
        """Refuse depth where it is known not to work. Link speed is checked later,
        after boot, because it is not observable before it."""
        if not want_depth:
            return False
        if self.v3 and not force_depth:
            print(f"[vision] depthai {self.dai.__version__} cannot read this device's mono "
                  "sensors, so depth is unavailable. Install 'depthai<3' for range data.")
            return False
        return True

    # ---------------------------------------------------------------- build

    def _build(self, want_depth: bool) -> None:
        self.has_depth = False
        self._depth_q = None
        self._last_depth = None
        (self._build_v3 if self.v3 else self._build_v2)(want_depth)
        self._read_device_info()

    def _build_v2(self, want_depth: bool) -> None:
        """
        Everything the camera can do for itself, it does.

        Face detection and tracking run on the MyriadX, and with depth wired in
        the detector reports spatial coordinates directly - so aiRon gets a
        distance per face without the depth map ever crossing USB. The Jetson
        receives a colour frame and a short list of tracklets.
        """
        dai = self.dai
        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setPreviewSize(self.width, self.height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(float(self.fps))
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("color")
        cam.preview.link(xout.input)

        blob = MODEL_DIR / FACE_BLOB
        self.has_detector = blob.exists()
        if not self.has_detector:
            print(f"[vision] {FACE_BLOB} missing - run tools/fetch_models.py. "
                  "Falling back to Haar face detection on the CPU.")

        stereo = None
        if want_depth:
            left = pipeline.create(dai.node.MonoCamera)
            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
            left.setFps(float(self.depth_fps))
            right = pipeline.create(dai.node.MonoCamera)
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
            right.setFps(float(self.depth_fps))

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
            stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(self.width, self.height)
            left.out.link(stereo.left)
            right.out.link(stereo.right)
            self.has_depth = True

            # The depth map itself is only needed for the tuning bench. With the
            # detector on board, distance arrives inside the tracklets instead,
            # which is a few hundred bytes a frame rather than a few hundred KB.
            if self.stream_depth:
                xdepth = pipeline.create(dai.node.XLinkOut)
                xdepth.setStreamName("depth")
                stereo.depth.link(xdepth.input)

        if self.has_detector:
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setResize(300, 300)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            manip.setMaxOutputFrameSize(300 * 300 * 3)
            cam.preview.link(manip.inputImage)

            if stereo is not None:
                detector = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
                detector.setBoundingBoxScaleFactor(BOX_SCALE)
                detector.setDepthLowerThreshold(DEPTH_MIN_MM)
                detector.setDepthUpperThreshold(DEPTH_MAX_MM)
                stereo.depth.link(detector.inputDepth)
            else:
                detector = pipeline.create(dai.node.MobileNetDetectionNetwork)
            detector.setBlobPath(str(blob))
            detector.setConfidenceThreshold(CONFIDENCE)
            detector.input.setBlocking(False)
            manip.out.link(detector.input)

            tracker = pipeline.create(dai.node.ObjectTracker)
            tracker.setDetectionLabelsToTrack([FACE_LABEL])
            # SHORT_TERM_IMAGELESS predicts a track forward through frames where
            # detection blinks out, which ZERO_TERM types cannot - "zero term"
            # means no temporal reasoning at all. Measured over 35 s with a face
            # in view: IMAGELESS 696 packets and a single acquisition, colour
            # histogram 695 packets but two, KCF only 149 packets - it holds the
            # track but collapses the frame rate on this device.
            tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS)
            # UNIQUE_ID over SMALLEST_ID: a recycled id would silently make a
            # new person look like the previous one, which matters the moment
            # face recognition starts attaching names to these tracks.
            tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
            detector.passthrough.link(tracker.inputTrackerFrame)
            detector.passthrough.link(tracker.inputDetectionFrame)
            detector.out.link(tracker.inputDetections)

            xtracks = pipeline.create(dai.node.XLinkOut)
            xtracks.setStreamName("tracklets")
            tracker.out.link(xtracks.input)
            self._build_recognizer(pipeline)

        self.device = dai.Device(pipeline)
        self._color_q = self.device.getOutputQueue("color", maxSize=4, blocking=False)
        if want_depth and self.stream_depth:
            self._depth_q = self.device.getOutputQueue("depth", maxSize=4, blocking=False)
        if self.has_detector:
            self._track_q = self.device.getOutputQueue("tracklets", maxSize=4, blocking=False)
        if self.has_recognizer:
            self._lm_in = self.device.getInputQueue("landmark_in")
            self._lm_out = self.device.getOutputQueue("landmark_out", maxSize=4, blocking=False)
            self._reid_in = self.device.getInputQueue("reid_in")
            self._reid_out = self.device.getOutputQueue("reid_out", maxSize=4, blocking=False)

    def _build_recognizer(self, pipeline) -> None:
        """
        Wire up the recognition stages, if their blobs are installed.

        Unlike the detector, these are fed from the host rather than from the
        colour node: the Jetson has to crop and warp between the two networks,
        and doing that on the device would mean a Script node reimplementing an
        affine warp in MicroPython. The frames involved are 7 KB and 49 KB, and
        only a few go each way per second, so the round trip is cheap.
        """
        dai = self.dai
        blobs = {"landmark": MODEL_DIR / LANDMARK_BLOB, "reid": MODEL_DIR / REID_BLOB}
        missing = [b.name for b in blobs.values() if not b.exists()]
        if missing:
            print(f"[vision] no face recognition: {', '.join(missing)} missing - "
                  "run tools/fetch_models.py")
            return

        for name, size in (("landmark", LANDMARK_SIZE), ("reid", REID_SIZE)):
            xin = pipeline.create(dai.node.XLinkIn)
            xin.setStreamName(f"{name}_in")
            xin.setMaxDataSize(size * size * 3)
            xin.setNumFrames(4)

            nn = pipeline.create(dai.node.NeuralNetwork)
            nn.setBlobPath(str(blobs[name]))
            nn.input.setBlocking(False)
            nn.input.setQueueSize(2)
            xin.out.link(nn.input)

            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName(f"{name}_out")
            nn.out.link(xout.input)

        self.has_recognizer = True

    # ------------------------------------------------- recognition stages

    def _send(self, queue, image: np.ndarray, size: int) -> bool:
        """Hand one square BGR crop to a host-fed network, planar as it wants."""
        if queue is None:
            return False
        if image.shape[0] != size or image.shape[1] != size:
            import cv2
            image = cv2.resize(image, (size, size))
        frame = self.dai.ImgFrame()
        frame.setData(np.ascontiguousarray(image.transpose(2, 0, 1)).flatten())
        frame.setType(self.dai.ImgFrame.Type.BGR888p)
        frame.setWidth(size)
        frame.setHeight(size)
        try:
            queue.send(frame)
        except Exception:
            return False
        return True

    def send_face_crop(self, image: np.ndarray) -> bool:
        """Queue a face crop for landmark regression."""
        return self._send(self._lm_in, image, LANDMARK_SIZE)

    def poll_landmarks(self) -> np.ndarray | None:
        """Five (x, y) points as fractions of the crop, or None if not ready."""
        if self._lm_out is None:
            return None
        try:
            packet = self._lm_out.tryGet()
        except Exception:
            return None
        if packet is None:
            return None
        return np.array(packet.getFirstLayerFp16(), dtype=np.float32).reshape(5, 2)

    def send_aligned_face(self, image: np.ndarray) -> bool:
        """Queue an aligned 128x128 face for embedding."""
        return self._send(self._reid_in, image, REID_SIZE)

    def poll_embedding(self) -> np.ndarray | None:
        """A unit-length 256-float face signature, or None if not ready."""
        if self._reid_out is None:
            return None
        try:
            packet = self._reid_out.tryGet()
        except Exception:
            return None
        if packet is None:
            return None
        vector = np.array(packet.getFirstLayerFp16(), dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    def _build_v3(self, want_depth: bool) -> None:
        dai = self.dai
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_out = cam.requestOutput((self.width, self.height),
                                      dai.ImgFrame.Type.BGR888i, fps=float(self.fps))
        self._color_q = color_out.createOutputQueue(maxSize=4, blocking=False)

        if want_depth:   # only reachable via --force-depth; known not to stream
            stereo = self.pipeline.create(dai.node.StereoDepth).build(
                autoCreateCameras=True,
                presetMode=dai.node.StereoDepth.PresetMode.FACE,
                size=(self.width, self.height),
                fps=float(self.depth_fps),
            )
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            self._depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
            self.has_depth = True
        self.pipeline.start()

    def _read_device_info(self) -> None:
        device = self.pipeline.getDefaultDevice() if self.v3 else self.device
        try:
            self.name = device.getDeviceName()
        except Exception:
            pass
        try:
            self.usb_speed = str(device.getUsbSpeed()).split(".")[-1]
        except Exception:
            pass

    def chip_temperature(self) -> float | None:
        """Average die temperature in C, or None if the device cannot be asked.

        Cheap enough to call on a stall, which is the moment it matters: a
        camera that stopped delivering frames at 70 C and one that stopped at
        50 C are different faults, and after the fact they look identical.
        """
        try:
            device = self.pipeline.getDefaultDevice() if self.v3 else self.device
            return float(device.getChipTemperature().average)
        except Exception:
            return None

    def _verify(self) -> bool:
        """
        Confirm the device actually streams; a starved pipeline configures
        silently and reports nothing wrong.

        Only wait for streams that are actually expected. With the detector on
        board the depth map stays on the camera - distance arrives inside the
        tracklets - so there is no depth queue to see frames on.
        """
        want_depth_frames = self._depth_q is not None
        want_tracks = self._track_q is not None
        color_seen = depth_seen = track_seen = 0

        deadline = time.monotonic() + self.VERIFY_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if self._color_q.tryGet() is not None:
                    color_seen += 1
                if want_depth_frames and self._depth_q.tryGet() is not None:
                    depth_seen += 1
                if want_tracks and self._track_q.tryGet() is not None:
                    track_seen += 1
            except Exception:
                return False
            if (color_seen >= self.VERIFY_COLOR_FRAMES
                    and (depth_seen >= 1 or not want_depth_frames)
                    and (track_seen >= 1 or not want_tracks)):
                return True
            time.sleep(0.005)
        return False

    # ----------------------------------------------------------------- read

    def read(self) -> Frame | None:
        """Newest colour frame, plus whatever the camera has tracked. Non-blocking."""
        try:
            packet = self._color_q.tryGet()
        except Exception:
            return None
        if packet is None:
            return None

        if self._depth_q is not None:
            try:
                depth_packet = self._depth_q.tryGet()
                if depth_packet is not None:
                    self._last_depth = depth_packet.getFrame()
            except Exception:
                pass

        if self._track_q is not None:
            try:
                tracklets = self._track_q.tryGet()
                if tracklets is not None:
                    self._last_tracks = self._decode(tracklets)
            except Exception:
                pass

        return Frame(color=packet.getCvFrame(), depth=self._last_depth,
                     tracks=getattr(self, "_last_tracks", []))

    def _decode(self, packet) -> list:
        """Tracklet ROIs are normalised; turn them into colour-frame pixels."""
        tracks = []
        for t in packet.tracklets:
            roi = t.roi.denormalize(self.width, self.height)
            spatial = None
            distance = None
            coords = getattr(t, "spatialCoordinates", None)
            if coords is not None and coords.z > 0:
                spatial = (coords.x, coords.y, coords.z)
                distance = coords.z / 1000.0
            tracks.append(Track(
                id=int(t.id),
                status=str(t.status).rsplit(".", 1)[-1],
                bbox=(int(roi.x), int(roi.y), int(roi.width), int(roi.height)),
                distance_m=distance,
                spatial_mm=spatial,
            ))
        return tracks

    def close(self) -> None:
        try:
            self.pipeline.stop() if self.v3 else self.device.close()
        except Exception:
            pass


def sample_distance(depth: np.ndarray | None, bbox: tuple[int, int, int, int],
                    frame_shape: tuple[int, int]) -> float | None:
    """
    Distance in metres to whatever fills `bbox`.

    The bbox comes from the colour frame, which may be a different resolution
    to the depth map, so it is normalised before sampling. The median of the
    central patch is used: stereo returns 0 where it cannot match, and a face
    box always includes background at its corners.
    """
    if depth is None or depth.size == 0:
        return None

    fh, fw = frame_shape[:2]
    dh, dw = depth.shape[:2]
    x, y, w, h = bbox
    # central half of the box - cheeks and forehead, not the edges
    cx0 = int((x + w * 0.25) / fw * dw)
    cx1 = int((x + w * 0.75) / fw * dw)
    cy0 = int((y + h * 0.25) / fh * dh)
    cy1 = int((y + h * 0.75) / fh * dh)
    patch = depth[max(cy0, 0):max(cy1, 1), max(cx0, 0):max(cx1, 1)]
    if patch.size == 0:
        return None

    valid = patch[patch > 0]
    if valid.size < 16:
        return None
    return float(np.median(valid)) / 1000.0
