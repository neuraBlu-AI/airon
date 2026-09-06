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

import numpy as np

# Depth costs frame rate, not bus speed. Measured on this OAK-D Lite: colour 20
# fps plus aligned depth 10 fps ran for three minutes without a wobble on a USB
# 2 link (chip plateaued near 50 C), while colour 30 plus mono 30 crashed the
# device inside a minute even at SuperSpeed. So the rates below are the default
# whenever depth is on, on any link, and --fps / --depth-fps override them.
#
# (An earlier version gated depth on a SuperSpeed link. That was wrong: it came
# from depth failing under depthai v3, where the mono sensors never delivered a
# frame on any bus, which looked like a bandwidth or power ceiling and was not.)
DEPTH_COLOR_FPS = 20
DEPTH_MONO_FPS = 10


@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray | None = None          # uint16, millimetres, aligned to colour
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
                 depth_fps: int | None = None):
        import depthai as dai

        self.dai = dai
        self.v3 = not hasattr(dai.node, "XLinkOut")
        self.width, self.height = width, height
        self.name = "OAK"
        self.usb_speed = "unknown"

        want_depth = self._gate_depth(want_depth, force_depth)
        # Running the stereo pair is what costs headroom, so back the colour
        # sensor off too rather than letting the device fall over mid-session.
        self.fps = min(fps, DEPTH_COLOR_FPS) if want_depth else fps
        self.depth_fps = depth_fps or DEPTH_MONO_FPS

        self._build(want_depth)
        if want_depth and not self._verify():
            self.close()
            raise RuntimeError(
                "the OAK accepted a stereo pipeline but streamed no depth. "
                "Re-run with --no-depth to continue without range data."
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
            stereo.setLeftRightCheck(True)          # required for depth alignment
            # Align onto the colour sensor AND match its resolution: aligned depth
            # defaults to the full 1920x1080 sensor, ~4 MB a frame, which is a lot
            # of USB for data we only ever median over a face box.
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            stereo.setOutputSize(self.width, self.height)
            left.out.link(stereo.left)
            right.out.link(stereo.right)
            xdepth = pipeline.create(dai.node.XLinkOut)
            xdepth.setStreamName("depth")
            stereo.depth.link(xdepth.input)
            self.has_depth = True

        self.device = dai.Device(pipeline)
        self._color_q = self.device.getOutputQueue("color", maxSize=4, blocking=False)
        if want_depth:
            self._depth_q = self.device.getOutputQueue("depth", maxSize=4, blocking=False)

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

    def _verify(self) -> bool:
        """Confirm the device actually streams; a starved pipeline configures silently."""
        color_seen = depth_seen = 0
        deadline = time.monotonic() + self.VERIFY_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if self._color_q.tryGet() is not None:
                    color_seen += 1
                if self._depth_q is not None and self._depth_q.tryGet() is not None:
                    depth_seen += 1
            except Exception:
                return False
            if color_seen >= self.VERIFY_COLOR_FRAMES and depth_seen >= 1:
                return True
            time.sleep(0.005)
        return False

    # ----------------------------------------------------------------- read

    def read(self) -> Frame | None:
        """Newest colour frame, with the newest depth alongside it. Non-blocking."""
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
        return Frame(color=packet.getCvFrame(), depth=self._last_depth)

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
