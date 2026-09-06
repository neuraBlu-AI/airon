"""
aiRon's eyes: the Luxonis OAK-D Lite AF over USB 3, via DepthAI v3.

The colour sensor gives the frame we detect faces in; the stereo pair gives
real distance in metres. Depth is aligned onto the colour sensor so a face
box found in the RGB image can be sampled directly in the depth image.
"""

from __future__ import annotations

import glob
import time
from dataclasses import dataclass, field

import numpy as np

#: Below this negotiated link speed the OAK-D Lite cannot run its stereo pair.
USB3_MBPS = 5000


def usb_link_speed_mbps() -> int | None:
    """
    Negotiated USB speed of the attached Movidius device, read from sysfs.

    Checked before the device is opened. The OAK-D Lite draws more current with
    three sensors active than a 500 mA USB 2.0 port provides, and it does not
    degrade - it browns out and crashes, killing the colour stream too, and
    depthai cannot rebuild a pipeline afterwards without terminating the
    process. So the only safe move is to know the link speed up front.
    """
    for device in glob.glob("/sys/bus/usb/devices/*/"):
        try:
            with open(device + "idVendor") as handle:
                if handle.read().strip() != "03e7":
                    continue
            with open(device + "speed") as handle:
                return int(float(handle.read().strip()))
        except (OSError, ValueError):
            continue
    return None


@dataclass
class Frame:
    color: np.ndarray
    depth: np.ndarray | None = None          # uint16, millimetres, aligned to colour
    timestamp: float = field(default_factory=time.monotonic)


class OakCamera:
    """
    DepthAI v3 pipeline. v2 is not supported: its API has no `Camera.build`
    and wires stereo differently, and requirements.txt pins depthai>=3.10.

    Depth is requested but never assumed. On an underpowered USB 2.0 port the
    OAK-D Lite browns out when the stereo pair spins up - it takes the whole
    device down, colour stream included - so the pipeline is verified after
    start and rebuilt without stereo if nothing arrives. aiRon then runs with
    no range data rather than not running at all.
    """

    #: Colour frames that must arrive before a pipeline is considered alive.
    VERIFY_COLOR_FRAMES = 5
    VERIFY_TIMEOUT_S = 6.0

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 want_depth: bool = True, force_depth: bool = False):
        import depthai as dai

        if hasattr(dai.node, "XLinkOut"):
            raise RuntimeError(
                f"depthai {dai.__version__} is the v2 API; aiRon needs v3 "
                "(pip install -U 'depthai>=3.10')"
            )

        self.dai = dai
        self.width, self.height = width, height
        self.fps = fps
        self.name = "OAK"
        self.usb_speed = "unknown"

        self.link_mbps = usb_link_speed_mbps()
        if want_depth and not force_depth and self.link_mbps and self.link_mbps < USB3_MBPS:
            print(f"[vision] OAK is on a {self.link_mbps} Mbps USB 2 link. Its stereo pair "
                  "browns the device out at that power budget, so depth stays off. Move it "
                  "to a USB 3 port for range data, or pass --force-depth to try anyway.")
            want_depth = False

        self._build(want_depth)
        if want_depth and not self._verify():
            self.close()
            raise RuntimeError(
                "the OAK accepted a stereo pipeline but streamed nothing - it has most "
                "likely browned out. Re-run with --no-depth, or give it a USB 3 port."
            )

    # ---------------------------------------------------------------- build

    def _build(self, want_depth: bool) -> None:
        dai = self.dai
        self.has_depth = False
        self._depth_q = None
        self._last_depth = None

        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_out = cam.requestOutput((self.width, self.height),
                                      dai.ImgFrame.Type.BGR888i, fps=float(self.fps))
        self._color_q = color_out.createOutputQueue(maxSize=4, blocking=False)

        if want_depth:
            try:
                stereo = self.pipeline.create(dai.node.StereoDepth).build(
                    autoCreateCameras=True,
                    presetMode=dai.node.StereoDepth.PresetMode.FACE,
                    size=(640, 400),
                    fps=float(min(self.fps, 15)),
                )
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                self._depth_q = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
                self.has_depth = True
            except Exception as exc:
                print(f"[vision] stereo unavailable: {exc}")

        self.pipeline.start()
        self._read_device_info()

    def _read_device_info(self) -> None:
        try:
            device = self.pipeline.getDefaultDevice()
            self.name = device.getDeviceName()
            self.usb_speed = str(device.getUsbSpeed()).split(".")[-1]
        except Exception:
            pass
        if self.usb_speed not in ("SUPER", "SUPER_PLUS", "unknown"):
            print(f"[vision] link is USB {self.usb_speed} - fine for colour, "
                  "not enough for the stereo pair")

    def _verify(self) -> bool:
        """Confirm the device actually streams; a crashed OAK accepts config silently."""
        import time as _t

        color_seen = depth_seen = 0
        deadline = _t.monotonic() + self.VERIFY_TIMEOUT_S
        while _t.monotonic() < deadline:
            try:
                if self._color_q.tryGet() is not None:
                    color_seen += 1
                if self._depth_q is not None and self._depth_q.tryGet() is not None:
                    depth_seen += 1
            except Exception:
                return False
            if color_seen >= self.VERIFY_COLOR_FRAMES and depth_seen >= 1:
                return True
            _t.sleep(0.005)
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
            self.pipeline.stop()
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
