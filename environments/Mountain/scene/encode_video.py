"""Streaming mp4 encoder: frames are written to ffmpeg's stdin one at a time as they're produced,
rather than collected into a list first -- matches run_pilot.py's flat-memory discipline (see its
module docstring), unlike Simulations_0_0_1/pipeline/encode_video.py's write_video(frames, ...)
which takes the whole list at once.

Shells out to ffmpeg rather than using opencv's VideoWriter: opencv-python-headless still pulls in
an X11-linked cv2 native module (missing libxcb.so.1) inside the Isaac Sim venv, while ffmpeg is
already present on this cluster and has no such issue (same finding as Simulations_0_0_1).
"""

import shutil
import subprocess
from pathlib import Path

_FALLBACK_FFMPEG = "/mnt/weka/shared-cache/miniforge3/bin/ffmpeg"

DEFAULT_FPS = 4.0  # one action step per frame here, not real-time footage -- 4fps is watchable
                    # without being so slow a 300-step trajectory takes minutes to review


def _find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    if Path(_FALLBACK_FFMPEG).exists():
        return _FALLBACK_FFMPEG
    raise RuntimeError("ffmpeg not found on PATH or at fallback location")


class VideoWriter:
    """Open once per trajectory, call write_frame() per step, close() when done.

    Usage:
        vw = VideoWriter(out_path, width, height, fps=4.0)
        for frame in frames:
            vw.write_frame(frame)   # (H, W, 3) uint8
        vw.close()
    """

    def __init__(self, out_path, width, height, fps=DEFAULT_FPS):
        ffmpeg = _find_ffmpeg()
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_path),
        ]
        self.out_path = Path(out_path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._closed = False
        self._wrote_any = False

    def write_frame(self, frame):
        if frame.shape[:2] == (0, 0):
            return
        self.proc.stdin.write(frame.tobytes())
        self._wrote_any = True

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.proc.stdin.close()
        err = self.proc.stderr.read()
        ret = self.proc.wait()
        if not self._wrote_any:
            # nothing was written -- ffmpeg produced no/garbage output, remove it rather than
            # leave a misleading empty file behind
            self.out_path.unlink(missing_ok=True)
            return
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed (exit {ret}): {err.decode(errors='replace')[-2000:]}")
