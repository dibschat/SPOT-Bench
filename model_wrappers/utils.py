import base64
import io
import os
import time

import numpy as np
from PIL import Image

VERBOSE = os.environ.get("SPOTBENCH_VERBOSE", "").strip().lower() in {"1", "true", "yes"}


def frames_to_base64_images(frames: np.ndarray, fmt: str = "PNG"):
    """Convert uint8 RGB frames (N, H, W, 3) into base64 data URLs."""
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "JPG": "image/jpeg"}[fmt.upper()]
    out = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame, "RGB").save(buf, format=fmt)
        out.append(f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}")
    return out


def verbose_log(msg: str):
    """Print wrapper diagnostics. Off unless SPOTBENCH_VERBOSE=1."""
    if VERBOSE:
        print(msg, flush=True)


class GenerationTimer:
    """GPU-synchronised timer for one generation.

    `start()` before generating, `finish()` once the response is decoded. The
    elapsed time is the latency SPOT-Bench reports: how long a user would have
    waited for that response.
    """

    def __init__(self, torch_module=None):
        self.torch = torch_module
        self.started_at = None
        self.elapsed_s = None

    def _sync(self):
        if self.torch is None:
            return
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.synchronize()
        except Exception:
            pass

    def start(self):
        self._sync()
        self.started_at = time.perf_counter()
        return self

    def finish(self) -> float | None:
        if self.started_at is None:
            return None
        self._sync()
        self.elapsed_s = time.perf_counter() - self.started_at
        return self.elapsed_s
