from __future__ import annotations

from hashlib import sha256
import io

import numpy as np
import torch
from PIL import Image

from .calibration_contract import DecodedFrame
from .pnlcalib_constants import INPUT_HEIGHT, INPUT_WIDTH


def decode_frame(frame_index: int, data: bytes) -> DecodedFrame:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Frame {frame_index} is not a readable image") from exc
    width, height = image.size
    resized = (
        image
        if image.size == (INPUT_WIDTH, INPUT_HEIGHT)
        # The API now sends source-resolution analysis frames.  PnLCalib still
        # owns a fixed 960x540 tensor, so use a high-quality single downsample
        # here instead of the old source -> 1280 JPEG -> bilinear chain.
        else image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return DecodedFrame(
        frame_index=frame_index,
        width=width,
        height=height,
        tensor=tensor,
        content_sha256=sha256(data).hexdigest(),
    )
