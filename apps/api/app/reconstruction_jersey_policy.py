"""Versioned sampling and fusion policy for reconstruction jersey evidence."""

from dataclasses import replace

from .jersey_ocr_contract import JerseyFusionConfig


JERSEY_OCR_PRE_RESOLVER_MAX_SELECTED_FRAMES = 5
JERSEY_OCR_MAX_CROPS_PER_PROSPECTIVE_PARTITION = 5
JERSEY_OCR_MIN_CROP_GAP_SECONDS = 0.45

JERSEY_OCR_FUSION_CONFIG = JerseyFusionConfig(
    min_ocr_confidence=0.01,
    min_frame_quality=0.0,
    min_back_visibility=0.0,
    min_effective_score=0.01,
)

JERSEY_OCR_PRE_RESOLVER_FUSION_CONFIG = replace(
    JERSEY_OCR_FUSION_CONFIG,
    max_selected_frames=JERSEY_OCR_PRE_RESOLVER_MAX_SELECTED_FRAMES,
)
