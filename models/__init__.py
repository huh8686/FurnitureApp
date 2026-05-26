"""5-step furniture-to-video pipeline.

segment    : Sam3FurnitureSegmenter        (facebook/sam3)
style      : StyleRecommender              (gpt-4o vision, JSON, 2 styles)
background : BackgroundGenerator           (gpt-image-1, system=relight, user=style)
shot_plan  : FurnitureShotPlanGenerator    (gpt-4o vision, text shot plan)
video      : SoraReferenceVideo            (gpt-4o + sora-2)
"""

from .background import (
    BackgroundGenerator,
    BackgroundResult,
    BaseBackgroundGenerator,
    InstagramFeedBackgroundGenerator,
    OhouThumbnailBackgroundGenerator,
    ShortsBackgroundGenerator,
    USE_CASE_REGISTRY,
)
from .base import PipelineStep, require_openai_api_key
from .segment import Sam3FurnitureSegmenter, SegmentationResult
from .shot_plan import FurnitureShotPlanGenerator, ShotPlanResult
from .style import StyleRecommendation, StyleRecommender, StyleSuggestion
from .video_ref import SoraReferenceVideo, VideoResult

__all__ = [
    "PipelineStep",
    "require_openai_api_key",
    "Sam3FurnitureSegmenter",
    "SegmentationResult",
    "StyleRecommender",
    "StyleRecommendation",
    "StyleSuggestion",
    "BackgroundGenerator",
    "BackgroundResult",
    "BaseBackgroundGenerator",
    "InstagramFeedBackgroundGenerator",
    "OhouThumbnailBackgroundGenerator",
    "ShortsBackgroundGenerator",
    "USE_CASE_REGISTRY",
    "FurnitureShotPlanGenerator",
    "ShotPlanResult",
    "SoraReferenceVideo",
    "VideoResult",
]
