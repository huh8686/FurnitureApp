"""Step 1 — SAM3 텍스트 프롬프트 세그멘테이션으로 가구 배경 제거.

Input  : 가구가 포함된 이미지 (path / bytes / PIL.Image).
Output : SegmentationResult
    .image_rgb       EXIF 보정된 원본 RGB
    .image_rgba      배경 alpha=0, 가구 alpha=255 인 RGBA  ← Step 3 입력
    .protect_mask_01 [H,W] float, 가구=1
    .masks_np        SAM3 raw masks [N,H,W]
    .scores_np       SAM3 scores [N]
    .selected_indices  최종 protect_mask 에 합쳐진 인스턴스 인덱스들
    .debug_overlay   모든 인스턴스를 색칠한 RGB 이미지 (인덱스/점수 라벨 포함)
    .orientation     EXIF 보정 정보
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
from pathlib import Path
from PIL import Image

from utils.images import (
    apply_alpha_mask,
    load_image_bytes,
    visualize_instance_masks,
)
from utils.orientation import OrientationInfo, auto_orient

from .base import PipelineStep

ImageLike = Union[str, Path, bytes, Image.Image]
SelectMode = Literal["union", "top", "largest", "center", "foreground", "index"]


@dataclass
class SegmentationResult:
    image_rgb: Image.Image
    image_rgba: Image.Image
    protect_mask_01: np.ndarray
    masks_np: np.ndarray
    scores_np: np.ndarray
    orientation: OrientationInfo
    selected_indices: list[int] = field(default_factory=list)
    debug_overlay: Optional[Image.Image] = None


class Sam3FurnitureSegmenter(PipelineStep):
    """`facebook/sam3` 로 가구 마스크 검출 → 배경 제거.

    여러 가구가 잡힐 때 어떤 것을 살릴지 ``select`` 로 결정한다.

    Args:
        text_prompt: SAM3 가 찾을 대상 (기본 "furniture").
        select:
            - "union"      — 검출된 모든 인스턴스 합집합.
            - "top"        — SAM3 score 최고 1개.
            - "largest"    — mask area 최대 1개 (앞 가구 잡기에 가장 안전, 기본).
            - "center"     — 이미지 중심과 가까운 mask 1개.
            - "foreground" — 0.6 * area + 0.4 * (1 - center_dist) 점수 1등.
            - "index"      — ``select_index`` 로 지정한 인덱스 1개.
        select_index: ``select="index"`` 일 때 사용할 인스턴스 번호.
        score_threshold / mask_threshold: SAM3 후처리 임계값.
        device: "cuda" / "cpu" / None(자동 감지).
        hf_model_id: HuggingFace 모델 ID.
    """

    name = "segment"

    def __init__(
        self,
        text_prompt: str = "furniture",
        select: SelectMode = "largest",
        select_index: int = 0,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        device: Optional[str] = None,
        hf_model_id: str = "facebook/sam3",
    ) -> None:
        self.text_prompt = text_prompt
        self.select = select
        self.select_index = select_index
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.device = device
        self.hf_model_id = hf_model_id

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def run(self, image: ImageLike) -> SegmentationResult:
        import torch
        from transformers import Sam3Model, Sam3Processor

        raw = load_image_bytes(image)
        base_image, info = auto_orient(raw)

        device = self._resolve_device()
        model = Sam3Model.from_pretrained(
            self.hf_model_id, use_safetensors=True
        ).to(device)
        processor = Sam3Processor.from_pretrained(self.hf_model_id)
        model.eval()

        try:
            inputs = processor(
                images=base_image,
                text=self.text_prompt,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            masks = results["masks"].cpu().numpy().astype(np.float32)
            scores = results["scores"].cpu().numpy().astype(np.float64)
        finally:
            del model
            del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if masks.shape[0] == 0:
            raise RuntimeError(
                f"SAM3 가 '{self.text_prompt}' 를 검출하지 못했습니다."
            )

        protect, selected = self._select_protect_mask(masks, scores)
        protect = np.clip(protect, 0.0, 1.0).astype(np.float32)

        rgba = apply_alpha_mask(base_image, protect)
        debug = visualize_instance_masks(
            base_image, masks, scores_np=scores, mask_threshold=self.mask_threshold
        )

        return SegmentationResult(
            image_rgb=base_image,
            image_rgba=rgba,
            protect_mask_01=protect,
            masks_np=masks,
            scores_np=scores,
            orientation=info,
            selected_indices=selected,
            debug_overlay=debug,
        )

    # ---- selection helpers --------------------------------------------------

    def _instance_features(
        self, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """각 instance 의 (area_frac, center_dist_norm) 계산."""
        n = masks.shape[0]
        h, w = masks.shape[1], masks.shape[2]
        areas = np.zeros(n, dtype=np.float64)
        center_dist = np.zeros(n, dtype=np.float64)
        diag = float(np.hypot(w, h))
        cx0, cy0 = w / 2.0, h / 2.0
        for i in range(n):
            m = masks[i].squeeze() > self.mask_threshold
            cnt = int(m.sum())
            areas[i] = cnt / float(h * w)
            if cnt == 0:
                center_dist[i] = 1.0
                continue
            ys, xs = np.where(m)
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            center_dist[i] = float(np.hypot(cx - cx0, cy - cy0)) / diag
        return areas, center_dist

    def _select_protect_mask(
        self, masks: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, list[int]]:
        n = masks.shape[0]
        if self.select == "union":
            protect = np.zeros(masks.shape[1:], dtype=np.float32)
            for i in range(n):
                protect = np.maximum(protect, masks[i].squeeze())
            return protect, list(range(n))

        if self.select == "top":
            best = int(np.argmax(scores))
        elif self.select == "largest":
            areas, _ = self._instance_features(masks)
            best = int(np.argmax(areas))
        elif self.select == "center":
            _, cdist = self._instance_features(masks)
            best = int(np.argmin(cdist))
        elif self.select == "foreground":
            areas, cdist = self._instance_features(masks)
            a_norm = areas / (areas.max() if areas.max() > 0 else 1.0)
            c_score = 1.0 - cdist
            fg_score = 0.6 * a_norm + 0.4 * c_score
            best = int(np.argmax(fg_score))
        elif self.select == "index":
            if not 0 <= self.select_index < n:
                raise IndexError(
                    f"select_index={self.select_index} 가 범위 밖입니다 "
                    f"(0 ≤ idx < {n})."
                )
            best = int(self.select_index)
        else:  # pragma: no cover
            raise ValueError(f"unknown select mode: {self.select}")

        return masks[best].squeeze().astype(np.float32), [best]
