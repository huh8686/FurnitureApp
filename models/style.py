"""Step 2 — 가구 이미지 → 어울리는 인테리어 스타일 2개 추천.

GPT-4o (vision) 가 가구를 보고 서로 결이 다른 스타일 2개를 JSON 으로 반환.
각 항목에는 짧은 name 과, Step 3 의 user prompt 로 그대로 쓸 수 있는 long
영문 prompt 가 들어 있다.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from utils.images import load_image_bytes

from .base import PipelineStep, require_openai_api_key

ImageLike = Union[str, Path, bytes, Image.Image]


SYSTEM_PROMPT = (
    "You are a senior interior designer. You will receive a photograph of "
    "ONE piece of furniture (its background may be transparent or removed). "
    "Propose EXACTLY TWO distinct interior styles that would visually "
    "complement the furniture's material, color, and form. The two styles "
    "must be meaningfully different from each other (e.g., one warm and "
    "organic, one cool and minimal). For each style provide:\n"
    "  - `name`: short style name (e.g., 'Warm Scandinavian').\n"
    "  - `prompt`: a long English description usable as a text-to-image "
    "style prompt for a background generator. Mention palette, materials, "
    "lighting direction and color temperature, mood, and a few props.\n"
    "Output STRICT JSON only, matching this schema:\n"
    "{\"styles\":[{\"name\":\"...\",\"prompt\":\"...\"},"
    "{\"name\":\"...\",\"prompt\":\"...\"}]}\n"
    "No markdown, no preamble, no extra keys."
)


@dataclass(frozen=True)
class StyleSuggestion:
    name: str
    prompt: str


@dataclass
class StyleRecommendation:
    styles: list[StyleSuggestion]

    def __getitem__(self, idx: int) -> StyleSuggestion:
        return self.styles[idx]

    def __len__(self) -> int:
        return len(self.styles)


class StyleRecommender(PipelineStep):
    """가구 이미지 → 2개 스타일 추천 (gpt-4o vision)."""

    name = "style"

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.5,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = api_key

    def run(self, furniture_image: ImageLike) -> StyleRecommendation:
        from openai import OpenAI

        client = OpenAI(api_key=require_openai_api_key(self.api_key))
        b64 = base64.b64encode(load_image_bytes(furniture_image)).decode("ascii")

        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Suggest exactly two complementary interior "
                                "styles for this furniture. Output JSON only."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Step2 LLM 응답이 JSON 이 아닙니다: {text[:300]}"
            ) from e

        items = payload.get("styles") or []
        if len(items) < 2:
            raise RuntimeError(
                f"Step2 가 스타일 2개를 반환하지 않았습니다: {text[:300]}"
            )
        styles = [
            StyleSuggestion(name=str(s["name"]), prompt=str(s["prompt"]))
            for s in items[:2]
        ]
        return StyleRecommendation(styles=styles)
