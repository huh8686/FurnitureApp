"""Step 4 — styled image -> text shot plan for Sora.

The Sora path does not need a generated 3x3 planning image. Instead, GPT-4o
looks at the final styled still and writes a concise shot plan that Step 5 can
fold directly into the video prompt while the styled still is used as Sora's
visual reference.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from utils.images import load_image_bytes

from .base import PipelineStep, require_openai_api_key

ImageLike = Union[str, Path, bytes, Image.Image]


SHOT_PLAN_SYSTEM_PROMPT = (
    "You are a senior commercial video director for furniture and "
    "interior products. You will receive a single styled product image. "
    "Write an 8-shot camera plan for ONE short Sora video that "
    "showcases this exact furniture in this exact room. The plan "
    "should evolve from a wide establishing view to detail and "
    "lifestyle looks and back to a satisfying close.\n\n"
    "Output plain text only, in EXACTLY this numbered structure. Each "
    "move is 1–3 sentences and specifies framing (wide / medium / "
    "close-up), camera direction, and what is featured:\n\n"
    "1. Establishing move: ...\n"
    "2. Hero furniture move: ...\n"
    "3. Detail styling move: ...\n"
    "4. Material close-up move: ...\n"
    "5. Functional/lifestyle move: ...\n"
    "6. Surface or storage detail move: ...\n"
    "7. Light and shadow move: ...\n"
    "8. Closing move: ...\n\n"
    "Finally, add a line named 'Continuity notes:' covering lighting "
    "consistency, color palette, smooth transitions between shots, "
    "and what to keep as the focal point throughout the video."
)


@dataclass
class ShotPlanResult:
    shot_plan: str
    output_path: Optional[Path]
    instruction: str


class FurnitureShotPlanGenerator(PipelineStep):
    """Styled image -> single continuous camera-path plan (GPT-4o vision)."""

    name = "shot_plan"

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.4,
        max_tokens: int = 1200,
        api_key: Optional[str] = None,
        instruction: str = SHOT_PLAN_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.instruction = instruction

    def run(
        self,
        styled_image: ImageLike,
        *,
        extra_instruction: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> ShotPlanResult:
        from openai import OpenAI

        client = OpenAI(api_key=require_openai_api_key(self.api_key))
        img_b64 = base64.b64encode(load_image_bytes(styled_image)).decode("ascii")

        user_text = (
            "Analyze this styled furniture image and write the video shot plan. "
            "Make it specific to the visible furniture, room, lighting, and "
            "camera perspective."
        )
        if extra_instruction:
            user_text += f"\n\nAdditional user request: {extra_instruction.strip()}"

        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        )
        shot_plan = (resp.choices[0].message.content or "").strip()
        if not shot_plan:
            raise RuntimeError("Step4 shot plan 응답이 비어 있습니다.")

        out_path = None
        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(shot_plan, encoding="utf-8")

        return ShotPlanResult(
            shot_plan=shot_plan,
            output_path=out_path,
            instruction=self.instruction,
        )
