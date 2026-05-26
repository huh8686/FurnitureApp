"""Step 5 — styled image + shot plan -> Sora 2 video.

Sora receives the Step 3 styled image as `input_reference`, so the generated
video starts from the actual furniture/background result instead of relying on
text alone. GPT-4o writes the final Sora prompt by combining the visual reference
with the Step 4 shot plan.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image

from utils.images import letterbox_to_size, load_image_bytes

from .base import PipelineStep, require_openai_api_key

ImageLike = Union[str, Path, bytes, Image.Image]


LLM_SYSTEM_INSTRUCTION = (
    "You are a senior video director writing a SINGLE cohesive prompt for "
    "Sora 2, an image-to-video model. You will receive the exact "
    "first-frame reference image and a numbered shot plan.\n\n"
    "Use the reference image as the ground truth for the scene, the "
    "furniture, the lighting, and the color palette. Translate the "
    "numbered shot plan into one flowing video prompt that describes each "
    "shot in order, with smooth transitions between them.\n\n"
    "Output ONLY the final prompt text. Use natural prose paragraphs — "
    "one paragraph per shot in the plan, in the same order. Maintain the "
    "lighting, palette, and focal point from the 'Continuity notes' "
    "section, and end with a short line stating that no faces, text, or "
    "logos should be visible."
)


@dataclass
class VideoResult:
    video_id: str
    status: str
    model: str
    size: str
    seconds: str
    output_path: Path
    prompt_path: Path
    video_prompt: str


class SoraReferenceVideo(PipelineStep):
    """Styled image reference + text shot plan -> MP4 (GPT-4o + Sora 2)."""

    name = "video"

    def __init__(
        self,
        video_model: str = "sora-2",
        size: str = "1280x720",
        seconds: str = "8",
        llm_model: str = "gpt-4o",
        max_prompt_chars: int = 3500,
        poll_interval: float = 10.0,
        timeout: float = 900.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.video_model = video_model
        self.size = size
        self.seconds = seconds
        self.llm_model = llm_model
        self.max_prompt_chars = max_prompt_chars
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.api_key = api_key

    def _generate_video_prompt(
        self,
        client,
        reference_b64: str,
        shot_plan: str,
        extra: Optional[str],
    ) -> str:
        user_text = (
            "Write one final Sora prompt using the image as the EXACT "
            "visual reference and the shot plan below. Keep the prompt "
            f"under {self.max_prompt_chars} characters.\n\n"
            f"Shot plan:\n{shot_plan.strip()}\n\n"
            "Translate each numbered move into a prose paragraph in the "
            "same order, describing camera framing, direction, and what "
            "is featured. Maintain the lighting, palette, and focal "
            "point from the Continuity notes."
        )
        if extra:
            user_text += f"\n\nAdditional constraints: {extra.strip()}"

        resp = client.chat.completions.create(
            model=self.llm_model,
            temperature=0.45,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{reference_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Step5 final video prompt 응답이 비어 있습니다.")
        if len(text) > self.max_prompt_chars:
            text = text[: self.max_prompt_chars].rsplit(" ", 1)[0]
        return text

    def run(
        self,
        reference_image: ImageLike,
        shot_plan: str,
        output_path: Union[str, Path],
        *,
        extra_llm_instruction: Optional[str] = None,
        on_progress: Optional[Callable[[str, int], None]] = None,
    ) -> VideoResult:
        from openai import OpenAI

        if not shot_plan or not shot_plan.strip():
            raise ValueError("shot_plan 이 비어 있습니다.")

        client = OpenAI(api_key=require_openai_api_key(self.api_key))
        reference_bytes = load_image_bytes(reference_image)
        reference_b64 = base64.b64encode(reference_bytes).decode("ascii")

        # (a) LLM 으로 최종 Sora prompt 작성
        video_prompt = self._generate_video_prompt(
            client,
            reference_b64,
            shot_plan,
            extra_llm_instruction,
        )
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path = out_path.with_suffix(".prompt.txt")
        prompt_path.write_text(video_prompt, encoding="utf-8")

        # (b) Step 3 결과 이미지를 Sora 첫 프레임 reference 로 전달
        ref_pil = Image.open(io.BytesIO(reference_bytes)).convert("RGB")
        ref_fit = letterbox_to_size(ref_pil, self.size)
        ref_buf = io.BytesIO()
        ref_fit.save(ref_buf, format="JPEG", quality=92)
        ref_buf.seek(0)
        ref_buf.name = "styled_reference.jpg"

        job = client.videos.create(
            model=self.video_model,
            prompt=video_prompt,
            size=self.size,
            seconds=self.seconds,
            input_reference=ref_buf,
        )
        if on_progress:
            on_progress(job.status, int(getattr(job, "progress", 0) or 0))

        # (c) 폴링
        deadline = time.time() + self.timeout
        last_progress = -1
        video = job
        while True:
            video = client.videos.retrieve(job.id)
            progress = int(getattr(video, "progress", 0) or 0)
            if on_progress and progress != last_progress:
                on_progress(video.status, progress)
                last_progress = progress
            if video.status == "completed":
                break
            if video.status == "failed":
                err = getattr(video, "error", None)
                msg = getattr(err, "message", None) if err else None
                raise RuntimeError(msg or "Sora video generation failed")
            if time.time() > deadline:
                raise TimeoutError(
                    f"Sora job {job.id} timed out after {self.timeout}s"
                )
            time.sleep(self.poll_interval)

        # (d) MP4 다운로드
        content = client.videos.download_content(job.id, variant="video")
        content.write_to_file(str(out_path))

        return VideoResult(
            video_id=job.id,
            status=video.status,
            model=getattr(video, "model", self.video_model) or self.video_model,
            size=getattr(video, "size", self.size) or self.size,
            seconds=str(getattr(video, "seconds", self.seconds)),
            output_path=out_path,
            prompt_path=prompt_path,
            video_prompt=video_prompt,
        )
