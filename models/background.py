"""Step 3 — 배경 제거된 가구 + 스타일 prompt → 새 배경 (with relighting).

구조:
  - 공통 hard constraint(가구 보존 / 전체 relight / seam 제거 ...)는
    ``BaseBackgroundGenerator.base_system_instruction`` 에 한 번만 정의.
  - 콘텐츠 용도별 컴포지션/비율은 서브클래스가 ``use_case_instruction`` 과
    ``default_size`` 를 override 해서 base 위에 append.
  - 최종 prompt = base system  +  use case 조항  +  (선택) extra hard rule
                 +  Step 2 가 추천한 style prompt (user 영역).

서브클래스:
  - ``InstagramFeedBackgroundGenerator``  (1:1, 피드 hero shot)
  - ``OhouThumbnailBackgroundGenerator``  (1:1, 오늘의집 lived-in vibe)
  - ``ShortsBackgroundGenerator``         (2:3, 세로 숏폼 첫 프레임)
  - ``BackgroundGenerator``               (generic / 입력 비율 따라감, 하위 호환)
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image

from utils.images import pick_image_size, png_filelike

from .base import PipelineStep, require_openai_api_key


BASE_SYSTEM_INSTRUCTION = (
    "You are an expert photo editor. The input image has a foreground "
    "subject (opaque, alpha > 0) — a single piece of furniture — and a "
    "transparent (alpha = 0) region that you must regenerate as a brand "
    "new background.\n\n"
    "Hard constraints (ALWAYS apply, regardless of use case):\n"
    "1. PRESERVE the foreground furniture exactly: identity, geometry, "
    "material, color, and fine details. Do NOT redesign, restyle, or "
    "reshape the furniture in any way.\n"
    "2. Do NOT simply paste a new background behind the cutout. Globally "
    "re-light, color-grade, and white-balance the ENTIRE image so the "
    "furniture looks like it was physically photographed inside the newly "
    "generated scene.\n"
    "3. Match the direction, hardness, color temperature, and intensity "
    "of the new background's implied light source. Add consistent contact "
    "shadows, ambient occlusion, and rim/edge lighting on the furniture.\n"
    "4. Eliminate hard cutout seams; ensure smooth transitions between the "
    "kept furniture and the new environment.\n"
    "5. Keep the original camera perspective, focal length, and "
    "depth-of-field feel.\n"
    "6. Output a single coherent, photorealistic image."
)


@dataclass
class BackgroundResult:
    image: Image.Image
    requested_size: str
    full_prompt: str
    style_used: str
    use_case: str = "generic"


class BaseBackgroundGenerator(PipelineStep):
    """가구 RGBA + 스타일 prompt → 새 배경 + 재조명된 가구 이미지.

    서브클래스는 ``use_case``, ``use_case_instruction``, ``default_size`` 만
    덮어쓰면 됩니다. base 의 system prompt 와 호출 흐름은 그대로 재사용.
    """

    name = "background"

    # subclass override 지점
    use_case: str = "generic"
    use_case_instruction: str = ""
    default_size: Optional[str] = None
    base_system_instruction: str = BASE_SYSTEM_INSTRUCTION

    def __init__(
        self,
        model: str = "gpt-image-1",
        quality: str = "high",
        api_key: Optional[str] = None,
        base_system_instruction: Optional[str] = None,
        use_case_instruction: Optional[str] = None,
        default_size: Optional[str] = None,
    ) -> None:
        self.model = model
        self.quality = quality
        self.api_key = api_key
        if base_system_instruction is not None:
            self.base_system_instruction = base_system_instruction
        if use_case_instruction is not None:
            self.use_case_instruction = use_case_instruction
        if default_size is not None:
            self.default_size = default_size

    def compose_system_instruction(self, extra: Optional[str] = None) -> str:
        parts: list[str] = [self.base_system_instruction]
        if self.use_case_instruction:
            parts.append(
                f"Composition rules for use case '{self.use_case}':\n"
                f"{self.use_case_instruction}"
            )
        if extra and extra.strip():
            parts.append(f"Additional hard constraints:\n{extra.strip()}")
        return "\n\n".join(parts)

    def resolve_size(
        self,
        furniture_rgba: Image.Image,
        explicit_size: Optional[str],
    ) -> str:
        if explicit_size:
            return explicit_size
        if self.default_size:
            return self.default_size
        return pick_image_size(*furniture_rgba.size)

    def run(
        self,
        furniture_rgba: Image.Image,
        style_prompt: str,
        *,
        size: Optional[str] = None,
        extra_system_instruction: Optional[str] = None,
    ) -> BackgroundResult:
        from openai import OpenAI

        if furniture_rgba.mode != "RGBA":
            raise ValueError(
                "furniture_rgba 는 RGBA (alpha=가구 마스크) 여야 합니다."
            )
        if not style_prompt or not style_prompt.strip():
            raise ValueError("style_prompt 가 비어 있습니다.")

        client = OpenAI(api_key=require_openai_api_key(self.api_key))
        chosen_size = self.resolve_size(furniture_rgba, size)
        system = self.compose_system_instruction(extra_system_instruction)
        full_prompt = (
            f"{system}\n\n"
            f"User-requested style for the new background:\n"
            f"{style_prompt.strip()}"
        )

        png = png_filelike(furniture_rgba, name="furniture.png")
        result = client.images.edit(
            model=self.model,
            image=png,
            prompt=full_prompt,
            size=chosen_size,
            quality=self.quality,
        )
        item = result.data[0]
        if getattr(item, "b64_json", None):
            data = base64.b64decode(item.b64_json)
        elif getattr(item, "url", None):
            import requests

            r = requests.get(item.url, timeout=60)
            r.raise_for_status()
            data = r.content
        else:
            raise RuntimeError("응답에 b64_json/url 이 없습니다.")

        return BackgroundResult(
            image=Image.open(io.BytesIO(data)).convert("RGB"),
            requested_size=chosen_size,
            full_prompt=full_prompt,
            style_used=style_prompt.strip(),
            use_case=self.use_case,
        )


# ---------------------------------------------------------------------------
# Use case 별 서브클래스
# ---------------------------------------------------------------------------


class BackgroundGenerator(BaseBackgroundGenerator):
    """Generic (use case 미지정). 입력 가구 사진의 비율을 그대로 따라간다."""

    use_case = "generic"
    use_case_instruction = ""
    default_size = None


class InstagramFeedBackgroundGenerator(BaseBackgroundGenerator):
    """인스타그램 피드 업로드용 1:1 hero shot (editorial / aesthetic-feed look)."""

    use_case = "instagram_feed"
    default_size = "1024x1024"
    use_case_instruction = (
        "Visual style:\n"
        "- Editorial / aesthetic-feed look. Muted earth-tone palette "
        "(sand, bone, oat, warm taupe, soft beige) with at most one or "
        "two restrained accent colors.\n"
        "- Soft directional natural light from a large window or golden "
        "hour. Highlights are gentle, shadows are lifted and matte — "
        "never crushed.\n"
        "- Subtle film-like color grade: slight warmth, mild desaturation "
        "in highlights, very low overall contrast. ABSOLUTELY NO aggressive "
        "Instagram filters, HDR look, or oversaturated palette.\n"
        "- Styling props are carefully minimal (one ceramic, one small "
        "plant, one folded textile, a single book at most). The frame must "
        "read like a slow-living / Pinterest-aesthetic post, never "
        "cluttered.\n"
        "- Surface finish reads slightly matte and photographic, never "
        "plasticky or CG.\n\n"
        "Composition:\n"
        "- Square 1:1 framing. Place the hero furniture confidently in "
        "the visual center.\n"
        "- Keep the top third and bottom third visually clean to leave "
        "room for caption / sticker overlays; avoid placing critical "
        "details there.\n"
        "- Treat this as a strong first-impression hero shot: bold "
        "framing, balanced symmetry, magazine-clean styling.\n"
        "- Image must remain instantly readable as a small thumbnail "
        "inside a busy feed grid."
    )


class OhouThumbnailBackgroundGenerator(BaseBackgroundGenerator):
    """오늘의집 썸네일 업로드용 — 실거주 vibe."""

    use_case = "ohou_thumbnail"
    default_size = "1024x1024"
    use_case_instruction = (
        "Visual style:\n"
        "- Authentic Korean home (오늘의집) vibe. Soft daytime window "
        "light through sheer curtains, slightly warm-neutral white walls, "
        "light wood floor or matte beige tones dominating the palette.\n"
        "- Lived-in but tidy: subtle imperfections (a folded blanket, a "
        "half-read book, a low ceramic vase, a small houseplant, soft "
        "textiles) — never cluttered, never staged like a showroom.\n"
        "- Color grade: clean and slightly bright, low contrast, "
        "natural-looking whites; no aggressive filter, no cinematic "
        "color treatment.\n"
        "- Materials read photographically: real wood grain, real fabric "
        "weave, real ceramic, not CG or plastic.\n\n"
        "Composition:\n"
        "- Square 1:1 framing. Frame the hero furniture as a real part of "
        "a believable interior, not isolated; show enough surrounding "
        "context (a corner of a window, a bit of floor, neighboring "
        "objects) to imagine the room.\n"
        "- Eye-level or slightly-below-eye-level camera, consistent with "
        "how Korean home photos are typically taken.\n"
        "- Thumbnail must remain readable at small scroll-card sizes in "
        "the Ohou feed."
    )


class ShortsBackgroundGenerator(BaseBackgroundGenerator):
    """숏폼 (Reels / Shorts / TikTok) 영상의 세로 첫 프레임 — cinematic hook."""

    use_case = "shorts"
    default_size = "1024x1536"  # gpt-image-1 이 지원하는 사이즈 중 9:16 에 최근접
    use_case_instruction = (
        "Visual style:\n"
        "- Cinematic look optimized as the FIRST FRAME of a vertical "
        "Reels / Shorts / TikTok video — it must work as a scroll-stop "
        "hook within the first second.\n"
        "- Strong directional lighting (side light, backlight, or window "
        "light with rim) producing visible depth, separation of the "
        "furniture from the background, and a gentle natural vignette.\n"
        "- Cinematic color grade: rich but controlled shadows, "
        "highlights kept off pure white, subtle warm/teal balance, mild "
        "film-grain feel. HIGHER contrast than the Instagram feed look, "
        "but still fully photorealistic — NOT CGI, NOT illustration.\n"
        "- Atmospheric depth cues: soft haze, dust in light beams, "
        "gentle bokeh in the far background, layered foreground / "
        "midground / background. Air and depth must be felt.\n"
        "- Materials read photographically: real wood, real fabric, real "
        "metal — no plastic CG look.\n\n"
        "Composition:\n"
        "- Tall vertical framing. The hero furniture occupies the central "
        "vertical band so it stays visible across mobile crops and safe "
        "areas.\n"
        "- Leave breathing room above and below the furniture; that "
        "space will be traversed by camera motion (tilt / dolly / push-in) "
        "in the resulting video.\n"
        "- Use strong vertical lines and clear depth lanes (lines of "
        "sight, perspective, layered planes) so the resulting video reads "
        "as dynamic, not flat.\n"
        "- Lighting and color must remain consistent throughout the full "
        "vertical frame because the camera will travel across it."
    )


USE_CASE_REGISTRY: dict[str, type[BaseBackgroundGenerator]] = {
    InstagramFeedBackgroundGenerator.use_case: InstagramFeedBackgroundGenerator,
    OhouThumbnailBackgroundGenerator.use_case: OhouThumbnailBackgroundGenerator,
    ShortsBackgroundGenerator.use_case: ShortsBackgroundGenerator,
}
