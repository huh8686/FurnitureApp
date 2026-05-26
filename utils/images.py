"""이미지 직렬화·리사이즈·마스크 유틸."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageLike = Union[str, Path, bytes, Image.Image]


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def load_image_bytes(image: ImageLike) -> bytes:
    """다양한 입력 (path/bytes/PIL) → PNG 바이트로 통일."""
    if isinstance(image, (str, Path)):
        return Path(image).read_bytes()
    if isinstance(image, bytes):
        return image
    if isinstance(image, Image.Image):
        return to_png_bytes(image)
    raise TypeError(f"Unsupported image type: {type(image)}")


def png_filelike(image: Image.Image, name: str = "image.png") -> io.BytesIO:
    """OpenAI SDK multipart 업로드용 파일유사 객체."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    buf.name = name
    return buf


def apply_alpha_mask(rgb: Image.Image, mask_01: np.ndarray) -> Image.Image:
    """RGB + 가구=1/배경=0 마스크 → 배경 투명한 RGBA."""
    rgba = rgb.convert("RGBA")
    if mask_01.ndim > 2:
        mask_01 = mask_01.squeeze()
    alpha = (np.clip(mask_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_pil = Image.fromarray(alpha, mode="L").resize(
        rgba.size, Image.Resampling.NEAREST
    )
    rgba.putalpha(alpha_pil)
    return rgba


_DEBUG_PALETTE: list[tuple[int, int, int]] = [
    (229, 57, 53), (30, 136, 229), (0, 150, 136), (255, 193, 7),
    (156, 39, 176), (76, 175, 80), (255, 87, 34), (63, 81, 181),
    (158, 158, 158), (121, 85, 72),
]


def visualize_instance_masks(
    image: Image.Image,
    masks_np: np.ndarray,
    *,
    scores_np: "np.ndarray | None" = None,
    alpha: float = 0.45,
    mask_threshold: float = 0.5,
) -> Image.Image:
    """각 instance mask 를 다른 색으로 합성하고 중심에 인덱스/점수 라벨.

    인덱스 = ``--sam-index`` 로 그대로 지정할 수 있는 정수. 큰 마스크가 작은
    마스크를 가리지 않도록 면적이 큰 순서대로 먼저 그리고 작은 것을 위에
    얹는다.
    """
    from PIL import ImageDraw, ImageFont

    img = image.convert("RGBA")
    if masks_np.shape[0] == 0:
        return img.convert("RGB")

    areas = (masks_np > mask_threshold).reshape(masks_np.shape[0], -1).sum(axis=1)
    draw_order = np.argsort(-areas)

    for rank, idx in enumerate(draw_order):
        i = int(idx)
        m = masks_np[i].squeeze() > mask_threshold
        if not m.any():
            continue
        color = _DEBUG_PALETTE[i % len(_DEBUG_PALETTE)]
        mask_pil = Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize(
            img.size, Image.Resampling.NEAREST
        )
        overlay = Image.new("RGBA", img.size, color + (0,))
        overlay.putalpha(mask_pil.point(lambda v: int(v * alpha)))
        img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
        )
    except OSError:
        font = ImageFont.load_default()
    for i in range(masks_np.shape[0]):
        m = masks_np[i].squeeze() > mask_threshold
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        sx = cx * img.width / m.shape[1]
        sy = cy * img.height / m.shape[0]
        area_frac = int(m.sum()) / (m.shape[0] * m.shape[1])
        if scores_np is not None:
            label = f"{i}  s={float(scores_np[i]):.2f}  a={area_frac:.2f}"
        else:
            label = f"{i}  a={area_frac:.2f}"
        pad = 6
        bbox = draw.textbbox((sx, sy), label, font=font)
        box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        draw.rectangle(box, fill=(0, 0, 0, 220))
        draw.text((sx, sy), label, font=font, fill=(255, 255, 255, 255))
    return img.convert("RGB")


_IMAGE_EDIT_SIZES: dict[str, float] = {
    "1024x1024": 1.0,
    "1536x1024": 1536 / 1024,
    "1024x1536": 1024 / 1536,
}


def pick_image_size(width: int, height: int) -> str:
    """gpt-image-1 지원 해상도 중 입력 비율과 가장 가까운 것."""
    aspect = width / max(1, height)
    return min(_IMAGE_EDIT_SIZES.items(), key=lambda kv: abs(kv[1] - aspect))[0]


_SORA_SIZES_STD: dict[str, float] = {
    "1280x720": 1280 / 720,
    "720x1280": 720 / 1280,
    "1792x1024": 1792 / 1024,
    "1024x1792": 1024 / 1792,
}
_SORA_SIZES_PRO: dict[str, float] = {
    "1920x1080": 1920 / 1080,
    "1080x1920": 1080 / 1920,
}


def pick_sora_size(width: int, height: int, *, pro: bool = False) -> str:
    candidates = _SORA_SIZES_PRO if pro else _SORA_SIZES_STD
    aspect = width / max(1, height)
    return min(candidates.items(), key=lambda kv: abs(kv[1] - aspect))[0]


def parse_size(size: str) -> tuple[int, int]:
    w, h = size.lower().split("x")
    return int(w), int(h)


def letterbox_to_size(
    image: Image.Image,
    size: str,
    fill: tuple[int, int, int] = (245, 242, 238),
) -> Image.Image:
    """비율 유지하며 size 픽셀 크기로 레터박스."""
    target_w, target_h = parse_size(size)
    src = image.convert("RGB")
    scale = min(target_w / src.width, target_h / src.height)
    nw = max(1, int(src.width * scale))
    nh = max(1, int(src.height * scale))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    canvas.paste(resized, ((target_w - nw) // 2, (target_h - nh) // 2))
    return canvas


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    """긴 변이 ``max_side`` 픽셀이 되도록 비율 유지 축소. 작으면 그대로."""
    w, h = image.size
    m = max(w, h)
    if m <= max_side:
        return image.copy()
    s = max_side / m
    return image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
