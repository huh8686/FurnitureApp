"""EXIF Orientation 안전 보정.

PIL.ImageOps.exif_transpose 는 회전 자체는 잘 하지만, 회전 후 EXIF 를
재직렬화하면서 깨진 RATIONAL 태그(예: 0/0 GPS) 때문에 ZeroDivisionError
가 날 수 있다. 그래서 orientation 태그만 안전하게 읽어 직접 transpose 하고
EXIF 는 통째로 버린다.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image

EXIF_ORIENTATION_TAG = 0x0112  # 274

_ORIENTATION_TO_TRANSPOSE: dict[int, int] = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,  # CW 90
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,   # CCW 90
}


@dataclass(frozen=True)
class OrientationInfo:
    raw_size: tuple[int, int]
    corrected_size: tuple[int, int]
    exif_orientation: int
    rotated: bool
    is_portrait: bool


def _read_orientation(src: Image.Image) -> int:
    try:
        exif = src.getexif() or {}
        v = exif.get(EXIF_ORIENTATION_TAG, 1)
        iv = int(v) if v is not None else 1
        return iv if iv in (1, 2, 3, 4, 5, 6, 7, 8) else 1
    except Exception:
        return 1


def auto_orient(
    image: Union[bytes, str, Path, Image.Image],
) -> tuple[Image.Image, OrientationInfo]:
    """입력을 EXIF 회전 보정된 RGB 이미지와 정보로 반환."""
    if isinstance(image, (str, Path)):
        src = Image.open(io.BytesIO(Path(image).read_bytes()))
    elif isinstance(image, bytes):
        src = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        src = image
    else:
        raise TypeError(f"Unsupported image input: {type(image)}")

    src.load()
    raw_size = src.size
    orient = _read_orientation(src)
    op = _ORIENTATION_TO_TRANSPOSE.get(orient)
    rotated = src.transpose(op) if op is not None else src.copy()
    fixed = rotated.convert("RGB")
    w, h = fixed.size

    return fixed, OrientationInfo(
        raw_size=raw_size,
        corrected_size=(w, h),
        exif_orientation=orient,
        rotated=(raw_size != (w, h)),
        is_portrait=h > w,
    )
