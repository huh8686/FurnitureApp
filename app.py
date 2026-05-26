"""5-step furniture → video pipeline (Streamlit UI).

흐름:
  Step 1  SAM3 가구 검출 → 사용자가 클릭/체크박스로 인스턴스 선택 → cutout RGBA
  Step 2  GPT-4o 스타일 추천 (2개) → 사용자가 1개 선택 (prompt 수정 가능)
  Step 3  gpt-image-1 로 새 배경 + 재조명 (system=relight, user=style)
  Step 4  GPT-4o 로 Step 3 이미지를 보고 8-shot 텍스트 플랜 생성
  Step 5  styled image reference + shot plan 으로 Sora 2 영상 생성

실행:
  cd furniture_video
  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from models import (  # noqa: E402
    BaseBackgroundGenerator,
    FurnitureShotPlanGenerator,
    InstagramFeedBackgroundGenerator,
    OhouThumbnailBackgroundGenerator,
    Sam3FurnitureSegmenter,
    ShortsBackgroundGenerator,
    SoraReferenceVideo,
    StyleRecommender,
)
from utils import (  # noqa: E402
    apply_alpha_mask,
    resize_max_side,
    visualize_instance_masks,
)

# ---------------------------------------------------------------------------
# 상수 / 세션 헬퍼
# ---------------------------------------------------------------------------

MAX_CLICK_SIDE = 1280
OUTPUT_DIR = ROOT / "output" / "app"

DOWNSTREAM_KEYS = {
    # SAM3 결과가 바뀌면 이후 step 결과 모두 무효
    "segment": [
        "selected_idx",
        "cutout_rgba",
        "sam_ts",
        "style_recs",
        "chosen_style_idx",
        "chosen_style_prompt",
        "bg_results",
        "bg_files",
        "bg_ts",
        "bg_style_slug",
        "chosen_use_case",
        "styled_image",
        "shot_plan",
        "shot_plan_editor",
        "shot_plan_file",
        "video_path",
        "video_prompt",
    ],
    # 가구 선택만 바뀌면 step 3~5 무효 (style 추천은 살림)
    "selection": [
        "bg_results",
        "bg_files",
        "bg_ts",
        "bg_style_slug",
        "chosen_use_case",
        "styled_image",
        "shot_plan",
        "shot_plan_editor",
        "shot_plan_file",
        "video_path",
        "video_prompt",
    ],
    "style": [
        "bg_results",
        "bg_files",
        "bg_ts",
        "bg_style_slug",
        "chosen_use_case",
        "styled_image",
        "shot_plan",
        "shot_plan_editor",
        "shot_plan_file",
        "video_path",
        "video_prompt",
    ],
    # Step 3 자체를 다시 돌릴 때 (use case 멀티셀렉트가 새로 호출됨)
    "background": [
        "bg_results",
        "bg_files",
        "bg_ts",
        "bg_style_slug",
        "chosen_use_case",
        "styled_image",
        "shot_plan",
        "shot_plan_editor",
        "shot_plan_file",
        "video_path",
        "video_prompt",
    ],
    # 결과 그리드에서 다른 use case 결과를 고른 경우 (Step 3 자체는 유효)
    "background_choice": [
        "shot_plan",
        "shot_plan_editor",
        "shot_plan_file",
        "video_path",
        "video_prompt",
    ],
    "shot_plan": ["video_path", "video_prompt"],
    "video": ["video_path", "video_prompt"],
}


def _clear(scope: str) -> None:
    for k in DOWNSTREAM_KEYS.get(scope, []):
        st.session_state.pop(k, None)


def _slug(value: str, max_len: int = 40) -> str:
    """파일명 안전 slug. 영숫자/하이픈/언더스코어만 남기고 lower-case."""
    cleaned = re.sub(r"[^\w\-]+", "_", (value or "").strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:max_len] or "untitled"


def _now_stamp() -> str:
    """현재 시각 ``YYYYMMDD_HHMMSS`` — 파일명 충돌 방지용."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# 클릭 좌표 헬퍼
# ---------------------------------------------------------------------------


def _click_to_original(click: dict, orig_w: int, orig_h: int) -> tuple[int, int]:
    dw = max(1, int(click.get("width", 1)))
    dh = max(1, int(click.get("height", 1)))
    x = int(click["x"] * orig_w / dw)
    y = int(click["y"] * orig_h / dh)
    return max(0, min(orig_w - 1, x)), max(0, min(orig_h - 1, y))


def _pick_mask_at(masks: np.ndarray, scores: np.ndarray, x: int, y: int) -> Optional[int]:
    order = np.argsort(-scores.reshape(-1))
    for ii in order:
        i = int(ii)
        m = masks[i].squeeze()
        if y < m.shape[0] and x < m.shape[1] and m[y, x] > 0.5:
            return i
    return None


def _union_protect(masks: np.ndarray, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros(masks.shape[1:], dtype=np.float32)
    out = np.zeros(masks.shape[1:], dtype=np.float32)
    for i in indices:
        out = np.maximum(out, masks[i].squeeze())
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 무거운 호출 캐시
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _segmenter(sam_prompt: str) -> Sam3FurnitureSegmenter:
    # `select="union"` 으로 모든 인스턴스 확보 → UI 에서 직접 선택.
    return Sam3FurnitureSegmenter(text_prompt=sam_prompt, select="union")


@st.cache_data(show_spinner="SAM3 가구 검출 중…")
def _run_segmentation(image_bytes: bytes, sam_prompt: str):
    seg = _segmenter(sam_prompt)
    return seg(image_bytes)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


st.set_page_config(page_title="FurnitureApp", layout="wide")
st.title("FurnitureApp")
st.caption(
    "SAM3 → 스타일 추천 → 배경 재생성 → 샷 플랜 → Sora 영상. "
    "각 단계의 산출물을 확인하고 다음 단계로 진행하세요."
)

# ===== Sidebar: 입력 / 키 상태 =====
with st.sidebar:
    st.header("입력")
    uploaded = st.file_uploader(
        "가구 이미지 업로드", type=["jpg", "jpeg", "png", "webp"]
    )
    default_sample = ROOT / "data" / "furniture.jpg"
    sample_path_str = st.text_input(
        "또는 샘플 경로", value=str(default_sample) if default_sample.exists() else ""
    )
    sam_prompt = st.text_input(
        "SAM3 prompt",
        value="furniture",
        help="예: furniture / cabinet / chair / sofa / desk",
    )

    st.divider()
    has_key = bool((os.environ.get("OPENAI_API_KEY") or "").strip())
    st.caption(f"OpenAI key: **{'OK' if has_key else '없음 — .env 또는 환경변수 필요'}**")
    st.caption(f"산출물 폴더: `{OUTPUT_DIR}`")

# 입력 이미지 확보
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    img_label = uploaded.name
elif sample_path_str and Path(sample_path_str).exists():
    raw_bytes = Path(sample_path_str).read_bytes()
    img_label = sample_path_str
else:
    st.info("사이드바에서 이미지를 업로드하거나 샘플 경로를 지정하세요.")
    st.stop()

img_sig = hashlib.md5(raw_bytes).hexdigest()
if st.session_state.get("_img_sig") != img_sig:
    # 새 이미지 → segment 부터 다 무효
    _clear("segment")
    st.session_state.pop("seg_result", None)
    st.session_state._img_sig = img_sig
    st.session_state._last_click_ut = None
    # 이전 이미지 체크박스 키 정리
    for k in list(st.session_state.keys()):
        if k.startswith("fur_cb_"):
            st.session_state.pop(k, None)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Step 1 — SAM3 + 가구 선택
# =====================================================================
st.header("Step 1 — SAM3 로 가구 검출 · 선택")
st.caption(f"입력: `{img_label}`  ·  prompt: `{sam_prompt}`")

c_run, c_hint = st.columns([1, 4])
with c_run:
    if st.button("SAM3 실행", type="primary"):
        _clear("segment")
        st.session_state.pop("seg_result", None)
        st.session_state.seg_result = _run_segmentation(raw_bytes, sam_prompt)
        st.session_state.sam_ts = _now_stamp()
        masks = st.session_state.seg_result.masks_np
        n = masks.shape[0]
        if n > 0:
            areas = (masks > 0.5).reshape(n, -1).sum(axis=1)
            st.session_state.selected_idx = [int(np.argmax(areas))]
        else:
            st.session_state.selected_idx = []
        for j in range(n):
            st.session_state[f"fur_cb_{img_sig}_{j}"] = j in st.session_state.selected_idx
with c_hint:
    st.caption(
        "SAM3 가 같은 prompt 로 잡은 모든 인스턴스를 보여줍니다. "
        "원하는 가구만 클릭하거나 체크박스로 선택하세요."
    )

if "seg_result" not in st.session_state:
    st.info("위 **SAM3 실행** 버튼을 눌러 가구 마스크를 검출하세요.")
    st.stop()

seg_result = st.session_state.seg_result
base_image = seg_result.image_rgb
masks = seg_result.masks_np
scores = seg_result.scores_np
n_inst = masks.shape[0]

if n_inst == 0:
    st.warning("SAM3 가 검출한 인스턴스가 없습니다. prompt 를 바꿔 다시 실행해 보세요.")
    st.stop()

debug_overlay = visualize_instance_masks(base_image, masks, scores_np=scores)
debug_for_click = resize_max_side(debug_overlay, MAX_CLICK_SIDE)

from streamlit_image_coordinates import streamlit_image_coordinates  # noqa: E402

list_c, click_c, prev_c = st.columns([0.85, 1.15, 1.0])

with list_c:
    st.markdown("**인스턴스 리스트**")
    st.caption(f"검출 {n_inst}개. 체크 또는 오른쪽 이미지 클릭으로 토글.")
    with st.container(height=420):
        for i in range(n_inst):
            area_frac = float((masks[i].squeeze() > 0.5).mean())
            st.checkbox(
                f"#{i}   score={float(scores[i]):.2f}   area={area_frac:.2f}",
                key=f"fur_cb_{img_sig}_{i}",
            )
    selected_now = [i for i in range(n_inst) if st.session_state.get(f"fur_cb_{img_sig}_{i}", False)]
    if selected_now != st.session_state.get("selected_idx", []):
        st.session_state.selected_idx = sorted(set(selected_now))
        _clear("selection")

with click_c:
    st.markdown("**클릭으로 선택 (디버그 오버레이)**")
    st.caption(
        "각 인스턴스 중앙에 표시된 번호가 #N 체크박스와 일치합니다. "
        "클릭하면 그 자리에 있는 마스크가 토글됩니다."
    )
    click = streamlit_image_coordinates(
        debug_for_click,
        key=f"clk_{img_sig}",
        use_column_width=True,
        cursor="crosshair",
    )
    if (
        click
        and isinstance(click, dict)
        and "unix_time" in click
        and st.session_state.get("_last_click_ut") != click["unix_time"]
    ):
        st.session_state._last_click_ut = click["unix_time"]
        ox, oy = _click_to_original(click, base_image.width, base_image.height)
        hit = _pick_mask_at(masks, scores, ox, oy)
        if hit is not None:
            cur = list(st.session_state.get("selected_idx", []))
            if hit in cur:
                cur.remove(hit)
            else:
                cur.append(hit)
            cur = sorted(set(cur))
            st.session_state.selected_idx = cur
            for j in range(n_inst):
                st.session_state[f"fur_cb_{img_sig}_{j}"] = j in cur
            _clear("selection")
            st.rerun()

with prev_c:
    st.markdown("**선택된 가구 cutout**")
    selected = list(st.session_state.get("selected_idx", []))
    if not selected:
        st.warning("최소 한 개의 인스턴스를 선택해 주세요.")
        st.session_state.pop("cutout_rgba", None)
    else:
        protect = _union_protect(masks, selected)
        cutout = apply_alpha_mask(base_image, protect)
        st.session_state.cutout_rgba = cutout
        st.image(cutout, width="stretch")
        st.caption(f"선택 인덱스: {selected}  (alpha=가구)")
        sam_ts = st.session_state.get("sam_ts") or _now_stamp()
        st.session_state.sam_ts = sam_ts
        cutout_path = OUTPUT_DIR / f"01_cutout_{sam_ts}.png"
        cutout.save(cutout_path)
        st.caption(f"저장: `{cutout_path.name}`")

if "cutout_rgba" not in st.session_state:
    st.stop()


# =====================================================================
# Step 2 — 스타일 추천
# =====================================================================
st.divider()
st.header("Step 2 — GPT-4o 스타일 추천 (2개)")

if st.button("스타일 추천 받기", type="primary"):
    _clear("style")
    with st.spinner("GPT-4o 호출 중…"):
        rec = StyleRecommender()
        rec_result = rec(st.session_state.cutout_rgba)
    st.session_state.style_recs = [
        {"name": s.name, "prompt": s.prompt} for s in rec_result.styles
    ]
    st.session_state.chosen_style_idx = 0
    st.session_state.chosen_style_prompt = st.session_state.style_recs[0]["prompt"]

if "style_recs" not in st.session_state:
    st.info("위 버튼을 눌러 가구에 어울리는 스타일 2개를 받아오세요.")
    st.stop()

recs = st.session_state.style_recs
names = [f"[{i}] {s['name']}" for i, s in enumerate(recs)]
prev_idx = st.session_state.get("chosen_style_idx", 0)
choice = st.radio(
    "사용할 스타일",
    options=list(range(len(recs))),
    format_func=lambda i: names[i],
    index=prev_idx,
    horizontal=True,
)
if choice != prev_idx:
    st.session_state.chosen_style_idx = choice
    st.session_state.chosen_style_prompt = recs[choice]["prompt"]
    _clear("style")

st.markdown(f"**선택: {recs[choice]['name']}**")
edited = st.text_area(
    "스타일 prompt (필요 시 수정)",
    value=st.session_state.get("chosen_style_prompt", recs[choice]["prompt"]),
    height=180,
)
if edited != st.session_state.get("chosen_style_prompt"):
    st.session_state.chosen_style_prompt = edited
    _clear("style")

with st.expander("두 추천 비교 보기"):
    for i, s in enumerate(recs):
        st.markdown(f"**[{i}] {s['name']}**")
        st.write(s["prompt"])


# =====================================================================
# Step 3 — 콘텐츠 용도별 배경 생성
# =====================================================================
st.divider()
st.header("Step 3 — 콘텐츠 용도별 배경 생성 (gpt-image-1)")
st.caption(
    "공통 hard constraint(가구 보존 + 전체 relighting) 위에 각 용도의 클래스가 "
    "컴포지션·비율 조항을 append 합니다. 여러 용도를 동시에 체크하면 그만큼 "
    "OpenAI 호출이 늘어납니다."
)

USE_CASE_CHOICES: list[tuple[str, str, type[BaseBackgroundGenerator]]] = [
    ("instagram_feed", "인스타그램 피드 (1:1)", InstagramFeedBackgroundGenerator),
    ("ohou_thumbnail", "오늘의집 썸네일 (1:1)", OhouThumbnailBackgroundGenerator),
    ("shorts",         "숏폼 세로 (2:3)",       ShortsBackgroundGenerator),
]
USE_CASE_LABEL = {k: lbl for k, lbl, _ in USE_CASE_CHOICES}
USE_CASE_CLASS = {k: cls for k, _, cls in USE_CASE_CHOICES}

selected_uses = st.multiselect(
    "콘텐츠 용도 (여러 개 선택 가능)",
    options=[k for k, _, _ in USE_CASE_CHOICES],
    default=st.session_state.get("_uc_default", ["instagram_feed"]),
    format_func=lambda k: USE_CASE_LABEL[k],
)
st.session_state._uc_default = selected_uses or ["instagram_feed"]

extra_bg = st.text_input(
    "배경 생성 추가 hard rule (선택)",
    value="",
    placeholder="예: no visible logo, no people, keep wall color neutral",
)

if st.button(
    "배경 생성",
    type="primary",
    disabled=(not selected_uses) or "cutout_rgba" not in st.session_state,
):
    _clear("background")
    style_name = recs[st.session_state.chosen_style_idx]["name"]
    style_slug = _slug(style_name)
    bg_ts = _now_stamp()
    st.session_state.bg_ts = bg_ts
    st.session_state.bg_style_slug = style_slug
    bg_results: dict[str, Image.Image] = {}
    bg_files: dict[str, Path] = {}
    for uc in selected_uses:
        cls = USE_CASE_CLASS[uc]
        with st.spinner(f"{USE_CASE_LABEL[uc]} 생성 중…"):
            gen = cls()
            r = gen(
                st.session_state.cutout_rgba,
                st.session_state.chosen_style_prompt,
                extra_system_instruction=extra_bg.strip() or None,
            )
        out_path = OUTPUT_DIR / f"03_styled__{uc}__{style_slug}__{bg_ts}.png"
        r.image.save(out_path)
        bg_results[uc] = r.image
        bg_files[uc] = out_path
    st.session_state.bg_results = bg_results
    st.session_state.bg_files = bg_files
    first = selected_uses[0]
    st.session_state.chosen_use_case = first
    st.session_state.styled_image = bg_results[first]

if "bg_results" in st.session_state and st.session_state.bg_results:
    bg_results = st.session_state.bg_results
    bg_files = st.session_state.get("bg_files", {})
    order = list(bg_results.keys())
    cols = st.columns(len(order))
    for col, uc in zip(cols, order):
        with col:
            img = bg_results[uc]
            st.image(img, width="stretch")
            st.caption(f"**{USE_CASE_LABEL[uc]}** · {img.size[0]}x{img.size[1]}")
            if uc == "shorts":
                st.caption("→ Step 4·5 로 영상 생성 진행")
            else:
                st.caption("→ Step 3 산출물이 최종 (인스타/오늘의집)")
            f = bg_files.get(uc)
            if f:
                st.caption(f"파일: `{f.name}`")
    cur = st.session_state.get("chosen_use_case", order[0])
    if cur not in order:
        cur = order[0]
    chosen = st.radio(
        "다음 단계로 진행할 결과 (숏폼만 Step 4·5 활성)",
        options=order,
        format_func=lambda k: USE_CASE_LABEL[k],
        index=order.index(cur),
        horizontal=True,
    )
    if chosen != st.session_state.get("chosen_use_case"):
        st.session_state.chosen_use_case = chosen
        st.session_state.styled_image = bg_results[chosen]
        _clear("background_choice")
    else:
        st.session_state.styled_image = bg_results[chosen]


# =====================================================================
# Step 4·5 는 **shorts** 결과로 선택했을 때만 표시
# (인스타·오늘의집은 Step 3 산출물이 최종이라 여기서 마무리)
# =====================================================================
chosen_uc_for_video = st.session_state.get("chosen_use_case")
if chosen_uc_for_video != "shorts":
    st.divider()
    if chosen_uc_for_video:
        st.info(
            f"현재 선택된 결과 **{USE_CASE_LABEL.get(chosen_uc_for_video, chosen_uc_for_video)}** 는 "
            "Step 3 이미지가 최종 산출물입니다. 영상이 필요하면 위 라디오에서 "
            "**숏폼 세로 (2:3)** 를 선택해 주세요."
        )
    st.stop()


# =====================================================================
# Step 4 — 샷 플랜 (shorts 전용)
# =====================================================================
st.divider()
st.header("Step 4 — 영상용 샷 플랜 생성 (GPT-4o)")
st.caption(
    "Step 3 의 shorts 이미지를 GPT-4o vision이 직접 보고, Sora 가 따라갈 카메라 무빙/샷 "
    "순서 텍스트를 만듭니다. 이 단계에서는 새 이미지를 생성하지 않습니다."
)

extra_shot = st.text_input(
    "샷 플랜 추가 제약 (선택)",
    value="",
    placeholder="예: slower camera, emphasize wood texture, no hands",
)

if st.button("샷 플랜 생성", type="primary", disabled="styled_image" not in st.session_state):
    _clear("shot_plan")
    style_slug = st.session_state.get("bg_style_slug") or "untitled"
    ts = _now_stamp()
    out_path = OUTPUT_DIR / f"04_shot_plan__{style_slug}__{ts}.txt"
    with st.spinner("GPT-4o vision 호출 중…"):
        planner = FurnitureShotPlanGenerator()
        plan_result = planner(
            st.session_state.styled_image,
            extra_instruction=extra_shot.strip() or None,
            output_path=out_path,
        )
    st.session_state.shot_plan = plan_result.shot_plan
    st.session_state.shot_plan_editor = plan_result.shot_plan
    st.session_state.shot_plan_file = out_path

if "shot_plan" in st.session_state:
    st.text_area(
        "Step 4 결과 — Sora용 shot plan",
        value=st.session_state.shot_plan,
        height=260,
        key="shot_plan_editor",
    )
    sp_file = st.session_state.get("shot_plan_file")
    if sp_file:
        st.caption(f"파일: `{Path(sp_file).name}`")
    if st.session_state.shot_plan_editor != st.session_state.shot_plan:
        st.session_state.shot_plan = st.session_state.shot_plan_editor
        _clear("shot_plan")


# =====================================================================
# Step 5 — Sora 비디오 (shorts 전용)
# =====================================================================
st.divider()
st.header("Step 5 — Sora 2 비디오 생성")
st.caption(
    "Step 3 의 shorts 이미지를 Sora 첫 프레임 reference로 넣고, Step 4 shot plan 을 "
    "최종 prompt 에 반영합니다."
)

c1, c2, c3 = st.columns(3)
with c1:
    v_model = st.selectbox("모델", ["sora-2", "sora-2-pro"], index=0)
with c2:
    v_size = st.selectbox(
        "해상도",
        ["720x1280", "1080x1920", "1280x720", "1792x1024", "1024x1792", "1920x1080"],
        index=0,
        help="숏폼 영상이므로 세로 해상도를 기본값으로 두었습니다.",
    )
with c3:
    v_secs = st.selectbox(
        "길이(초)",
        ["4", "8", "12"],
        index=1,
        help=(
            "한 take 안에서 여러 slow 카메라 무브가 blend 되는 시네마틱 "
            "결과는 8초 정도가 적절합니다. 4초는 카메라 모션이 단조로워질 "
            "수 있고, 12초는 가구 정체성 drift 와 호출당 비용(8초 ~$2 기준 "
            "약 1.5배)이 함께 커집니다."
        ),
    )

extra_llm = st.text_input(
    "LLM 추가 제약 (선택)",
    value="",
    placeholder="예: no people, slow cinematic dolly only",
)

if st.button(
    "Sora 비디오 생성",
    type="primary",
    disabled=("styled_image" not in st.session_state or "shot_plan" not in st.session_state),
):
    _clear("video")
    progress = st.progress(0, text="initializing…")
    status_box = st.empty()

    def _cb(status: str, pct: int) -> None:
        progress.progress(min(100, max(0, pct)), text=f"sora {status}: {pct}%")
        status_box.write(f"`{time.strftime('%H:%M:%S')}` sora **{status}** — {pct}%")

    style_slug = st.session_state.get("bg_style_slug") or "untitled"
    ts = _now_stamp()
    out_path = OUTPUT_DIR / f"05_video__{style_slug}__{ts}.mp4"

    try:
        sv = SoraReferenceVideo(
            video_model=v_model,
            size=v_size,
            seconds=v_secs,
        )
        result = sv(
            st.session_state.styled_image,
            st.session_state.shot_plan,
            output_path=out_path,
            extra_llm_instruction=extra_llm.strip() or None,
            on_progress=_cb,
        )
    except Exception as e:  # noqa: BLE001
        st.exception(e)
    else:
        st.session_state.video_path = result.output_path
        st.session_state.video_prompt = result.video_prompt
        progress.progress(100, text="done")

if "video_path" in st.session_state:
    p = Path(st.session_state.video_path)
    if p.exists():
        st.success(f"완료: `{p.name}`")
        st.video(str(p))
        with st.expander("실제로 Sora 에 보낸 prompt"):
            st.code(st.session_state.video_prompt, language="text")
    else:
        st.warning(f"비디오 파일이 없습니다: {p}")
