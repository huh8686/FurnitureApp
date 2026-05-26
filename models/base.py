"""파이프라인 step 공통 베이스."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class PipelineStep(ABC):
    """Step 1~5 공통 인터페이스 (run() 만 구현하면 됨)."""

    name: str = "step"

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def require_openai_api_key(api_key: Optional[str] = None) -> str:
    """env 또는 인자에서 OpenAI 키를 가져온다. 없으면 명확한 에러."""
    key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY 가 설정되어 있지 않습니다. "
            ".env 또는 환경변수를 설정하세요."
        )
    return key
