# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
Optional LLM judge — one OpenAI-compatible chat call, no SDK tree.

Used wherever ``generate_fn`` is expected:
``ResponseEvaluator(generate_fn=build_judge_from_env())``.

Env:
  FINKEY_EVALS_JUDGE_BACKEND   openai | openrouter | ollama   (auto-detect)
  FINKEY_EVALS_JUDGE_MODEL     default gpt-4o-mini / equivalents
  OPENAI_API_KEY / OPENROUTER_API_KEY
  OLLAMA_HOST                  default http://localhost:11434
  FINKEY_EVALS_JUDGE_TIMEOUT_SEC  default 30
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("finkey_evals.judge")

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "ollama": "qwen2.5:7b",
}


def _post_chat(url: str, payload: dict, headers: dict, timeout: float) -> str:
    import requests

    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


class _Judge:
    def __init__(self, backend: str, model: str, api_key: str, base_url: str, timeout: float) -> None:
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    def __call__(self, messages: list[dict]) -> str:
        url = self.base_url or {
            "openai": _OPENAI_URL,
            "openrouter": _OPENROUTER_URL,
            "ollama": (os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/") + "/api/chat",
        }[self.backend]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "temperature": 0.0}
        if self.backend == "ollama":
            payload = {"model": self.model, "messages": messages, "stream": False}
            import requests

            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return (resp.json().get("message") or {}).get("content", "").strip()
        return _post_chat(url, payload, headers, self.timeout)


def build_judge_from_env() -> Optional[Callable[[list[dict]], str]]:
    """
    Auto-detect a judge backend from env. Returns None when nothing is
    configured — heuristic tiers then run with no LLM at all.
    """
    backend = (os.getenv("FINKEY_EVALS_JUDGE_BACKEND") or "").strip().lower()
    timeout = float(os.getenv("FINKEY_EVALS_JUDGE_TIMEOUT_SEC") or "30")

    if not backend:
        if os.getenv("OPENAI_API_KEY"):
            backend = "openai"
        elif os.getenv("OPENROUTER_API_KEY"):
            backend = "openrouter"
        else:
            return None

    key = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
        "ollama": "",
    }.get(backend, "")
    if backend not in ("openai", "openrouter", "ollama"):
        logger.warning("Unknown judge backend %r, judge disabled", backend)
        return None
    model = (os.getenv("FINKEY_EVALS_JUDGE_MODEL") or _DEFAULT_MODELS[backend]).strip()
    base_url = (os.getenv("FINKEY_EVALS_JUDGE_BASE_URL") or "").strip()
    judge = _Judge(backend, model, key, base_url, timeout)
    return lambda messages: judge(list(messages))
