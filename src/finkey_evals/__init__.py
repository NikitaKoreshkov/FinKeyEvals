# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FinKeyEvals — dependency-light evaluation for LLM apps.

Components:
  evaluator     — score every response: context_relevance, faithfulness, groundedness
  quality_gate  — optional blocking gate (re-gen if score < threshold)
  feedback      — store thumbs-up/down + reasons, feed the low-quality queue
  harness       — CI runner: JSON case files, RAG/agent/red-team metrics
  judge         — optional OpenAI-compatible LLM judge for deep scoring
"""
from .evaluator import EvalResult, ResponseEvaluator
from .feedback import FeedbackStore, FeedbackEntry
from .harness import (
    EvalCase,
    HarnessReport,
    default_smoke_suite,
    load_cases_json,
    run_harness,
    run_live_harness,
)
from .judge import build_judge_from_env
from .quality_gate import QualityGate

__all__ = [
    "EvalResult",
    "ResponseEvaluator",
    "QualityGate",
    "FeedbackStore",
    "FeedbackEntry",
    "EvalCase",
    "HarnessReport",
    "run_harness",
    "run_live_harness",
    "load_cases_json",
    "default_smoke_suite",
    "build_judge_from_env",
]
