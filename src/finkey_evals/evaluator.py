# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
ResponseEvaluator — automatic scoring for every RAG-augmented response.

Scores (0.0 – 1.0):
  context_relevance   — are the retrieved chunks relevant to the question?
  answer_faithfulness — does the answer stay within the provided context?
  answer_groundedness — is each claim in the answer supported by a source chunk?
  overall             — weighted average (configurable)

Two evaluation modes:
  1. heuristic  — fast (~1ms), regex + overlap, no LLM calls. Always runs.
  2. llm        — deeper (~2-5s), optional. Enabled when FINKEY_LLM_EVAL=1.

The heuristic mode gives a good-enough signal for production logging.
LLM mode is used when you need high-confidence gating (quality_gate.py).
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger("finkey.eval.evaluator")

_STOPWORDS = frozenset(
    "и или но а что как то это из за на в по с к от до же ли бы "
    "the a an is are was were be been being have has had do does did "
    "will would should could may might must shall".split()
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"\b[а-яёa-z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _coverage(answer: str, chunks: list[str]) -> float:
    """What fraction of answer n-grams are covered by any chunk."""
    if not chunks or not answer:
        return 0.0
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    covered = sum(1 for t in ans_tokens if any(t in _tokens(c) for c in chunks))
    return covered / len(ans_tokens)



@dataclass
class EvalResult:
    context_relevance: float   = 0.0
    answer_faithfulness: float = 0.0
    answer_groundedness: float = 0.0
    overall: float             = 0.0
    mode: str                  = "heuristic"
    latency_ms: float          = 0.0
    metadata: dict             = field(default_factory=dict)

    def passed(self, threshold: float = 0.45) -> bool:
        return self.overall >= threshold

    def to_dict(self) -> dict:
        return {
            "context_relevance": round(self.context_relevance, 3),
            "answer_faithfulness": round(self.answer_faithfulness, 3),
            "answer_groundedness": round(self.answer_groundedness, 3),
            "overall": round(self.overall, 3),
            "mode": self.mode,
            "latency_ms": round(self.latency_ms, 1),
            **self.metadata,
        }



class ResponseEvaluator:
    """
    Stateless evaluator. Instantiate once per gateway/router.

    Parameters
    ----------
    generate_fn : callable(messages) → str
        The LLM call to use for deep (LLM-mode) evaluation.
        Required only when llm_eval=True is passed to evaluate().
    weights : dict
        Override default score weights.
    """

    _DEFAULT_WEIGHTS = {
        "context_relevance":   0.30,
        "answer_faithfulness": 0.45,
        "answer_groundedness": 0.25,
    }

    def __init__(
        self,
        generate_fn: Optional[Callable] = None,
        weights: Optional[dict] = None,
    ) -> None:
        self._generate_fn = generate_fn
        self._weights = {**self._DEFAULT_WEIGHTS, **(weights or {})}


    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        context_chunks: Sequence[str],
        llm_eval: bool = False,
    ) -> EvalResult:
        """
        Score a (question, answer, context) triple.

        Set ``llm_eval=True`` only when you need high-confidence scoring
        (e.g. quality gate) — it adds a single LLM call.
        """
        t0 = time.monotonic()
        result = self._heuristic(question=question, answer=answer,
                                  context_chunks=list(context_chunks))
        if llm_eval and self._generate_fn:
            result = self._llm_score(question=question, answer=answer,
                                      context_chunks=list(context_chunks),
                                      heuristic=result)
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result


    def _heuristic(self, *, question: str, answer: str, context_chunks: list[str]) -> EvalResult:
        q_tok = _tokens(question)
        a_tok = _tokens(answer)

        if context_chunks:
            chunk_scores = [_jaccard(q_tok, _tokens(c)) for c in context_chunks]
            ctx_rel = min(1.0, max(chunk_scores) * 3.5)
        else:
            ctx_rel = 0.0

        if context_chunks:
            faithfulness = _coverage(answer, context_chunks)
            combined_len = sum(len(c) for c in context_chunks)
            if len(answer) > combined_len * 1.5 and combined_len > 100:
                faithfulness *= 0.75
        else:
            hedges = re.findall(
                r"\b(не знаю|не могу|нет данных|недостаточно|i don.t know|no information|cannot)\b",
                answer.lower(),
            )
            faithfulness = 0.8 if hedges else 0.3

        groundedness = self._heuristic_groundedness(answer=answer,
                                                      context_chunks=context_chunks)

        overall = (
            ctx_rel         * self._weights["context_relevance"]
            + faithfulness  * self._weights["answer_faithfulness"]
            + groundedness  * self._weights["answer_groundedness"]
        )
        return EvalResult(
            context_relevance=ctx_rel,
            answer_faithfulness=faithfulness,
            answer_groundedness=groundedness,
            overall=min(1.0, overall),
            mode="heuristic",
        )

    def _heuristic_groundedness(self, *, answer: str, context_chunks: list[str]) -> float:
        """Detect likely hallucination patterns in the answer."""
        if not context_chunks:
            return 0.4

        nums_in_answer = set(re.findall(r"\b\d[\d.,]*\b", answer))
        if not nums_in_answer:
            return 0.85

        combined_ctx = " ".join(context_chunks)
        nums_in_ctx = set(re.findall(r"\b\d[\d.,]*\b", combined_ctx))
        unsupported = nums_in_answer - nums_in_ctx
        if not unsupported:
            return 0.95
        ratio_bad = len(unsupported) / len(nums_in_answer)
        return max(0.2, 1.0 - ratio_bad * 0.8)


    def _llm_score(
        self,
        *,
        question: str,
        answer: str,
        context_chunks: list[str],
        heuristic: EvalResult,
    ) -> EvalResult:
        ctx_text = "\n---\n".join(context_chunks[:6])
        prompt = (
            f"Rate the quality of this AI answer on a scale 0.0–1.0 for each metric.\n\n"
            f"QUESTION: {question}\n\n"
            f"CONTEXT (retrieved chunks):\n{ctx_text}\n\n"
            f"ANSWER: {answer}\n\n"
            f"Respond ONLY with JSON, no explanation:\n"
            f'{{"context_relevance": 0.0, "answer_faithfulness": 0.0, "answer_groundedness": 0.0}}'
        )
        try:
            raw = self._generate_fn([{"role": "user", "content": prompt}])
            import json
            m = re.search(r'\{[^{}]+\}', raw)
            if not m:
                return heuristic
            scores = json.loads(m.group())
            ctx_rel     = float(scores.get("context_relevance", heuristic.context_relevance))
            faithfulness = float(scores.get("answer_faithfulness", heuristic.answer_faithfulness))
            groundedness = float(scores.get("answer_groundedness", heuristic.answer_groundedness))
            overall = (
                ctx_rel      * self._weights["context_relevance"]
                + faithfulness * self._weights["answer_faithfulness"]
                + groundedness * self._weights["answer_groundedness"]
            )
            return EvalResult(
                context_relevance=max(0.0, min(1.0, ctx_rel)),
                answer_faithfulness=max(0.0, min(1.0, faithfulness)),
                answer_groundedness=max(0.0, min(1.0, groundedness)),
                overall=max(0.0, min(1.0, overall)),
                mode="llm",
            )
        except Exception as exc:
            logger.warning("LLM eval failed, using heuristic: %s", exc)
            return heuristic
