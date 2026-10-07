# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
QualityGate — block or re-generate responses that score below threshold.

Usage in the router:
    gate = QualityGate(evaluator, threshold=0.45, max_retries=2)

    async for token in gate.stream_with_gate(
        generate_fn=...,
        question=...,
        context_chunks=...,
    ):
        yield token
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Callable, Sequence

from .evaluator import EvalResult, ResponseEvaluator

logger = logging.getLogger("finkey.eval.quality_gate")


class QualityGate:
    """
    Wraps a streaming generator. After collecting the full response,
    evaluates quality. If below threshold and retries remain, regenerates.

    Parameters
    ----------
    evaluator : ResponseEvaluator
    threshold : float  — minimum acceptable overall score (0.0–1.0)
    max_retries : int  — how many times to try before giving up
    use_llm_eval : bool — use expensive LLM scoring for gating
    on_low_quality : callable(result) → None  — hook for alerting / logging
    """

    def __init__(
        self,
        evaluator: ResponseEvaluator,
        threshold: float = 0.45,
        max_retries: int = 2,
        use_llm_eval: bool = False,
        on_low_quality: Callable[[EvalResult], None] | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._threshold = threshold
        self._max_retries = max_retries
        self._use_llm_eval = use_llm_eval
        self._on_low_quality = on_low_quality


    async def run(
        self,
        *,
        generate_fn: Callable[[], AsyncIterator[str]],
        question: str,
        context_chunks: Sequence[str],
    ) -> tuple[str, EvalResult]:
        """
        Collect full response, evaluate, retry if needed.
        Returns (final_answer, final_eval_result).
        """
        attempt = 0
        while attempt <= self._max_retries:
            answer = await _collect(generate_fn())
            result = self._evaluator.evaluate(
                question=question,
                answer=answer,
                context_chunks=context_chunks,
                llm_eval=self._use_llm_eval,
            )
            if result.passed(self._threshold):
                if attempt > 0:
                    logger.info("Quality gate: passed on attempt %d, score=%.3f", attempt + 1, result.overall)
                return answer, result

            logger.warning(
                "Quality gate: attempt %d failed (score=%.3f < %.3f)",
                attempt + 1, result.overall, self._threshold,
            )
            if self._on_low_quality:
                try:
                    self._on_low_quality(result)
                except Exception:
                    pass
            attempt += 1

        logger.error("Quality gate: exhausted retries, returning last response")
        return answer, result  # type: ignore[possibly-undefined]

    async def stream_with_gate(
        self,
        *,
        generate_fn: Callable[[], AsyncIterator[str]],
        question: str,
        context_chunks: Sequence[str],
        attach_score_header: bool = True,
    ) -> AsyncIterator[str]:
        """
        Stream-compatible wrapper:
        1. Silently collect & evaluate.
        2. If passes → stream the buffered answer token by token.
        3. If fails (after retries) → stream the best-effort answer with a warning header.
        """
        answer, result = await self.run(
            generate_fn=generate_fn,
            question=question,
            context_chunks=context_chunks,
        )
        if attach_score_header:
            import json
            header = json.dumps({"__eval": result.to_dict()})
            yield f"data: {header}\n\n"

        for word in answer.split(" "):
            yield word + " "
            await asyncio.sleep(0)



async def _collect(it: AsyncIterator[str]) -> str:
    chunks = []
    async for chunk in it:
        chunks.append(chunk)
    return "".join(chunks)
