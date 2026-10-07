# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
FeedbackStore — persists 👍👎 signals and links them to EvalResults.

Storage backend: PostgreSQL via async psycopg2/asyncpg.
Falls back to in-memory ring buffer when DB is unavailable.

Schema (applied from ``schema.sql``):
    feedback(
        id UUID PRIMARY KEY,
        company_id TEXT,
        session_id TEXT,
        turn_id TEXT,
        question TEXT,
        answer TEXT,
        rating SMALLINT,       -- 1=👍 -1=👎 0=neutral
        reason TEXT,
        eval_context_relevance FLOAT,
        eval_faithfulness FLOAT,
        eval_groundedness FLOAT,
        eval_overall FLOAT,
        created_at TIMESTAMPTZ DEFAULT now()
    )
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .evaluator import EvalResult

logger = logging.getLogger("finkey.eval.feedback")


@dataclass
class FeedbackEntry:
    company_id: str
    session_id: str
    turn_id: str
    question: str
    answer: str
    rating: int
    reason: str = ""
    eval_result: Optional[EvalResult] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "company_id": self.company_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "question": self.question,
            "answer": self.answer,
            "rating": self.rating,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }
        if self.eval_result:
            d["eval_context_relevance"]   = self.eval_result.context_relevance
            d["eval_faithfulness"]        = self.eval_result.answer_faithfulness
            d["eval_groundedness"]        = self.eval_result.answer_groundedness
            d["eval_overall"]             = self.eval_result.overall
        return d



class FeedbackStore:
    """
    Async feedback store with an in-memory ring buffer fallback.

    Parameters
    ----------
    db_pool : asyncpg.Pool or None
        If provided, entries are persisted to PostgreSQL.
    buffer_size : int
        In-memory ring buffer capacity (fallback when DB is unavailable).
    """

    def __init__(self, db_pool=None, buffer_size: int = 500) -> None:
        self._pool = db_pool
        self._buffer: collections.deque[FeedbackEntry] = collections.deque(maxlen=buffer_size)
        self._lock = asyncio.Lock()


    async def record(self, entry: FeedbackEntry) -> None:
        async with self._lock:
            self._buffer.append(entry)

        if self._pool:
            await self._persist(entry)
        else:
            logger.debug("FeedbackStore: buffered (no DB) — %s rating=%d", entry.turn_id, entry.rating)

    async def _persist(self, entry: FeedbackEntry) -> None:
        d = entry.to_dict()
        sql = """
            INSERT INTO feedback (
                id, company_id, session_id, turn_id, question, answer, rating, reason,
                eval_context_relevance, eval_faithfulness, eval_groundedness, eval_overall,
                created_at
            ) VALUES (
                $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13
            ) ON CONFLICT (id) DO NOTHING
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    sql,
                    d["id"], d["company_id"], d["session_id"], d["turn_id"],
                    d["question"], d["answer"], d["rating"], d.get("reason", ""),
                    d.get("eval_context_relevance"), d.get("eval_faithfulness"),
                    d.get("eval_groundedness"), d.get("eval_overall"),
                    entry.created_at,
                )
        except Exception as exc:
            logger.error("FeedbackStore: DB write failed: %s", exc)


    async def list_by_company(
        self,
        company_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        if self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(
                        """SELECT * FROM feedback
                           WHERE company_id=$1
                           ORDER BY created_at DESC
                           LIMIT $2 OFFSET $3""",
                        company_id, limit, offset,
                    )
                return [dict(r) for r in rows]
            except Exception as exc:
                logger.error("FeedbackStore: DB read failed: %s", exc)

        return [
            e.to_dict()
            for e in reversed(self._buffer)
            if e.company_id == company_id
        ][:limit]

    async def score_summary(self, company_id: str) -> dict:
        """Return thumbs-up/down ratio and avg eval scores for a company."""
        entries = await self.list_by_company(company_id, limit=1000)
        if not entries:
            return {"thumbs_up": 0, "thumbs_down": 0, "avg_overall": None}

        up   = sum(1 for e in entries if e.get("rating", 0) == 1)
        down = sum(1 for e in entries if e.get("rating", 0) == -1)
        scores = [e["eval_overall"] for e in entries if e.get("eval_overall") is not None]
        return {
            "thumbs_up":   up,
            "thumbs_down": down,
            "total":       len(entries),
            "avg_overall": round(sum(scores) / len(scores), 3) if scores else None,
        }


    async def low_quality_queue(
        self,
        company_id: str,
        max_score: float = 0.4,
        limit: int = 50,
    ) -> list[dict]:
        """
        Returns bad-rated turns that need human review / fine-tuning.
        Sorted by eval_overall ASC (worst first).
        """
        entries = await self.list_by_company(company_id, limit=1000)
        bad = [
            e for e in entries
            if (e.get("rating", 0) == -1 or
                (e.get("eval_overall") is not None and e["eval_overall"] < max_score))
        ]
        bad.sort(key=lambda e: e.get("eval_overall") or 0)
        return bad[:limit]
