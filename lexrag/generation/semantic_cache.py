"""Semantic answer cache.

An exact-string cache misses paraphrases: "how many vacation days roll over?"
and "what's the vacation carryover limit?" would both hit the full pipeline.
A *semantic* cache stores the query embedding and returns a cached answer when
a new query is within a cosine-similarity threshold of a cached one.

Tradeoff to discuss: too low a threshold returns stale/wrong answers for
merely-related questions (false hits); too high and you rarely hit. 0.95 is a
conservative default. Production concerns: TTL/invalidation when the corpus
changes, and per-user scoping so cached answers don't leak across permissions.
"""

from __future__ import annotations

import numpy as np

from lexrag.models import Answer


class SemanticCache:
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        self._vectors: list[np.ndarray] = []
        self._answers: list[Answer] = []

    @staticmethod
    def _normalize(vector: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        return arr / norm if norm else arr

    def get(self, query_vector: list[float]) -> Answer | None:
        if not self._vectors:
            return None
        q = self._normalize(query_vector)
        sims = [float(q @ v) for v in self._vectors]  # cosine (vectors normalized)
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold:
            return self._answers[best]
        return None

    def set(self, query_vector: list[float], answer: Answer) -> None:
        self._vectors.append(self._normalize(query_vector))
        self._answers.append(answer)
