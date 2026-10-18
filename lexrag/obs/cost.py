"""Token-based cost estimation.

You cannot optimize what you don't measure. Tracking tokens + dollars per query
is table stakes for a production RAG system: it's how you catch a prompt that
ballooned, justify a caching layer, or decide between models. Prices are USD per
1M tokens (update as vendor pricing changes).
"""

from __future__ import annotations

# USD per 1,000,000 tokens. Keep this in one place so pricing is auditable.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    price = PRICING.get(model)
    if price is None:
        return 0.0
    return (
        input_tokens / 1_000_000 * price["input"]
        + output_tokens / 1_000_000 * price["output"]
    )
