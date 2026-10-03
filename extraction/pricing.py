"""
Token and cost arithmetic (Phase 4).

Real cost is always computed from the token counts the API REPORTS back
(authoritative). The helpers here are only used for two things:
  - turning reported token counts into dollars, using the prices in the config
    (never hardcoded - prices change; verify them on OpenAI's pricing page);
  - ESTIMATING cost before a run (--dry-run, spending cap), which needs a guess
    at token counts because nothing has been sent yet.

The estimate is deliberately simple and is labelled as an estimate everywhere:
about 1.33 tokens per English word, plus a fixed output size per chunk.
"""

from __future__ import annotations

import math

WORDS_TO_TOKENS = 1.33                 # rough English average
ASSUMED_OUTPUT_TOKENS_PER_CHUNK = 250  # rough size of the extracted JSON


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text.split()) * WORDS_TO_TOKENS)


def cost_usd(input_tokens: int, output_tokens: int,
             price_input_per_1m: float, price_output_per_1m: float) -> float:
    """Dollar cost of a call, at the given USD-per-1M-token prices."""
    return input_tokens * price_input_per_1m / 1_000_000 + output_tokens * price_output_per_1m / 1_000_000
