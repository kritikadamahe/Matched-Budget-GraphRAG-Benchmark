"""
Cost model for extraction (indexing) and answering (query time).

Calibrated on the real extraction calls of the 75-question benchmark run: tokens per chunk
are fitted as  tokens = base + per_word * words  (least squares over 759 calls). The fixed
estimate in extraction/pricing.py assumes 250 output tokens per chunk; the measured mean was
~730, so uncalibrated estimates come out about 3x too low. A pilot on the user's own
documents re-measures these numbers and replaces the calibration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from extraction.pricing import ASSUMED_OUTPUT_TOKENS_PER_CHUNK, estimate_tokens

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "results" / "experiments" / "benchmark_75q_hotpotqa"
CORPUS = ROOT / "data" / "corpus" / "hotpotqa_huggingface_75q_seed42_w250o40"
FULL_CONDITION = "hotpotqa__random__b100__seed42__rs0"


@dataclass
class Calibration:
    in_base: float
    in_per_word: float
    out_base: float
    out_per_word: float
    query_in: float              # answer call: input tokens per question (700-word context)
    query_out: float
    n_calls: int
    source: str

    def chunk_tokens(self, words: float) -> tuple[float, float]:
        return self.in_base + self.in_per_word * words, max(50.0, self.out_base + self.out_per_word * words)


# Used when the benchmark files are missing (e.g. a fresh clone): the fitted values, rounded.
FALLBACK = Calibration(in_base=493, in_per_word=1.39, out_base=263, out_per_word=5.14,
                       query_in=1300, query_out=75, n_calls=0, source="built-in defaults (benchmark files not found)")


@lru_cache(maxsize=1)
def benchmark_calibration() -> Calibration:
    try:
        words = {c["chunk_id"]: len(c["text"].split()) for c in json.loads((CORPUS / "chunks.json").read_text())}
        rows = [json.loads(line) for line in (RUN / "conditions" / FULL_CONDITION / "extraction" / "extractions.jsonl").open()]
        rows = [r for r in rows if r.get("usage") and r["chunk_id"] in words]
        w = np.array([words[r["chunk_id"]] for r in rows], dtype=float)
        X = np.c_[np.ones_like(w), w]
        (ib, iw), *_ = np.linalg.lstsq(X, np.array([r["usage"]["input_tokens"] for r in rows], float), rcond=None)
        (ob, ow), *_ = np.linalg.lstsq(X, np.array([r["usage"]["output_tokens"] for r in rows], float), rcond=None)
        ev = json.loads((RUN / "conditions" / "hotpotqa__ketrag__b100__seed42" / "eval_summary.json").read_text())
        gen = ev["cost"]["generation"]
        return Calibration(round(float(ib), 1), round(float(iw), 3), round(float(ob), 1), round(float(ow), 3),
                           round(gen["input_tokens"] / gen["n"]), round(gen["output_tokens"] / gen["n"]),
                           len(rows), f"{len(rows)} real extraction calls (qwen-3.8-27b, benchmark run)")
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        return FALLBACK


def from_measurements(chunk_words: list[int], in_tokens: list[int], out_tokens: list[int],
                      query_in: float, query_out: float, source: str) -> Calibration:
    """Calibration re-fitted on a pilot's own extraction calls (falls back to a ratio if all chunks are alike)."""
    w = np.asarray(chunk_words, float)
    X = np.c_[np.ones_like(w), w]
    if len(w) >= 10 and np.ptp(w) > 20:
        (ib, iw), *_ = np.linalg.lstsq(X, np.asarray(in_tokens, float), rcond=None)
        (ob, ow), *_ = np.linalg.lstsq(X, np.asarray(out_tokens, float), rcond=None)
    else:
        ib, ob = 0.0, 0.0
        iw, ow = sum(in_tokens) / max(1, w.sum()), sum(out_tokens) / max(1, w.sum())
    return Calibration(round(float(ib), 1), round(float(iw), 3), round(float(ob), 1), round(float(ow), 3),
                       round(query_in), round(query_out), len(w), source)


def usd(tokens_in: float, tokens_out: float, price_in: float, price_out: float) -> float:
    return tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out


def extraction_table(chunk_words: list[int], budgets: list[int], cal: Calibration,
                     price_in: float, price_out: float) -> list[dict]:
    """Projected indexing cost of extracting each budget share of the corpus (expected value:
    a budget share of the chunks, at the corpus' average chunk size)."""
    n = len(chunk_words)
    tin = sum(cal.chunk_tokens(w)[0] for w in chunk_words)
    tout = sum(cal.chunk_tokens(w)[1] for w in chunk_words)
    naive = sum(estimate_tokens(" ".join(["x"] * w)) for w in chunk_words), ASSUMED_OUTPUT_TOKENS_PER_CHUNK * n
    rows = []
    for b in budgets:
        k = max(1, int(b / 100 * n + 0.5))
        f = k / n
        rows.append({"budget_pct": b, "n_chunks": k, "input_tokens": round(float(tin * f)),
                     "output_tokens": round(float(tout * f)), "usd": round(float(usd(tin * f, tout * f, price_in, price_out)), 4),
                     "usd_uncalibrated": round(usd(naive[0] * f, naive[1] * f, price_in, price_out), 4)})
    return rows


def query_cost(n_questions: int, cal: Calibration, price_in: float, price_out: float) -> float:
    return round(usd(cal.query_in * n_questions, cal.query_out * n_questions, price_in, price_out), 6)


def calibration_dict(cal: Calibration) -> dict:
    return asdict(cal)
