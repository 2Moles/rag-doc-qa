"""Retrieval evaluation: hit rate@k and mean reciprocal rank on a labelled question set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .retrieval import HybridIndex


@dataclass(frozen=True)
class Example:
    question: str
    source: str
    answer_contains: str


@dataclass(frozen=True)
class Scores:
    mode: str
    k: int
    hit_rate: float  # share of questions where a top-k chunk contains the answer text
    mrr: float       # mean of 1/rank of the first chunk containing the answer (0 if none)


def load_examples(path: Path) -> list[Example]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Example(**json.loads(line)) for line in lines if line.strip()]


def evaluate(index: HybridIndex, examples: list[Example], k: int = 3, mode: str = "hybrid") -> Scores:
    hits_total, rr_total = 0, 0.0
    for ex in examples:
        results = index.search(ex.question, k=k, mode=mode)
        for rank, hit in enumerate(results, start=1):
            # A retrieval counts only if it is the right document AND contains the answer.
            if hit.chunk.source == ex.source and ex.answer_contains.lower() in hit.chunk.text.lower():
                hits_total += 1
                rr_total += 1 / rank
                break
    n = len(examples)
    return Scores(mode, k, hits_total / n, rr_total / n)
