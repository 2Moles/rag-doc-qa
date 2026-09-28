"""Hybrid retrieval: BM25 (exact terms) + TF-IDF cosine (weighted n-grams), fused with RRF."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .documents import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:[.%][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it its of on or that the this to was what when where which "
    "who why will with do does must should can many much".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25, written out so the scoring is easy to read and test."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.avg_len = sum(self.lengths) / max(len(docs), 1)
        df = Counter(term for d in docs for term in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs))
        for i, (tf, length) in enumerate(zip(self.tfs, self.lengths, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            out[i] = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf
            )
        return out


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class HybridIndex:
    MODES = ("bm25", "tfidf", "hybrid")

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60):
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        self.chunks = chunks
        self.rrf_k = rrf_k
        texts = [f"{c.heading}. {c.text}" for c in chunks]  # heading words help short chunks match
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)

    def _ranked(self, scores: np.ndarray) -> list[int]:
        # Stable sort so ties keep document order, which makes results reproducible.
        return [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0]

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Hit]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        bm25_scores = self.bm25.scores(tokenize(query))
        tfidf_scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        if mode == "bm25":
            order, scores = self._ranked(bm25_scores), bm25_scores
        elif mode == "tfidf":
            order, scores = self._ranked(tfidf_scores), tfidf_scores
        else:
            # Reciprocal Rank Fusion: combine by rank, so the two score scales never need calibrating.
            fused = np.zeros(len(self.chunks))
            for ranking in (self._ranked(bm25_scores), self._ranked(tfidf_scores)):
                for rank, i in enumerate(ranking):
                    fused[i] += 1.0 / (self.rrf_k + rank + 1)
            order, scores = self._ranked(fused), fused
        return [Hit(self.chunks[i], float(scores[i])) for i in order[:k]]
