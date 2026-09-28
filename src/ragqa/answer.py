"""Answer generation: extractive by default, Claude when ANTHROPIC_API_KEY (or an ant profile) is set."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .documents import split_sentences
from .retrieval import Hit, tokenize

MODEL = os.environ.get("RAGQA_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.
Cite every fact with its source number in square brackets, like [2].
If the sources do not contain the answer, say that the documents do not cover it; do not use outside knowledge.
Keep answers short: one to three sentences."""

NO_ANSWER = "The documents do not appear to cover this."


@dataclass
class Answer:
    text: str
    mode: str                       # "extractive" or "claude"
    sources: list[dict] = field(default_factory=list)


def _sources(hits: list[Hit]) -> list[dict]:
    return [{"n": i, "citation": h.chunk.citation, "chunk_id": h.chunk.chunk_id, "text": h.chunk.text}
            for i, h in enumerate(hits, start=1)]


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 2) -> Answer:
    """Return the retrieved sentences that share the most terms with the question, with citations.

    No model is involved, so this never makes anything up; it can only quote the documents.
    """
    if not hits:
        return Answer(NO_ANSWER, "extractive")
    q_terms = set(tokenize(question))
    # One shared word is often coincidence ("stay home" vs "must stay below"), so longer
    # questions need two matching words before a sentence is quoted as the answer.
    min_overlap = 2 if len(q_terms) >= 3 else 1
    candidates = []
    for n, hit in enumerate(hits, start=1):
        for pos, sentence in enumerate(split_sentences(hit.chunk.text)):
            overlap = len(q_terms & set(tokenize(sentence)))
            if overlap >= min_overlap:
                # Prefer higher-ranked chunks and earlier sentences when overlap ties.
                candidates.append((-overlap, n, pos, sentence))
    if not candidates:
        return Answer(NO_ANSWER, "extractive", _sources(hits))
    best = sorted(candidates)[:max_sentences]
    text = " ".join(f"{sentence} [{n}]" for _, n, _, sentence in sorted(best, key=lambda c: (c[1], c[2])))
    return Answer(text, "extractive", _sources(hits))


def claude_answer(question: str, hits: list[Hit]) -> Answer:
    import anthropic

    sources = _sources(hits)
    context = "\n\n".join(f"[{s['n']}] {s['citation']}\n{s['text']}" for s in sources)
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"}],
        # If the primary model declines, the API retries on a fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return Answer("The model declined to answer this question.", "claude", sources)
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return Answer(text or NO_ANSWER, "claude", sources)


def claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def answer(question: str, hits: list[Hit], use_claude: bool | None = None) -> Answer:
    if use_claude is None:
        use_claude = claude_available()
    return claude_answer(question, hits) if use_claude else extractive_answer(question, hits)
