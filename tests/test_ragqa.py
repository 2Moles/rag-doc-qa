import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragqa import answer as answer_mod
from ragqa.documents import Page, chunk_pages, load_folder, split_sentences
from ragqa.evaluate import evaluate, load_examples
from ragqa.retrieval import BM25, HybridIndex, tokenize

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "sample_docs"


@pytest.fixture(scope="module")
def index():
    return HybridIndex(chunk_pages(load_folder(DOCS)))


# --- documents -------------------------------------------------------------

def test_split_sentences_keeps_decimals_together():
    assert split_sentences("At 1.5% methane, power is cut. Workers leave.") == [
        "At 1.5% methane, power is cut.", "Workers leave."]


def test_chunks_respect_max_size_and_headings():
    text = "# A\n\n" + " ".join(f"Sentence number {i} is here." for i in range(40)) + "\n\n# B\n\nShort one."
    chunks = chunk_pages([Page("doc.md", None, text)], max_chars=200)
    assert all(len(c.text) <= 200 + 40 for c in chunks)  # a single sentence may overflow slightly
    assert {c.heading for c in chunks} == {"A", "B"}
    assert chunks[-1].text == "Short one."


def test_consecutive_chunks_overlap_by_one_sentence():
    text = "# A\n\n" + " ".join(f"Fact {i} is true." for i in range(30))
    a, b = chunk_pages([Page("doc.md", None, text)], max_chars=120)[:2]
    assert a.text.split(". ")[-1].rstrip(".") in b.text


def test_chunk_ids_are_unique(index):
    ids = [c.chunk_id for c in index.chunks]
    assert len(ids) == len(set(ids))


# --- retrieval -------------------------------------------------------------

def test_tokenize_drops_stopwords_and_keeps_numbers():
    assert tokenize("What is the pH at 6.5?") == ["ph", "6.5"]


def test_bm25_prefers_rarer_matching_term():
    bm25 = BM25([["gas", "methane"], ["gas", "fan"], ["gas", "noise"]])
    scores = bm25.scores(["methane", "gas"])
    assert scores.argmax() == 0


@pytest.mark.parametrize("mode", HybridIndex.MODES)
def test_each_mode_finds_the_methane_rule(index, mode):
    hits = index.search("At what methane level is power cut?", k=3, mode=mode)
    assert any("1.5% methane" in h.chunk.text for h in hits)


def test_unknown_mode_rejected(index):
    with pytest.raises(ValueError):
        index.search("anything", mode="dense")


def test_hybrid_retrieval_meets_quality_bar(index):
    """Guards against regressions: the fused ranking should find the answer for nearly every question."""
    examples = load_examples(ROOT / "eval" / "questions.jsonl")
    scores = evaluate(index, examples, k=3, mode="hybrid")
    assert scores.hit_rate >= 0.9
    assert scores.mrr >= 0.8


# --- answering -------------------------------------------------------------

def test_extractive_answer_quotes_and_cites(index):
    q = "What is the allowed pH range for discharge water?"
    result = answer_mod.extractive_answer(q, index.search(q))
    assert result.mode == "extractive"
    assert "6.5 to 8.5" in result.text and "[1]" in result.text
    assert result.sources[0]["citation"].startswith("environmental_monitoring.md")


def test_extractive_answer_with_no_hits():
    assert answer_mod.extractive_answer("anything", []).text == answer_mod.NO_ANSWER


def _fake_anthropic(monkeypatch, stop_reason="end_turn", text="pH must be 6.5 to 8.5 [1]."):
    calls = {}

    class FakeMessages:
        def create(self, **kwargs):
            calls.update(kwargs)
            block = types.SimpleNamespace(type="text", text=text)
            return types.SimpleNamespace(stop_reason=stop_reason, content=[block])

    fake = types.SimpleNamespace(
        Anthropic=lambda: types.SimpleNamespace(beta=types.SimpleNamespace(messages=FakeMessages())))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return calls


def test_claude_answer_sends_numbered_sources(monkeypatch, index):
    calls = _fake_anthropic(monkeypatch)
    q = "How is water sampled and what pH is allowed?"
    hits = index.search(q, k=2)
    assert len(hits) == 2
    result = answer_mod.claude_answer(q, hits)
    assert result.mode == "claude" and "[1]" in result.text
    prompt = calls["messages"][0]["content"]
    assert prompt.startswith("Sources:") and "[1]" in prompt and "[2]" in prompt
    assert calls["model"] == answer_mod.MODEL
    assert calls["fallbacks"] == "default"


def test_claude_refusal_is_handled(monkeypatch, index):
    _fake_anthropic(monkeypatch, stop_reason="refusal", text="")
    result = answer_mod.claude_answer("q", index.search("pH"))
    assert "declined" in result.text


# --- API -------------------------------------------------------------------

def test_api_ask_and_health(monkeypatch):
    monkeypatch.setenv("RAGQA_DOCS", str(DOCS))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    from ragqa import api

    api.get_index.cache_clear()
    client = TestClient(api.app)
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["answer_mode"] == "extractive"

    body = client.post("/ask", json={"question": "How many days of paid annual leave do employees get?"}).json()
    assert "18 days" in body["answer"]
    assert body["sources"][0]["citation"].startswith("employee_policies.md")


def test_api_validates_input():
    from ragqa import api

    client = TestClient(api.app)
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "dense"}).status_code == 422


def test_single_coincidental_word_is_not_an_answer(index):
    q = "How long can staff stay home ill?"  # only "stay" matches, in an unrelated water-quality rule
    assert answer_mod.extractive_answer(q, index.search(q)).text == answer_mod.NO_ANSWER
