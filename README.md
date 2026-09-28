# rag-doc-qa

Ask questions about your own documents (Markdown, text or PDF) and get short answers with citations back to the source section.

It runs **without any API key**: answers are quoted directly from the retrieved text. Set `ANTHROPIC_API_KEY` and the same pipeline uses Claude to write the answer, still restricted to the retrieved sources and still citing them.

```
 files -> sections -> sentence chunks (with overlap) -> BM25 + TF-IDF -> Reciprocal Rank Fusion -> top-k
                                                                                            |
                                                  extractive answer (no key)  <-------------+------>  Claude answer (key set)
```

```
$ python -m ragqa ask "When do I need a medical certificate?"
A medical certificate is required for any absence longer than 2 consecutive days. [1]

(extractive mode)
  [1] employee_policies.md (Leave)
```

## What it shows

- **Chunking that keeps citations accurate:** chunks are built from whole sentences, never cross a Markdown heading, and overlap by one sentence, so a fact on a boundary is still found. Decimals like `1.5%` aren't split into separate sentences.
- **Hybrid retrieval:** a hand-written Okapi BM25 (exact keyword match) plus TF-IDF cosine over word 1–2-grams (scikit-learn), combined with **Reciprocal Rank Fusion**, so the two score scales never need calibrating.
- **Grounded answers:** extractive mode can only quote the documents. Claude mode gets numbered sources, must cite them, and must say when the documents don't cover the question. It also handles `refusal` stop reasons, with server-side fallback turned on.
- **Evaluation:** a labelled question set ([`eval/questions.jsonl`](eval/questions.jsonl)) scored by hit rate@k and MRR. A hit only counts when the retrieved chunk is from the right document **and** contains the answer text. The test suite fails if hybrid retrieval drops below 0.9 hit@3.
- **API:** FastAPI with input validation (`/ask`, `/health`), Docker, and GitHub Actions CI.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # add ,claude to enable the Claude answer mode
python -m ragqa ask "What is the allowed pH range for discharge water?"
python -m ragqa eval               # compare bm25 / tfidf / hybrid
python -m ragqa --docs path/to/your/pdfs ask "..."
uvicorn ragqa.api:app --reload     # then POST /ask {"question": "..."}; docs at /docs
pytest -q
```

To use Claude: `pip install -e ".[claude]"`, then `export ANTHROPIC_API_KEY=...`. The default model is `claude-opus-5`; override it with `RAGQA_MODEL`.

Docker: `docker build -t ragqa . && docker run -p 8000:8000 ragqa`

## Evaluation results (sample documents, k = 3)

| mode | hit@3 | MRR |
|---|---|---|
| bm25 | 1.00 | 0.97 |
| tfidf | 1.00 | 1.00 |
| hybrid | 1.00 | 0.97 |

The sample set is small (15 questions, 17 chunks), and most questions share words with their answers, so all three modes score near-perfectly. It works as a **regression check**, not a benchmark. Harder, paraphrased questions are the next step (see below).

## Sample data

`data/sample_docs/` contains three short **fictional** documents (a mining safety handbook, an environmental monitoring plan and HR policies) written for this demo. The company, and the limits in them, are made up.

## Limitations and next steps

- **Lexical retrieval only.** A question that shares no words with the answer ("How long can staff stay home ill?") can miss. The fix is to add a dense embedding retriever as a third ranking in the RRF fusion.
- **No re-ranking.** A cross-encoder re-ranker over the top 20 would improve precision on larger collections.
- **In-memory index,** rebuilt on start-up. Fine for hundreds of documents; beyond that, persist to a vector store such as Milvus or pgvector.
- The Claude answer path is covered by tests with a stand-in client. It has not been evaluated for answer quality. An answer-level eval (faithfulness, citation accuracy) would be the next addition.

## License

MIT
