"""Command line: python -m ragqa ask "question" | python -m ragqa eval"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .answer import answer
from .documents import chunk_pages, load_folder
from .evaluate import evaluate, load_examples
from .retrieval import HybridIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragqa", description="Question answering over your documents.")
    parser.add_argument("--docs", type=Path, default=Path("data/sample_docs"), help="folder of .md/.txt/.pdf files")
    sub = parser.add_subparsers(dest="command", required=True)

    ask_p = sub.add_parser("ask", help="answer a question")
    ask_p.add_argument("question")
    ask_p.add_argument("-k", type=int, default=4)
    ask_p.add_argument("--mode", choices=HybridIndex.MODES, default="hybrid")
    ask_p.add_argument("--extractive", action="store_true", help="never call Claude, even if a key is set")

    eval_p = sub.add_parser("eval", help="compare retrieval modes on a labelled question set")
    eval_p.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    eval_p.add_argument("-k", type=int, default=3)

    args = parser.parse_args(argv)
    index = HybridIndex(chunk_pages(load_folder(args.docs)))

    if args.command == "ask":
        hits = index.search(args.question, k=args.k, mode=args.mode)
        result = answer(args.question, hits, use_claude=False if args.extractive else None)
        print(f"{result.text}\n\n({result.mode} mode)")
        for s in result.sources:
            print(f"  [{s['n']}] {s['citation']}")
        return 0

    examples = load_examples(args.questions)
    print(f"{len(examples)} questions, {len(index.chunks)} chunks, k={args.k}\n")
    print(f"{'mode':<8} {'hit@k':>6} {'MRR':>6}")
    for mode in HybridIndex.MODES:
        s = evaluate(index, examples, k=args.k, mode=mode)
        print(f"{mode:<8} {s.hit_rate:>6.2f} {s.mrr:>6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
