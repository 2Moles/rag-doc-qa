"""Loading files and splitting them into overlapping, citeable chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


@dataclass(frozen=True)
class Page:
    source: str       # file name, used in citations
    page: int | None  # 1-based page number for PDFs, None for text files
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str      # e.g. "safety_handbook.md#4"
    source: str
    page: int | None
    heading: str       # nearest Markdown heading above the chunk ("" if none)
    text: str

    @property
    def citation(self) -> str:
        where = f"p.{self.page}" if self.page else self.heading or "start"
        return f"{self.source} ({where})"


def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [Page(path.name, i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    if suffix in {".md", ".txt"}:
        return [Page(path.name, None, path.read_text(encoding="utf-8"))]
    raise ValueError(f"unsupported file type: {path.name}")


def load_folder(folder: Path) -> list[Page]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise FileNotFoundError(f"no .md, .txt or .pdf files in {folder}")
    return [page for f in files for page in load_file(f)]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_pages(pages: list[Page], max_chars: int = 600, overlap_sentences: int = 1) -> list[Chunk]:
    """Split each page into chunks of whole sentences, up to max_chars each.

    Chunks never cross a Markdown heading, so each chunk belongs to one section and its
    citation stays accurate. Consecutive chunks inside a section share `overlap_sentences`
    sentences, so a fact split across a boundary is still retrievable.
    """
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for page in pages:
        for heading, body in _sections(page.text):
            sentences = [s for para in body.split("\n\n") for s in split_sentences(para.replace("\n", " "))]
            start = 0
            while start < len(sentences):
                end, size = start, 0
                while end < len(sentences) and (end == start or size + len(sentences[end]) <= max_chars):
                    size += len(sentences[end]) + 1
                    end += 1
                n = counters.get(page.source, 0)
                counters[page.source] = n + 1
                chunks.append(Chunk(f"{page.source}#{n}", page.source, page.page, heading,
                                    " ".join(sentences[start:end])))
                if end >= len(sentences):
                    break
                start = max(end - overlap_sentences, start + 1)
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading, body) pairs. Plain text becomes one section with no heading."""
    sections: list[tuple[str, str]] = []
    heading, lines = "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if any(line.strip() for line in lines):
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    if any(line.strip() for line in lines):
        sections.append((heading, "\n".join(lines).strip()))
    return sections
