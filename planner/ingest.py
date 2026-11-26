"""
Uploaded files -> documents ({doc_id: {"title", "text"}}) for src.chunking.chunk_corpus.

Plain text, Markdown (split at headings), HTML (tags stripped), CSV/JSON (as text) and
PDF (via poppler's `pdftotext`, which most Linux systems have). Anything else is
reported back as skipped rather than guessed at.
"""

from __future__ import annotations

import hashlib
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

MAX_FILES = 200
MAX_TOTAL_BYTES = 40 * 1024 * 1024
SUPPORTED = (".txt", ".md", ".markdown", ".pdf", ".html", ".htm", ".csv", ".json", ".rst")


def corpus_id(files: list[tuple[str, bytes]]) -> str:
    h = hashlib.sha1()
    for name, data in sorted(files, key=lambda f: f[0]):
        h.update(name.encode() + b"\0" + hashlib.sha1(data).digest())
    return h.hexdigest()[:12]


def _pdf_text(data: bytes) -> str:
    if not shutil.which("pdftotext"):
        raise ValueError("PDF support needs poppler's pdftotext (apt install poppler-utils)")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(data)
        f.flush()
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", f.name, "-"], capture_output=True, timeout=120)
    if out.returncode != 0:
        raise ValueError(f"pdftotext failed: {out.stderr.decode(errors='replace')[:200]}")
    return out.stdout.decode("utf-8", errors="replace")


def _html_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    return html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))


def _markdown_sections(stem: str, text: str) -> list[tuple[str, str]]:
    """Split at #/## headings so chunk titles say which section they came from."""
    parts, title, buf = [], stem, []
    for line in text.splitlines():
        m = re.match(r"^\s{0,3}#{1,2}\s+(.+?)\s*#*\s*$", line)
        if m:
            if " ".join(buf).strip():
                parts.append((title, "\n".join(buf)))
            title, buf = f"{stem} › {m.group(1)}", []
        else:
            buf.append(line)
    if " ".join(buf).strip():
        parts.append((title, "\n".join(buf)))
    return parts


def parse_files(files: list[tuple[str, bytes]]) -> tuple[dict, list[dict]]:
    """Returns (documents, report): one report row per file with words or the reason it was skipped."""
    if len(files) > MAX_FILES:
        raise ValueError(f"at most {MAX_FILES} files per corpus")
    if sum(len(d) for _, d in files) > MAX_TOTAL_BYTES:
        raise ValueError(f"at most {MAX_TOTAL_BYTES // (1024 * 1024)} MB per corpus")
    documents, report = {}, []
    for name, data in files:
        path = Path(name)
        ext, stem = path.suffix.lower(), path.stem
        try:
            if ext not in SUPPORTED:
                raise ValueError(f"unsupported type {ext or '(none)'}")
            if ext == ".pdf":
                sections = [(stem, _pdf_text(data))]
            else:
                text = data.decode("utf-8", errors="replace")
                if ext in (".html", ".htm"):
                    sections = [(stem, _html_text(text))]
                elif ext in (".md", ".markdown"):
                    sections = _markdown_sections(stem, text)
                else:
                    sections = [(stem, text)]
            words = 0
            for i, (title, text) in enumerate(sections):
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    documents[f"{name}#{i}"] = {"title": title, "text": text}
                    words += len(text.split())
            if not words:
                raise ValueError("no text found (a scanned PDF needs OCR first)")
            report.append({"file": name, "ok": True, "words": words, "sections": len(sections)})
        except (ValueError, subprocess.TimeoutExpired) as e:
            report.append({"file": name, "ok": False, "reason": str(e)})
    return documents, report
