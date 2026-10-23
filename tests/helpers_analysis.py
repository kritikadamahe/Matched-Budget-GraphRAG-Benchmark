"""Fixtures for the Phase 12 tests. Matrices come from the REAL Phase 10 runner (via the Phase 11 helpers).

SYNTHETIC "real-looking" matrices: `real_matrix` uses the same non-mock test doubles as the Phase 11 tests
(a fake OpenAI client behind backends labelled "openai", data_source "huggingface" over the bundled mock
corpus). They exercise the analysis CODE only. They are not benchmark results and say nothing about the
research question. Nothing here can make an API call."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.helpers_aggregation import (build_matrix, cond_dir, cond_ids, copy_matrix, edit_json, edit_jsonl,  # noqa: F401
                                       priced_components)

# the first mock question (HotpotQA fixture): used to make retrieval fail for one question
Q1_TEXT = "What nationality is the director of the film that won the 1994 Best Picture Oscar?"


def real_matrix(root, **kw) -> Path:
    """Synthetic non-mock matrix (see module docstring). Defaults: hotpotqa, random + ketrag, 25% and 50%."""
    return build_matrix(root, mock=False, **kw)


def mock_matrix(root, **kw) -> Path:
    """What the free mock runs produce (mock dataset, mock backends)."""
    return build_matrix(root, mock=True, **kw)


def file_hashes(root) -> dict:
    """{relative path: sha256} for every file under root (to prove a read-only operation)."""
    import hashlib
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}
