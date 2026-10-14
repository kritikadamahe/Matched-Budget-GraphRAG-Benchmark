"""Fixtures for the Phase 11 tests. Matrices are produced by the REAL Phase 10 runner, using doubles
that report known prices / tokens / runtimes, so every expected total is exact. Nothing here can make
an API call. Edge cases are made by editing a copy of the resulting files."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.conditions import MatrixSpec, generate_conditions
from experiments.runner import run_matrix, write_manifest
from extraction.mock_extractor import MockExtractor
from extraction.pricing import cost_usd
from extraction.schemas import Usage
from src.corpus import build_chunk_manifest
from tests.helpers_experiments import (answer_reply, base_cfg, fake_answerer, fake_judge, judge_verdict, make_ctx,
                                       spy_components)

# per-call constants of the priced doubles
EXT_IN, EXT_OUT, EXT_COST, EXT_RUNTIME = 100, 50, 0.01, 0.5
GEN_RUNTIME, JUDGE_RUNTIME = 0.2, 0.1
FAKE_IN, FAKE_OUT = 100, 50                      # tokens the fake OpenAI client reports per call
CALL_COST = cost_usd(FAKE_IN, FAKE_OUT, 0.15, 0.60)


class PricedExtractor(MockExtractor):
    """A non-mock ('openai') extractor with known usage and runtime per call."""
    name = "openai"

    def extract(self, item):
        return super().extract(item).model_copy(update={
            "usage": Usage(input_tokens=EXT_IN, output_tokens=EXT_OUT, cost_usd=EXT_COST), "runtime_seconds": EXT_RUNTIME})


class PartlyUnknownExtractor(PricedExtractor):
    """The first `n_unknown` calls report no usage at all (unknown, not zero)."""
    def __init__(self, n_unknown=1):
        self.n_unknown, self.n = n_unknown, 0

    def extract(self, item):
        self.n += 1
        r = super().extract(item)
        return r.model_copy(update={"usage": Usage.unknown()}) if self.n <= self.n_unknown else r


class AbortingPricedExtractor(PricedExtractor):
    def __init__(self, after=1):
        self.after, self.n = after, 0

    def extract(self, item):
        from extraction.base_extractor import ExtractionAbort
        self.n += 1
        if self.n > self.after:
            raise ExtractionAbort("insufficient_quota: you exceeded your current quota")
        return super().extract(item)


class TimedAnswerer:
    """Wraps a fake-client Answerer and stamps a known runtime on each real call."""
    def __init__(self, inner):
        self.inner = inner

    def answer(self, question, context):
        r = self.inner.answer(question, context)
        return r if r.status == "skipped_empty_context" else r.model_copy(update={"runtime_seconds": GEN_RUNTIME})

    def __getattr__(self, name):
        return getattr(self.inner, name)


class TimedJudge:
    def __init__(self, inner):
        self.inner = inner

    def judge(self, *a, **k):
        return self.inner.judge(*a, **k).model_copy(update={"runtime_seconds": JUDGE_RUNTIME})

    def __getattr__(self, name):
        return getattr(self.inner, name)


def mock_corpus(cfg):
    """The bundled mock corpus, loadable under any data_source label (lets 'huggingface' configs run offline)."""
    return build_chunk_manifest(cfg.dataset, "mock", cfg.num_questions, cfg.seed, cfg.chunk_size_words,
                                cfg.chunk_overlap_words)


def priced_components(extractor=PricedExtractor, **kw):
    return spy_components([], build_extractor=lambda cfg, dry_run=False: extractor(),
                          answerer=lambda cfg: TimedAnswerer(fake_answerer(handler=lambda k: answer_reply("Yes"))),
                          judge=lambda cfg: TimedJudge(fake_judge(handler=lambda k: judge_verdict(True))),
                          load_corpus=mock_corpus, **kw)


PRICED_CFG = dict(experiment_id="priced", data_source="huggingface", extraction_backend="openai",
                  generation_backend="openai", judge_backend="openai", judge_model="gpt-4o")


def build_matrix(root, *, strategies=("random", "ketrag"), budgets=(0.25, 0.5), datasets=("hotpotqa",), seeds=None,
                 random_seeds=None, comps=None, no_judge=False, mock=False, **cfg_kw) -> Path:
    """Run the real runner over a small matrix and return the matrix directory. `mock=True` uses the plain
    mock components and the mock dataset (what the free runs produce); otherwise the priced, non-mock doubles."""
    root = Path(root)
    if mock:
        cfg_kw = {"experiment_id": "mockbench", **cfg_kw}
        ctx = make_ctx(root, cfg=base_cfg(root, **cfg_kw), no_judge=no_judge)
    else:
        cfg = base_cfg(root, **{**PRICED_CFG, **cfg_kw})
        ctx = make_ctx(root, cfg=cfg, comps=comps or priced_components(), no_judge=no_judge)
    spec = MatrixSpec(datasets=list(datasets), strategies=list(strategies), budgets=list(budgets), seeds=seeds,
                      random_seeds=random_seeds)
    conds = generate_conditions(spec, ctx.base_cfg.seed)
    write_manifest(ctx, spec, conds)
    run_matrix(conds, ctx, all_conditions=conds)
    return ctx.matrix_dir


def copy_matrix(src, dst) -> Path:
    shutil.copytree(src, dst)
    return Path(dst)


# ----------------------------------------------------------------------------- editing helpers
def cond_dir(matrix_dir, cond_id) -> Path:
    return Path(matrix_dir) / "conditions" / cond_id


def edit_json(path, fn):
    path = Path(path)
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def edit_jsonl(path, fn):
    """fn(rows) mutates the list of row dicts in place."""
    path = Path(path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    fn(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def cond_ids(matrix_dir):
    return sorted(p.name for p in (Path(matrix_dir) / "conditions").iterdir())
