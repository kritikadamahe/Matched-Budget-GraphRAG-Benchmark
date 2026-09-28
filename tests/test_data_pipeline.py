"""Roadmap Phases 2-3: dataset loading, pooling, chunk manifest, checkpoints.
Real-data tests are skipped unless the dev sets are in data/raw/
(run `python -m src.prepare_data --config configs/hotpotqa_dev.yaml` once)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pytest

from src.chunking import chunk_corpus
from src.config import ExperimentConfig
from src.corpus import (_read_all, build_chunk_manifest, load_raw_questions,
                        pool_and_dedup_contexts, raw_file_path)
from src.prepare_data import corpus_id, prepare


# ------------------------------------------------------------- MuSiQue (mock)
def test_musique_mock_loads_and_skips_unanswerable():
    rows = _read_all("musique", "mock")
    assert [q["question_id"] for q in rows] == ["2hop__m1", "2hop__m2", "3hop__m3"]
    assert rows[0]["answer_aliases"] == ["French Republic"]


def test_same_title_different_paragraphs_stay_separate():
    questions = load_raw_questions("musique", "mock", num_questions=3, seed=0)
    documents = pool_and_dedup_contexts(questions)
    ravel = [d for d in documents.values() if d["title"] == "Maurice Ravel"]
    green = [d for d in documents.values() if d["title"] == "Green"]
    assert len(ravel) == 2 and len(green) == 2            # different text -> different documents
    paris = [d for d in documents.values() if d["title"] == "Paris"]
    assert len(paris) == 1                                 # identical text in 2 questions -> one document


def test_gold_is_the_exact_paragraph_not_the_title():
    # q m1's gold "Maurice Ravel" paragraph is the SECOND one with that title
    # (the case title-only dedup silently dropped in 466 real MuSiQue questions).
    questions = load_raw_questions("musique", "mock", num_questions=3, seed=0)
    documents = pool_and_dedup_contexts(questions)
    m1 = next(q for q in questions if q["question_id"] == "2hop__m1")
    gold_texts = sorted(documents[d]["text"] for d in m1["gold_doc_ids"])
    assert any("born in Ciboure" in t for t in gold_texts)
    assert not any(t == "Maurice Ravel was a French composer, pianist and conductor." for t in gold_texts)


def test_chunk_ids_unique_when_titles_repeat():
    chunks, _, _ = build_chunk_manifest("musique", "mock", 3, 0, 250, 40)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    gold = [c for c in chunks if "2hop__m1" in c.is_gold_for_question_ids]
    assert {c.source_title for c in gold} == {"Bolero", "Maurice Ravel"} and len(gold) == 2


def test_simple_title_dict_still_works_for_chunk_corpus():
    chunks = chunk_corpus({"B": "x y z", "A": "p q r"}, chunk_size_words=2, overlap_words=0)
    assert chunks[0].source_title == "A" and chunks[0].source_doc_id == "A"


# ------------------------------------------------------- prepare_data (mock)
@pytest.mark.parametrize("dataset,n", [("hotpotqa", 5), ("musique", 3)])
def test_prepare_writes_manifest_and_passes_checkpoints(tmp_path, dataset, n):
    cfg = ExperimentConfig(experiment_id="t", seed=0, dataset=dataset, data_source="mock",
                           num_questions=n, strategy="random", budget=0.1)
    meta = prepare(cfg, out_root=tmp_path)
    out = tmp_path / corpus_id(cfg)
    assert {p.name for p in out.iterdir()} == {"documents.json", "questions.json", "chunks.json", "meta.json"}
    assert all(meta["checks"].values())
    assert meta["N_chunks"] == len(json.loads((out / "chunks.json").read_text()))
    assert meta["chunks_per_budget"]["100%"] == meta["N_chunks"]
    # same config twice -> same chunk_ids (Phase 3 checkpoint across processes)
    assert prepare(cfg, out_root=tmp_path)["chunk_id_fingerprint"] == meta["chunk_id_fingerprint"]


# ------------------------------------------------------------- real dev sets
def _needs(dataset):
    return pytest.mark.skipif(not raw_file_path(dataset).exists(),
                              reason=f"{dataset} dev set not downloaded")


@_needs("hotpotqa")
def test_real_hotpotqa_counts_and_format():
    rows = _read_all("hotpotqa", "huggingface")
    assert len(rows) == 7405                                         # official dev size
    assert all(q["supporting_paragraphs"] for q in rows)
    assert all(len({t for t, _ in q["context"]}) == len(q["context"]) for q in rows)   # titles unique per question


@_needs("musique")
def test_real_musique_counts():
    rows = _read_all("musique", "huggingface")
    assert len(rows) == 2417                                         # official MuSiQue-Ans dev size
    hops = [q["question_id"].split("__")[0][:2] for q in rows]
    assert (hops.count("2h"), hops.count("3h"), hops.count("4h")) == (1252, 760, 405)


@pytest.mark.parametrize("dataset", [pytest.param("hotpotqa", marks=_needs("hotpotqa")),
                                     pytest.param("musique", marks=_needs("musique"))])
def test_real_prepare_20_questions(tmp_path, dataset):
    cfg = ExperimentConfig(experiment_id="t", seed=42, dataset=dataset, data_source="huggingface",
                           num_questions=20, strategy="random", budget=0.1)
    meta = prepare(cfg, out_root=tmp_path)
    assert all(meta["checks"].values())
    assert meta["N_chunks"] >= meta["num_documents"] > 0
