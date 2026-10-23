"""Phase 12, increment 3: analysis/evidence.py. The metric core is checked against hand-calculated cases (every
metric, every undefined case); the integration tests run the real Phase 10 runner (free doubles; the 'real-looking'
matrices are SYNTHETIC, see tests/helpers_analysis.py) and recompute every number independently with plain set
arithmetic from the raw files. No API call."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import json
import re

import numpy as np
import pytest

from analysis.evidence import (EVIDENCE_METRICS, EvidenceError, GoldSource, QuestionEvidence, compute_evidence,
                               evidence_arm, find_gold_source, gold_from_chunks, gold_from_corpus_dir,
                               question_evidence)
from analysis.loader import load_analysis_input
from analysis.random_avg import ArmError, pair_arms
from analysis.stats import compare_paired, describe_arm
from src.config import ExperimentConfig
from src.corpus import build_chunk_manifest
from src.prepare_data import prepare
from tests.helpers_aggregation import PricedExtractor, TimedAnswerer, TimedJudge, mock_corpus
from tests.helpers_analysis import (Q1_TEXT, cond_dir, copy_matrix, edit_json, edit_jsonl, file_hashes, mock_matrix,
                                    priced_components, real_matrix)
from tests.helpers_experiments import (SpyExtractor, answer_reply, base_cfg, fake_answerer, fake_judge, judge_verdict,
                                       spy_components)

PROJECT = Path(__file__).resolve().parents[1]
MOCK_CHUNKS, MOCK_QUESTIONS, _ = build_chunk_manifest("hotpotqa", "mock", 3, 42, 250, 40)
CID, KET = "hotpotqa__random__b050__seed42", "hotpotqa__ketrag__b050__seed42"


def gold_source():
    return {"hotpotqa": gold_from_chunks(MOCK_CHUNKS, origin="test mock corpus")}


def ev(G, S, R):
    return question_evidence(G, S, R)


# ============================================================================== the metric core, by hand
def test_all_metrics_for_a_typical_question():
    # G={a,b}  S={a,b,c}  R={a,c}:  |G∩S|=2  |G∩R|=1
    e = ev({"a", "b"}, {"a", "b", "c"}, {"a", "c"})
    assert e.selection_recall == 1.0                              # 2/2
    assert e.retrieval_recall == 0.5                              # 1/2   (unconditional)
    assert e.conditional_retrieval_recall == 0.5                  # 1/2   (|G∩R| / |G∩S|)
    assert (e.all_gold_selected, e.all_gold_retrieved) == (1.0, 0.0)
    assert e.all_gold_retrieved_given_selected == 0.0             # G ⊆ S, but G ⊄ R
    assert (e.n_gold, e.n_selected_gold, e.n_retrieved_gold, e.n_selected, e.n_retrieved) == (2, 2, 1, 3, 2)
    assert e.undefined == {} and e.retrieved_not_selected == ()


def test_partial_selection_makes_the_three_recalls_differ():
    # G={a,b,c}  S={a,b}  R={a}:  selection 2/3, unconditional retrieval 1/3, conditional 1/2
    e = ev({"a", "b", "c"}, {"a", "b"}, {"a"})
    assert e.selection_recall == pytest.approx(2 / 3)
    assert e.retrieval_recall == pytest.approx(1 / 3)
    assert e.conditional_retrieval_recall == pytest.approx(1 / 2)
    assert (e.all_gold_selected, e.all_gold_retrieved) == (0.0, 0.0)
    assert e.all_gold_retrieved_given_selected is None and e.undefined == {"all_gold_retrieved_given_selected": "not_all_gold_selected"}


def test_everything_selected_and_retrieved():
    e = ev({"a", "b"}, {"a", "b", "c"}, {"a", "b"})
    assert (e.selection_recall, e.retrieval_recall, e.conditional_retrieval_recall) == (1.0, 1.0, 1.0)
    assert (e.all_gold_selected, e.all_gold_retrieved, e.all_gold_retrieved_given_selected) == (1.0, 1.0, 1.0)


def test_no_gold_chunk_selected_leaves_the_conditional_metric_undefined_not_zero():
    # G={a,b}  S={c,d}  R={c}: selection 0, retrieval 0, conditional = 0/0
    e = ev({"a", "b"}, {"c", "d"}, {"c"})
    assert (e.selection_recall, e.retrieval_recall) == (0.0, 0.0) and (e.all_gold_selected, e.all_gold_retrieved) == (0.0, 0.0)
    assert e.conditional_retrieval_recall is None and e.undefined["conditional_retrieval_recall"] == "no_selected_gold"
    assert e.undefined["all_gold_retrieved_given_selected"] == "not_all_gold_selected"


def test_gold_selected_but_nothing_retrieved_is_a_real_zero():
    # G={a}  S={a}  R={} (e.g. an empty context): the retriever missed a selected gold chunk -> 0, not undefined
    e = ev({"a"}, {"a"}, set())
    assert (e.selection_recall, e.retrieval_recall, e.conditional_retrieval_recall) == (1.0, 0.0, 0.0)
    assert (e.all_gold_retrieved, e.all_gold_retrieved_given_selected) == (0.0, 0.0) and e.n_retrieved == 0


def test_an_empty_gold_set_leaves_every_metric_undefined():
    for S, R in (({"a"}, {"a"}), (None, None), (set(), set())):
        e = ev(set(), S, R)
        assert all(getattr(e, m) is None for m in EVIDENCE_METRICS)
        assert e.undefined == {m: "empty_gold" for m in EVIDENCE_METRICS} and e.n_gold == 0


def test_an_unknown_selection_leaves_only_the_selection_based_metrics_undefined():
    e = ev({"a", "b"}, None, {"a"})
    assert e.selection_recall is None and e.all_gold_selected is None
    assert e.retrieval_recall == 0.5 and e.all_gold_retrieved == 0.0                   # needs R only
    assert e.undefined == {"selection_recall": "no_selection_file", "all_gold_selected": "no_selection_file",
                           "conditional_retrieval_recall": "no_selection_file",
                           "all_gold_retrieved_given_selected": "no_selection_file"}


def test_a_question_without_retrieval_keeps_its_selection_metrics():
    e = ev({"a", "b"}, {"a"}, None)
    assert e.selection_recall == 0.5 and e.all_gold_selected == 0.0
    assert e.retrieval_recall is None and e.all_gold_retrieved is None
    assert e.undefined == {"retrieval_recall": "no_prediction", "all_gold_retrieved": "no_prediction",
                           "conditional_retrieval_recall": "no_prediction", "all_gold_retrieved_given_selected": "no_prediction"}


def test_retrieved_chunks_that_were_never_selected_are_reported_and_void_the_conditional_metrics():
    # S={a}  R={a,b}  G={a,b}: R ⊄ S is impossible in a correct run. Reported, not hidden or "fixed".
    e = ev({"a", "b"}, {"a"}, {"a", "b"})
    assert e.retrieved_not_selected == ("b",)
    assert e.selection_recall == 0.5 and e.retrieval_recall == 1.0                       # shown as computed: retrieval > selection
    assert e.conditional_retrieval_recall is None and e.undefined["conditional_retrieval_recall"] == "retrieved_not_in_selected"
    assert e.undefined["all_gold_retrieved_given_selected"] == "retrieved_not_in_selected"


def test_inputs_are_treated_as_sets():
    a = ev(["a", "a", "b"], ["a", "b", "b", "c"], ["a", "a"])
    b = ev({"a", "b"}, {"a", "b", "c"}, {"a"})
    assert a == b and a.n_gold == 2 and a.n_selected == 3 and a.n_retrieved == 1


def test_the_metric_identities_hold_on_random_consistent_inputs():
    rng = np.random.default_rng(0)
    chunks = [f"c{i}" for i in range(12)]
    for _ in range(300):
        G = set(rng.choice(chunks, size=int(rng.integers(1, 5)), replace=False))
        S = set(rng.choice(chunks, size=int(rng.integers(0, 12)), replace=False))
        R = set(c for c in S if rng.random() < 0.5)                                   # R ⊆ S by construction
        e = ev(G, S, R)
        assert e.retrieved_not_selected == ()
        assert e.retrieval_recall <= e.selection_recall + 1e-12                        # unconditional never exceeds selection
        if e.conditional_retrieval_recall is not None:                                 # recall_ret = cond * recall_sel
            assert e.retrieval_recall == pytest.approx(e.conditional_retrieval_recall * e.selection_recall)
            assert 0 <= e.conditional_retrieval_recall <= 1
        else:
            assert e.n_selected_gold == 0 and e.retrieval_recall == 0
        assert (e.all_gold_selected == 1.0) == (e.selection_recall == 1.0)
        assert (e.all_gold_retrieved == 1.0) == (e.retrieval_recall == 1.0)
        assert e.all_gold_retrieved <= e.all_gold_selected                             # needs selection first
        if e.all_gold_selected == 1.0:
            assert e.all_gold_retrieved_given_selected == e.all_gold_retrieved
        else:
            assert e.all_gold_retrieved_given_selected is None


# ============================================================================== gold sources
def test_gold_from_chunks_accepts_objects_and_dicts_and_multiple_questions():
    from src.chunking import Chunk
    objs = [Chunk("c1", "t", "T", 0, 1, "d1", ["q1", "q2"]), Chunk("c2", "t", "T", 0, 1, "d2", []),
            Chunk("c3", "t", "T", 0, 1, "d3", ["q2"])]
    g = gold_from_chunks(objs, origin="o")
    assert g.gold == {"q1": frozenset({"c1"}), "q2": frozenset({"c1", "c3"})} and g.chunk_ids == ("c1", "c2", "c3")
    assert gold_from_chunks([c.to_dict() for c in objs]).gold == g.gold and g.origin == "o"
    assert g.chunk_set == frozenset({"c1", "c2", "c3"})
    with pytest.raises(EvidenceError, match="duplicate chunk ids"):
        gold_from_chunks([objs[0], objs[0]])


def test_gold_is_read_from_the_prepared_corpus_directory(tmp_path):
    cfg = base_cfg(tmp_path)                                                           # mock data: offline
    meta = prepare(cfg, out_root=tmp_path / "corpus")
    g = gold_from_corpus_dir(tmp_path / "corpus" / meta["corpus_id"])
    ref = gold_from_chunks(build_chunk_manifest("hotpotqa", "mock", 3, 42, 250, 40)[0])
    assert g.gold == ref.gold and g.chunk_ids == ref.chunk_ids and "chunks.json" in g.origin
    with pytest.raises(EvidenceError, match="cannot read"):
        gold_from_corpus_dir(tmp_path / "nowhere")


def test_find_gold_source_prefers_the_prepared_corpus_and_refuses_to_guess(tmp_path):
    m = mock_matrix(tmp_path / "m")
    a = load_analysis_input(m, allow_mock_layout_test=True)
    root = tmp_path / "corpus"
    with pytest.raises(EvidenceError, match="were not found.*prepare_data.*allow_rebuild"):
        find_gold_source(a, "hotpotqa", corpus_root=root)
    prepare(base_cfg(tmp_path), out_root=root)
    g = find_gold_source(a, "hotpotqa", corpus_root=root)
    assert "chunks.json" in g.origin and g.gold
    rebuilt = find_gold_source(a, "hotpotqa", corpus_root=tmp_path / "empty", allow_rebuild=True)
    assert "rebuilt" in rebuilt.origin and rebuilt.gold == g.gold and rebuilt.chunk_ids == g.chunk_ids


def test_a_failing_rebuild_is_an_error_not_missing_gold(tmp_path):
    a = load_analysis_input(mock_matrix(tmp_path / "m"), allow_mock_layout_test=True)
    def boom(cfg):
        raise RuntimeError("no network")
    with pytest.raises(EvidenceError, match="rebuilding the corpus.*RuntimeError: no network"):
        find_gold_source(a, "hotpotqa", corpus_root=tmp_path / "x", allow_rebuild=True, corpus_builder=boom)


# ============================================================================== a whole matrix
@pytest.fixture(scope="module")
def real(tmp_path_factory):
    return real_matrix(tmp_path_factory.mktemp("real"))


@pytest.fixture
def real_copy(real, tmp_path):
    return copy_matrix(real, tmp_path / "m")


def independent_rows(matrix_dir, cond_id):
    """Recompute every question's metrics from the RAW files with plain set arithmetic (no analysis code)."""
    gold = {}
    for c in MOCK_CHUNKS:
        for q in c.is_gold_for_question_ids:
            gold.setdefault(q, set()).add(c.chunk_id)
    d = Path(matrix_dir) / "conditions" / cond_id
    S = set(json.loads((d / "extraction" / "selection.json").read_text())["selected_chunk_ids"])
    out = {}
    for line in (d / "predictions.jsonl").read_text().splitlines():
        p = json.loads(line)
        G, R = gold[p["question_id"]], set(p["retrieval"]["passage_chunk_ids"])
        out[p["question_id"]] = dict(
            selection_recall=len(G & S) / len(G), retrieval_recall=len(G & R) / len(G),
            conditional=(len(G & R) / len(G & S)) if G & S else None,
            all_sel=float(G <= S), all_ret=float(G <= R))
    return out


def test_every_question_matches_an_independent_recomputation(real):
    a = load_analysis_input(real)
    rep = compute_evidence(a, gold_sources=gold_source())
    assert len(rep.rows) == 4 * 3 and rep.status == "real_complete"
    assert any(any(c.is_gold_for_question_ids for c in MOCK_CHUNKS) for _ in [0])                    # there IS gold to find
    for row in rep.rows:
        want = independent_rows(real, row.condition_id)[row.question_id]
        assert row.selection_recall == pytest.approx(want["selection_recall"])
        assert row.retrieval_recall == pytest.approx(want["retrieval_recall"])
        assert row.all_gold_selected == want["all_sel"] and row.all_gold_retrieved == want["all_ret"]
        if want["conditional"] is None:
            assert row.conditional_retrieval_recall is None and row.undefined["conditional_retrieval_recall"] == "no_selected_gold"
        else:
            assert row.conditional_retrieval_recall == pytest.approx(want["conditional"])


def test_the_runners_stored_evidence_is_reproduced(real):
    a = load_analysis_input(real)
    rep = compute_evidence(a, gold_sources=gold_source())
    assert rep.reconciled and rep.n_subset_violations == 0
    assert not [i for i in rep.issues if i.severity == "error"], rep.issues
    by_id = {r.condition_id: r for r in a.records}
    for ce in rep.conditions:
        stored = by_id[ce.condition_id].evidence
        m = ce.metrics
        assert m["selection_recall"].mean == pytest.approx(stored["selection_recall"], abs=1.5e-6)
        assert m["all_gold_selected"].mean == pytest.approx(stored["selection_all_gold_rate"], abs=1.5e-6)
        assert m["retrieval_recall"].mean == pytest.approx(stored["retrieval_recall"], abs=1.5e-6)
        assert m["all_gold_retrieved"].mean == pytest.approx(stored["retrieval_all_gold_rate"], abs=1.5e-6)
        assert m["selection_recall"].n_defined == stored["selection_n_questions"] == 3
        assert ce.n_selected / ce.n_chunks_total == pytest.approx(stored["expected_random_selection_recall"], abs=1.5e-6)


def test_retrieval_never_exceeds_selection_on_runner_output(real):
    rep = compute_evidence(load_analysis_input(real), gold_sources=gold_source())
    for r in rep.rows:
        assert r.retrieved_not_selected == () and r.retrieval_recall <= r.selection_recall + 1e-12


def test_at_full_budget_everything_is_selected(tmp_path):
    m = real_matrix(tmp_path, budgets=(1.0,))
    rep = compute_evidence(load_analysis_input(m), gold_sources=gold_source())
    for r in rep.rows:
        assert r.selection_recall == 1.0 and r.all_gold_selected == 1.0 and r.n_selected == len(MOCK_CHUNKS)
    for ce in rep.conditions:
        assert ce.metrics["selection_recall"].mean == 1.0 and ce.n_selected == ce.n_chunks_total
        assert ce.metrics["selection_recall"].n_undefined == 0
    assert rep.reconciled


def test_a_question_without_a_prediction_has_selection_but_no_retrieval_metrics(tmp_path):
    m = real_matrix(tmp_path, comps=priced_components(fail_retrieval_on=Q1_TEXT))
    a = load_analysis_input(m)
    rep = compute_evidence(a, gold_sources=gold_source())
    q1 = [r for r in rep.rows if r.question_id == "q1"]
    assert len(q1) == 4
    for r in q1:
        assert r.selection_recall is not None and r.retrieval_recall is None
        assert r.undefined["retrieval_recall"] == "no_prediction" and r.undefined["conditional_retrieval_recall"] == "no_prediction"
    for ce in rep.conditions:
        assert ce.metrics["selection_recall"].n_defined == 3 and ce.metrics["retrieval_recall"].n_defined == 2
        assert ce.metrics["retrieval_recall"].undefined_reasons == {"no_prediction": 1}
    assert rep.reconciled                                                            # the runner counted 3 vs 2 the same way


def test_a_missing_selection_file_leaves_selection_metrics_undefined(real_copy):
    (cond_dir(real_copy, CID) / "extraction" / "selection.json").unlink()
    rep = compute_evidence(load_analysis_input(real_copy), gold_sources=gold_source())
    ce = next(c for c in rep.conditions if c.condition_id == CID)
    assert ce.n_selected is None and ce.metrics["selection_recall"].n_defined == 0
    assert ce.metrics["selection_recall"].undefined_reasons == {"no_selection_file": 3}
    assert ce.metrics["retrieval_recall"].n_defined == 3                             # unconditional retrieval needs no S
    assert ce.metrics["retrieval_recall"].mean is not None and ce.metrics["selection_recall"].mean is None


def test_an_empty_gold_question_is_undefined_and_counted(real):
    assert {"q1", "q3", "q5"} == set(load_analysis_input(real).question_ids["hotpotqa"])          # the mock questions
    no_gold_q3 = GoldSource(gold={q: g for q, g in gold_source()["hotpotqa"].gold.items() if q != "q3"},
                            chunk_ids=gold_source()["hotpotqa"].chunk_ids, origin="test: q3 has no gold")
    rep = compute_evidence(load_analysis_input(real), gold_sources={"hotpotqa": no_gold_q3})
    q3 = [r for r in rep.rows if r.question_id == "q3"]
    assert len(q3) == 4 and all(r.undefined == {m: "empty_gold" for m in EVIDENCE_METRICS} for r in q3)
    for ce in rep.conditions:
        assert ce.n_empty_gold == 1 and ce.metrics["selection_recall"].n_defined == 2
        assert ce.metrics["selection_recall"].undefined_reasons == {"empty_gold": 1}
    # the runner counted 3 questions with gold: a gold source that disagrees is caught by the reconciliation
    assert not rep.reconciled and any(i.code == "evidence_mismatch" and "selection_n_questions" in i.message for i in rep.issues)


# ============================================================================== consistency checks
def test_a_retrieved_chunk_that_was_never_selected_is_reported(real_copy):
    sel = set(json.loads((cond_dir(real_copy, CID) / "extraction" / "selection.json").read_text())["selected_chunk_ids"])
    stray = next(c.chunk_id for c in MOCK_CHUNKS if c.chunk_id not in sel)                # a real chunk, not selected
    edit_jsonl(cond_dir(real_copy, CID) / "predictions.jsonl",
               lambda rows: rows[0]["retrieval"]["passage_chunk_ids"].append(stray))
    rep = compute_evidence(load_analysis_input(real_copy), gold_sources=gold_source())
    assert not rep.reconciled and rep.n_subset_violations == 1
    bad = next(r for r in rep.rows if r.retrieved_not_selected)
    assert bad.condition_id == CID and bad.retrieved_not_selected == (stray,)
    assert bad.conditional_retrieval_recall is None and bad.undefined["conditional_retrieval_recall"] == "retrieved_not_in_selected"
    issue = next(i for i in rep.issues if i.code == "retrieved_not_selected")
    assert issue.severity == "error" and issue.condition_id == CID and stray in issue.message
    ce = next(c for c in rep.conditions if c.condition_id == CID)
    assert ce.n_questions_retrieved_not_selected == 1


def test_a_retrieved_chunk_outside_the_corpus_is_an_error(real_copy):
    edit_jsonl(cond_dir(real_copy, CID) / "predictions.jsonl",
               lambda rows: rows[0]["retrieval"]["passage_chunk_ids"].append("deadbeef0000"))
    rep = compute_evidence(load_analysis_input(real_copy), gold_sources=gold_source())
    assert not rep.reconciled and any(i.code == "retrieved_not_in_corpus" and i.severity == "error" for i in rep.issues)


def test_a_selected_chunk_outside_the_corpus_is_an_error(real_copy):
    edit_json(cond_dir(real_copy, CID) / "extraction" / "selection.json",
              lambda d: d["selected_chunk_ids"].__setitem__(0, "deadbeef0000"))
    rep = compute_evidence(load_analysis_input(real_copy), gold_sources=gold_source())
    assert any(i.code == "selected_not_in_corpus" and i.condition_id == CID for i in rep.issues) and not rep.reconciled


def test_stored_evidence_that_disagrees_is_an_error(real_copy):
    edit_json(cond_dir(real_copy, CID) / "condition_result.json", lambda d: d["evidence"].update(selection_recall=0.987654))
    rep = compute_evidence(load_analysis_input(real_copy), gold_sources=gold_source())
    bad = [i for i in rep.issues if i.code == "evidence_mismatch"]
    assert len(bad) == 1 and bad[0].condition_id == CID and "selection_recall" in bad[0].message
    assert bad[0].expected == 0.987654 and not rep.reconciled


def test_a_run_that_stored_no_evidence_gives_warnings_not_errors(real_copy):
    edit_json(cond_dir(real_copy, CID) / "condition_result.json", lambda d: d.update(evidence={}))
    rep = compute_evidence(load_analysis_input(real_copy), gold_sources=gold_source())
    assert rep.reconciled and any(i.code == "no_stored_evidence" and i.severity == "warning" and i.condition_id == CID
                                  for i in rep.issues)


def test_gold_from_the_wrong_corpus_is_refused(real):
    a = load_analysis_input(real)
    other = gold_from_chunks(build_chunk_manifest("musique", "mock", 3, 42, 250, 40)[0], origin="a different corpus")
    with pytest.raises(EvidenceError, match="wrong corpus"):
        compute_evidence(a, gold_sources={"hotpotqa": other})
    good = gold_source()["hotpotqa"]
    shuffled = GoldSource(gold=good.gold, chunk_ids=tuple(reversed(good.chunk_ids)), origin="same chunks, other order")
    with pytest.raises(EvidenceError, match="fingerprint.*wrong corpus"):
        compute_evidence(a, gold_sources={"hotpotqa": shuffled})
    no_overlap = GoldSource(gold={"zz": frozenset(good.chunk_ids[:1])}, chunk_ids=good.chunk_ids, origin="foreign questions")
    with pytest.raises(EvidenceError, match="none of the condition's questions"):
        compute_evidence(a, gold_sources={"hotpotqa": no_overlap})


def test_without_a_stored_fingerprint_the_chunk_count_is_the_remaining_defence(real_copy):
    """An older run without a corpus fingerprint: a warning, and a gold source with the wrong number of chunks is still refused."""
    edit_json(cond_dir(real_copy, CID) / "condition_result.json",
              lambda d: d["stages"]["corpus"]["details"].pop("all_chunks_fingerprint"))
    a = load_analysis_input(real_copy)
    rep = compute_evidence(a, gold_sources=gold_source())
    assert rep.reconciled and any(i.code == "no_corpus_fingerprint" and i.condition_id == CID for i in rep.issues)
    good = gold_source()["hotpotqa"]
    short = GoldSource(gold=good.gold, chunk_ids=good.chunk_ids[:-1], origin="one chunk short")
    with pytest.raises(EvidenceError, match=f"the gold source has {len(good.chunk_ids) - 1} chunks but the run used"):
        compute_evidence(a, gold_sources={"hotpotqa": short})


def test_a_missing_gold_source_is_an_error_with_instructions(real, tmp_path):
    with pytest.raises(EvidenceError, match="prepare_data"):
        compute_evidence(load_analysis_input(real), corpus_root=tmp_path / "none")


# ============================================================================== Random shuffles
@pytest.fixture(scope="module")
def shuffled(tmp_path_factory):
    m = real_matrix(tmp_path_factory.mktemp("shuf"), strategies=("random", "ketrag"), budgets=(0.25, 0.5),
                    random_seeds=[0, 1, 2])
    a = load_analysis_input(m)
    return a, compute_evidence(a, gold_sources=gold_source())


def test_every_shuffle_has_its_own_evidence_rows(shuffled):
    a, rep = shuffled
    assert len(rep.rows) == (3 + 1) * 2 * 3 and rep.reconciled                      # (3 Random + KET-RAG) x 2 budgets x 3 questions
    assert sorted({c.shuffle_seed for c in rep.conditions if c.strategy == "random"}) == [0, 1, 2]


def test_a_random_evidence_arm_has_one_value_per_question(shuffled):
    a, rep = shuffled
    U = a.question_ids["hotpotqa"]
    arm = evidence_arm(rep.rows, dataset="hotpotqa", strategy="random", budget_pct=50, metric="selection_recall", universe=U)
    assert arm.n == 3 == len(U) and arm.replicates == (0, 1, 2) and arm.n_partial == 0
    for q in U:                                                                      # hand-check: the mean over the 3 shuffles
        per = [r.selection_recall for r in rep.rows if r.strategy == "random" and r.budget_pct == 50 and r.question_id == q]
        assert len(per) == 3 and arm.values[q] == pytest.approx(sum(per) / 3)
    ket = evidence_arm(rep.rows, dataset="hotpotqa", strategy="ketrag", budget_pct=50, metric="selection_recall", universe=U)
    assert ket.replicates == (None,) and pair_arms(arm, ket).n_pairs == 3


def test_evidence_arms_feed_the_statistics_unchanged(shuffled):
    a, rep = shuffled
    U = a.question_ids["hotpotqa"]
    r = evidence_arm(rep.rows, dataset="hotpotqa", strategy="random", budget_pct=25, metric="retrieval_recall", universe=U)
    k = evidence_arm(rep.rows, dataset="hotpotqa", strategy="ketrag", budget_pct=25, metric="retrieval_recall", universe=U)
    d = describe_arm(r, n_resamples=200)
    assert d.n == 3 and d.ci.n == 3                                                  # questions, not questions x shuffles
    c = compare_paired(pair_arms(k, r), n_resamples=200)
    assert c.n_pairs <= 3 and c.metric == "retrieval_recall" and "shuffle_averaged_arm" in c.flags


def test_every_strategy_and_budget_reconciles_with_the_runner_on_the_full_mock_matrix(tmp_path):
    """72 mock conditions (2 datasets x 6 budgets x (3 Random shuffles + 3 strategies)): the recomputed evidence equals
    what the runner stored for EVERY condition. Mock data: a layout/plumbing check only. The corpus is rebuilt offline
    from the saved config (data_source mock)."""
    m = mock_matrix(tmp_path, datasets=("hotpotqa", "musique"),
                    strategies=("random", "ketrag", "lazygraphrag", "fastgraphrag"),
                    budgets=(0.05, 0.10, 0.25, 0.50, 0.75, 1.0), random_seeds=[0, 1, 2])
    a = load_analysis_input(m, allow_mock_layout_test=True)
    rep = compute_evidence(a, corpus_root=tmp_path / "no_corpus_dir", allow_rebuild=True)
    assert a.status == "mock_layout_test" and rep.status == "mock_layout_test"
    assert len(rep.conditions) == 72 and set(rep.gold_origin) == {"hotpotqa", "musique"}
    assert all("rebuilt" in o for o in rep.gold_origin.values())
    assert rep.reconciled and rep.n_subset_violations == 0, [i for i in rep.issues if i.severity == "error"][:3]
    for ce in rep.conditions:                                                        # monotone in the budget for every strategy
        assert 0.0 <= ce.metrics["retrieval_recall"].mean <= ce.metrics["selection_recall"].mean <= 1.0 + 1e-12
    for ds in ("hotpotqa", "musique"):
        full = [c for c in rep.conditions if c.dataset == ds and c.budget_pct == 100]
        assert full and all(c.metrics["selection_recall"].mean == 1.0 for c in full)


def rows_for(table, metric_value=None):
    """Hand-built evidence rows for a Random cell: table {shuffle: {question: QuestionEvidence}}."""
    rows = []
    for s, per_q in table.items():
        for q, e in per_q.items():
            rows.append(e.model_copy(update=dict(condition_id=f"r{s}", dataset="d", strategy="random", budget_pct=10,
                                                 seed=1, shuffle_seed=s, question_id=q)))
    return rows


def test_evidence_arm_hand_calculated_missing_handling():
    full = ev({"a", "b"}, {"a", "b"}, {"a"})                       # selection 1.0, retrieval 0.5
    half = ev({"a", "b"}, {"a"}, {"a"})                            # selection 0.5, retrieval 0.5
    nogold = ev(set(), {"a"}, {"a"})
    norev = ev({"a", "b"}, {"a", "b"}, None)                       # selection 1.0, retrieval undefined
    U = ("q1", "q2", "q3")
    rows = rows_for({0: {"q1": full, "q2": nogold, "q3": norev}, 1: {"q1": half, "q2": nogold, "q3": full}})
    arm = lambda m, **kw: evidence_arm(rows, dataset="d", strategy="random", budget_pct=10, metric=m, universe=U, **kw)
    sel = arm("selection_recall")
    assert sel.values == {"q1": 0.75, "q3": 1.0} and sel.missing == {"q2": "empty_gold"} and sel.n == 2
    ret = arm("retrieval_recall")
    assert ret.values == {"q1": 0.5, "q3": 0.5} and ret.k_available["q3"] == 1 and ret.n_partial == 1   # q3 rests on 1 of 2 shuffles
    strict = arm("retrieval_recall", strict=True)
    assert strict.values == {"q1": 0.5} and strict.missing["q3"] == "incomplete_shuffles" and strict.missing["q2"] == "empty_gold"
    assert arm("selection_recall", shuffles=[1]).values == {"q1": 0.5, "q3": 1.0}


def test_evidence_arm_refuses_inconsistent_input():
    e = ev({"a"}, {"a"}, {"a"})
    U = ("q1",)
    rows = rows_for({0: {"q1": e}})
    with pytest.raises(ArmError, match="unknown evidence metric"):
        evidence_arm(rows, dataset="d", strategy="random", budget_pct=10, metric="f1", universe=U)
    with pytest.raises(ArmError, match="no evidence rows"):
        evidence_arm(rows, dataset="d", strategy="ketrag", budget_pct=10, metric="selection_recall", universe=U)
    with pytest.raises(ArmError, match="not in the dataset's question set"):
        evidence_arm(rows, dataset="d", strategy="random", budget_pct=10, metric="selection_recall", universe=("q9",))
    with pytest.raises(ArmError, match="appears twice"):
        evidence_arm(rows + rows, dataset="d", strategy="random", budget_pct=10, metric="selection_recall", universe=U)


# ============================================================================== gold is post hoc only
def test_evidence_imports_no_pipeline_code():
    """The module that reads gold must not be able to reach the code that selects, extracts, retrieves or generates."""
    tree = ast.parse((PROJECT / "analysis" / "evidence.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("strategies", "extraction", "retrieval", "generation", "graph", "native", "experiments.runner",
                 "experiments.run", "evaluation.judge")
    assert not [m for m in imported if any(m == f or m.startswith(f + ".") for f in forbidden)], imported


def test_no_phase_1_to_11_code_imports_the_analysis_package():
    pattern = re.compile(r"^\s*(from|import)\s+analysis(\.|\s|$)", re.M)
    offenders = []
    for folder in ("aggregation", "evaluation", "experiments", "extraction", "generation", "graph", "native",
                   "retrieval", "src", "strategies"):
        for py in (PROJECT / folder).rglob("*.py"):
            if pattern.search(py.read_text(encoding="utf-8")):
                offenders.append(str(py.relative_to(PROJECT)))
    assert offenders == []                                                          # nothing in the pipeline can read analysis


def test_computing_evidence_changes_no_file(real_copy, tmp_path):
    a = load_analysis_input(real_copy)
    corpus_dir = tmp_path / "corpus" / a.conditions[0].corpus_id                     # what src.prepare_data would have written
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "chunks.json").write_text(json.dumps([c.to_dict() for c in MOCK_CHUNKS]), encoding="utf-8")
    before = (file_hashes(real_copy), file_hashes(tmp_path / "corpus"))
    rep = compute_evidence(a, corpus_root=tmp_path / "corpus")
    assert rep.rows and rep.reconciled and "chunks.json" in rep.gold_origin["hotpotqa"]       # found on disk, not rebuilt
    assert (file_hashes(real_copy), file_hashes(tmp_path / "corpus")) == before


def test_the_pipeline_never_sees_gold_while_evidence_still_finds_it(tmp_path):
    """Run the real runner with spies on every stage. No stage receives a gold label; afterwards the evidence
    metrics (which need gold) are computed from the corpus on disk-independent labels."""
    events = []
    comps = spy_components(
        events, build_extractor=lambda cfg, dry_run=False: SpyExtractor(PricedExtractor(), events),
        answerer=lambda cfg: TimedAnswerer(fake_answerer(handler=lambda k: answer_reply("Yes"))),
        judge=lambda cfg: TimedJudge(fake_judge(handler=lambda k: judge_verdict(True))), load_corpus=mock_corpus)
    from tests.helpers_analysis import build_matrix
    m = build_matrix(tmp_path, mock=False, comps=comps, strategies=("random", "ketrag"), budgets=(0.5,))
    seen = [e for e in events if e[0] in ("rank", "make_retriever")]
    assert seen and all(not any(labels) for e in seen for labels in (e[-1],))        # every chunk copy had its labels stripped
    answers = [e for e in events if e[0] == "answer"]
    assert answers and all(e[1:3] == ("str", "str") for e in answers)                # only question text and context text
    extracts = [e for e in events if e[0] == "extract"]
    assert extracts and all(e[1] == "ExtractionInput" for e in extracts)            # the extractor got {chunk_id, text} only
    rep = compute_evidence(load_analysis_input(m), gold_sources=gold_source())
    assert any(r.n_gold > 0 for r in rep.rows) and rep.reconciled                  # gold exists, and only the analysis used it
