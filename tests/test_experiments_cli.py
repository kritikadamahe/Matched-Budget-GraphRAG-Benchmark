"""Phase 10 runner: the command line (python -m experiments.run). Mock-backed; no API call, no key."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pytest
import yaml

from experiments.run import main, parse_budget


def write_cfg(tmp_path, matrix=None, name="bench.yaml", **kw) -> Path:
    cfg = dict(experiment_id="bench", seed=42, dataset="hotpotqa", strategy="random", budget=0.1, data_source="mock",
               num_questions=3, embedding_backend="tfidf", ketrag_mode="tfidf", fast_use_spacy=False,
               extraction_backend="mock", generation_backend="mock", judge_backend="mock",
               cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "out"))
    cfg.update(kw)
    if matrix is not None:
        cfg["matrix"] = matrix
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def matrix_dir(tmp_path) -> Path:
    return tmp_path / "out" / "experiments" / "bench"


def rows(tmp_path):
    return [json.loads(l) for l in (matrix_dir(tmp_path) / "results.jsonl").read_text().splitlines()]


SMALL = ["--datasets", "hotpotqa", "--strategies", "random", "--budgets", "25,100"]


# --------------------------------------------------------------------- budgets
@pytest.mark.parametrize("token,value", [("5", 0.05), ("5%", 0.05), ("0.05", 0.05), ("25", 0.25), ("100", 1.0),
                                         ("100%", 1.0), ("1", 1.0), ("0.5", 0.5), (" 10 ", 0.10)])
def test_budget_tokens(token, value):
    assert parse_budget(token) == pytest.approx(value)


# --------------------------------------------------------------------- dry run
def test_dry_run_lists_all_48_conditions_and_writes_nothing(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "48 condition(s)" in out
    assert len([l for l in out.splitlines() if "__seed42" in l]) == 48
    assert "hotpotqa__random__b005__seed42" in out and "musique__fastgraphrag__b100__seed42" in out
    assert "all mock" in out
    assert not (tmp_path / "out").exists() and not (tmp_path / "cache").exists()


def test_dry_run_of_a_real_config_needs_no_key_no_network_and_warns_about_cost(tmp_path, capsys, no_network, no_api_key):
    cfg = write_cfg(tmp_path, extraction_backend="openai", generation_backend="openai", judge_backend="openai",
                    matrix={"datasets": ["hotpotqa"], "strategies": ["random"], "budgets": [0.5, 1.0]})
    assert main(["--config", str(cfg), "--dry-run", "--max-total-usd", "0.0000001"]) == 0
    out = capsys.readouterr().out
    assert "--confirm-paid" in out and "extraction (gpt-4o-mini)" in out
    assert "ABOVE the limit" in out                                      # the estimate is compared with the cap
    assert not (tmp_path / "out").exists()


def test_dry_run_respects_the_selection_flags(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg), "--dry-run", "--datasets", "musique", "--strategies", "ketrag", "--budgets", "5%,50%"]) == 0
    out = capsys.readouterr().out
    assert "2 condition(s)" in out and "musique__ketrag__b005__seed42" in out and "musique__ketrag__b050__seed42" in out


# --------------------------------------------------------------------- paid-run safety
def test_a_run_with_real_backends_refuses_to_start_without_confirm_paid(tmp_path, capsys, no_network, no_api_key):
    cfg = write_cfg(tmp_path, generation_backend="openai")
    assert main(["--config", str(cfg)]) == 2
    err = capsys.readouterr().err
    assert "STOPPED" in err and "generation (gpt-4o-mini)" in err and "--confirm-paid" in err and "--dry-run" in err
    assert not (tmp_path / "out").exists()                                # nothing was run or written


def test_the_full_paid_matrix_is_never_started_by_accident(tmp_path, capsys, no_network, no_api_key):
    cfg = write_cfg(tmp_path, extraction_backend="openai", generation_backend="openai", judge_backend="openai")
    assert main(["--config", str(cfg)]) == 2 and not (tmp_path / "out").exists()
    assert main(["--config", str(cfg), "--no-judge"]) == 2                # still paid (extraction + generation)


def test_a_judge_only_config_is_free_when_the_judge_is_switched_off(tmp_path, capsys):
    cfg = write_cfg(tmp_path, judge_backend="openai")
    assert main(["--config", str(cfg), "--no-judge"] + SMALL) == 0        # nothing paid is left: allowed without --confirm-paid
    assert rows(tmp_path)[0]["metrics"]["judge_enabled"] is False
    assert main(["--config", str(cfg)] + SMALL) == 2                      # with the judge it is paid: refused


def test_confirmed_paid_run_without_a_key_stops_before_any_condition_and_never_uses_the_mock(
        tmp_path, capsys, no_network, no_api_key, monkeypatch):
    import extraction.openai_extractor as ox
    import generation.llm_client as lc
    monkeypatch.setattr(ox, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(lc, "load_dotenv", lambda *a, **k: False)
    cfg = write_cfg(tmp_path, extraction_backend="openai")
    assert main(["--config", str(cfg), "--confirm-paid"] + SMALL) == 2
    err = capsys.readouterr().err
    assert "STOPPED before running anything" in err and "NOT fall back" in err
    assert not (matrix_dir(tmp_path) / "conditions").exists() and not (matrix_dir(tmp_path) / "results.jsonl").exists()


# --------------------------------------------------------------------- a small mock run
def test_small_mock_run_writes_the_expected_outputs(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg)] + SMALL) == 0
    out = capsys.readouterr().out
    assert "ran 2 condition(s) {'completed': 2}" in out and "MOCK RESULTS" in out and "NOT reportable" in out
    r = rows(tmp_path)
    assert [x["condition_id"] for x in r] == ["hotpotqa__random__b025__seed42", "hotpotqa__random__b100__seed42"]
    assert [(x["n_selected_expected"], x["n_selected"]) for x in r] == [(2, 2), (8, 8)]
    assert all(x["status"] == "completed" and x["is_mock"] and not x["reportable"] for x in r)
    manifest = json.loads((matrix_dir(tmp_path) / "matrix.json").read_text())
    assert manifest["n_conditions"] == 48                                  # the manifest lists the whole matrix, not the subset
    for cid in ("hotpotqa__random__b025__seed42", "hotpotqa__random__b100__seed42"):
        d = matrix_dir(tmp_path) / "conditions" / cid
        assert {p.name for p in d.iterdir()} == {"condition_result.json", "predictions.jsonl", "scored.jsonl",
                                                  "eval_summary.json", "extraction", "graph"}


def test_out_dir_overrides_the_default_location(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg), "--out-dir", str(tmp_path / "elsewhere")] + SMALL) == 0
    assert (tmp_path / "elsewhere" / "results.jsonl").exists() and not (tmp_path / "out").exists()


def test_condition_flag_is_repeatable_and_limit_truncates(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg), "--condition", "hotpotqa__random__b005__seed42",
                 "--condition", "musique__ketrag__b050__seed42"]) == 0
    assert [x["condition_id"] for x in rows(tmp_path)] == ["hotpotqa__random__b005__seed42", "musique__ketrag__b050__seed42"]
    assert main(["--config", str(cfg), "--limit", "3", "--out-dir", str(tmp_path / "lim")]) == 0
    assert len((tmp_path / "lim" / "results.jsonl").read_text().splitlines()) == 3


def test_no_judge_runs_em_and_f1_only(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg), "--no-judge"] + SMALL) == 0
    r = rows(tmp_path)[0]
    assert r["metrics"]["judge_enabled"] is False and "mock judge backend" not in r["not_reportable_reasons"]
    assert r["cost"]["stages_included"] == ["extraction", "generation"]


def test_max_total_usd_does_not_stop_a_free_run(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg), "--max-total-usd", "0.01"] + SMALL) == 0
    assert len(rows(tmp_path)) == 2


# --------------------------------------------------------------------- resume / force / stale
def test_rerunning_the_same_command_resumes_without_duplicating_results(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg)] + SMALL) == 0
    capsys.readouterr()
    assert main(["--config", str(cfg)] + SMALL) == 0
    assert "ran 0 condition(s) {}; skipped (already completed): 2" in capsys.readouterr().out
    assert len(rows(tmp_path)) == 2
    assert main(["--config", str(cfg), "--force"] + SMALL) == 0
    assert "ran 2 condition(s)" in capsys.readouterr().out and len(rows(tmp_path)) == 2


def test_resuming_with_a_wider_selection_only_runs_the_new_conditions(tmp_path, capsys):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg)] + SMALL) == 0
    capsys.readouterr()
    assert main(["--config", str(cfg), "--datasets", "hotpotqa", "--strategies", "random", "--budgets", "5,25,100"]) == 0
    out = capsys.readouterr().out
    assert "ran 1 condition(s) {'completed': 1}" in out and "skipped (already completed): 2" in out
    assert [x["budget_pct"] for x in rows(tmp_path)] == [5, 25, 100]            # matrix order, one row each


def test_a_changed_config_is_reported_as_stale_not_overwritten(tmp_path, capsys):
    assert main(["--config", str(write_cfg(tmp_path))] + SMALL) == 0
    before = (matrix_dir(tmp_path) / "conditions" / "hotpotqa__random__b025__seed42" / "condition_result.json").read_bytes()
    capsys.readouterr()
    changed = write_cfg(tmp_path, name="changed.yaml", retrieval_top_m=7)
    assert main(["--config", str(changed)] + SMALL) == 1
    err = capsys.readouterr().err
    assert "STALE" in err and "--force" in err
    after = (matrix_dir(tmp_path) / "conditions" / "hotpotqa__random__b025__seed42" / "condition_result.json").read_bytes()
    assert after == before


# --------------------------------------------------------------------- bad input
@pytest.mark.parametrize("argv", [
    ["--datasets", "squad"], ["--strategies", "nope"], ["--budgets", "7"], ["--condition", "x__y"], ["--limit", "0"],
    ["--condition", "musique__random__b005__seed42", "--datasets", "hotpotqa"],       # selection is empty
    ["--max-total-usd", "-1"],
])
def test_bad_selections_stop_with_exit_2_and_run_nothing(tmp_path, capsys, argv):
    cfg = write_cfg(tmp_path)
    assert main(["--config", str(cfg)] + argv) == 2
    assert "STOPPED" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_a_missing_config_or_unsafe_id_stops(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml")]) == 2
    assert main(["--config", str(write_cfg(tmp_path, experiment_id="a/b"))]) == 2
    assert main(["--config", str(write_cfg(tmp_path, matrix={"budgets": [0.123]}))]) == 2
    assert main(["--config", str(write_cfg(tmp_path, matrix={"strategies": ["lazygraphrag_native"]}))]) == 2
    assert not (tmp_path / "out").exists()


def test_an_invalid_condition_config_is_caught_by_the_preflight(tmp_path, capsys):
    cfg = write_cfg(tmp_path, ketrag_mode="faithful", matrix={"strategies": ["ketrag"], "datasets": ["hotpotqa"], "budgets": [0.5]})
    assert main(["--config", str(cfg)]) == 2
    assert "faithful" in capsys.readouterr().err and not (matrix_dir(tmp_path) / "conditions").exists()


def test_the_shipped_mock_config_dry_runs_to_48_conditions(capsys):
    root = Path(__file__).resolve().parents[1]
    assert main(["--config", str(root / "configs" / "benchmark_mock.yaml"), "--dry-run"]) == 0
    assert "48 condition(s)" in capsys.readouterr().out
