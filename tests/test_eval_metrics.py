"""Phase 8 Evaluation: answer normalisation, Exact Match, token F1, aliases, yes/no."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from evaluation.metrics import best_scores, exact_match, f1_score
from evaluation.normalize import normalize_answer
from generation.schemas import clean_answer


# ---------------------------------------------------------------- normalisation
@pytest.mark.parametrize("raw,expected", [
    ("The Beatles", "beatles"),
    ("  New   York  City ", "new york city"),
    ("Washington, D.C.", "washington dc"),
    ("U.S.", "us"),
    ("An apple a day", "apple day"),
    ("O\u2019Neil", "oneil"),                 # curly apostrophe behaves like the ASCII one
    ("1,000", "1000"),
    ("\uff21\uff22\uff23", "abc"),            # full-width letters -> NFKC
    ("", ""),
    ("...", ""),
])
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalisation_is_not_clean_answer():
    # clean_answer keeps case and articles; scoring must not rely on it.
    assert clean_answer("The Beatles.") == "The Beatles"
    assert normalize_answer(clean_answer("The Beatles.")) == "beatles"


# ----------------------------------------------------------------- exact match
def test_exact_match_ignores_case_articles_and_punctuation():
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert exact_match("Eiffel Tower", "Eiffel Towers") == 0.0


# -------------------------------------------------------------------- token F1
def test_f1_exact_partial_and_zero():
    assert f1_score("New York City", "new york city") == 1.0
    assert f1_score("New York", "New York City") == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
    assert f1_score("Paris", "London") == 0.0


def test_f1_counts_repeated_tokens_as_a_bag():
    # pred has "a b b", gold "a b": overlap = 2 (a, b once each), precision 2/3, recall 1
    assert f1_score("alpha beta beta", "alpha beta") == pytest.approx(0.8)


def test_f1_both_empty_is_a_match_one_empty_is_not():
    assert f1_score("", "") == 1.0
    assert f1_score("", "Chicago") == 0.0
    assert f1_score("Chicago", "") == 0.0


# --------------------------------------------------------------------- aliases
def test_best_over_gold_and_aliases():
    golds = ["United States of America", "USA", "US"]
    assert best_scores("U.S.", golds) == (1.0, 1.0)            # matches the alias "US"
    em, f1 = best_scores("United States", golds)
    assert em == 0.0 and 0.0 < f1 < 1.0                       # best partial overlap


def test_em_is_order_sensitive_but_f1_is_a_bag_of_tokens():
    em, f1 = best_scores("alpha beta", ["alpha beta gamma", "beta alpha"])
    assert em == 0.0 and f1 == 1.0              # F1 maximised over the aliases independently of EM


def test_best_scores_needs_a_gold_answer():
    with pytest.raises(ValueError):
        best_scores("x", [])
    with pytest.raises(ValueError):
        best_scores("x", ["", "   "])


# -------------------------------------------------------------------- yes / no
@pytest.mark.parametrize("pred,gold,em,f1", [
    ("Yes.", "yes", 1.0, 1.0),
    ("no", "No", 1.0, 1.0),
    ("yes", "no", 0.0, 0.0),
    ("yes, it is", "yes", 0.0, 0.0),        # no partial credit for yes/no (official HotpotQA rule)
    ("yes", "yes it did", 0.0, 0.0),
])
def test_yes_no_has_no_partial_credit(pred, gold, em, f1):
    assert (exact_match(pred, gold), f1_score(pred, gold)) == (em, f1)
