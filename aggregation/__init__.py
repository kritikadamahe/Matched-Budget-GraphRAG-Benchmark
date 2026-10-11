"""Cost / runtime aggregation (Phase 11): turns the Phase 10 results into per-condition and per-group
cost, token and runtime summaries for Phase 12.

    aggregation/schemas.py    Measure (measured / lower_bound / unknown / not_run / disabled / not_applicable),
                              ConditionRecord, GroupSummary, AggregateReport
    aggregation/loader.py     finds the results; duplicates and mixed configs are errors unless opted in
    aggregation/measures.py   one flat record per condition; tokens / runtimes rebuilt from the per-item files;
                              cross-checks against what the runner recorded
    aggregation/summarize.py  group statistics with sample counts
    aggregation/aggregate.py  CLI: python -m aggregation.aggregate --matrix-dir DIR

It reads results; it runs nothing, calls no API and plots nothing (Phase 12).
"""

from aggregation.loader import AggregationError
from aggregation.schemas import AggregateReport, ConditionRecord, GroupSummary, Measure

__all__ = ["AggregateReport", "AggregationError", "ConditionRecord", "GroupSummary", "Measure",
           "aggregate_matrices", "assert_reportable", "write_outputs"]

_LAZY = ("aggregate_matrices", "assert_reportable", "write_outputs")


def __getattr__(name):
    # lazy: importing the package must not import the CLI module (python -m aggregation.aggregate)
    if name in _LAZY:
        from aggregation import aggregate
        return getattr(aggregate, name)
    raise AttributeError(name)
