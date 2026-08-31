"""
Category B -- live model semantic-reliability MEASUREMENT, not a
correctness gate. Unlike test_structured_sql_builder.py's Category A (which
proves canonical-plan -> byte-identical-SQL unconditionally, no LLM
involved), these tests make real calls to the configured LLM (Ollama/
llama3.2 in this deployment) and report how often it chooses the *correct*
plan for a fixed set of paraphrases -- a quality metric that varies with the
model and prompt, not a guarantee this codebase can make.

Requires a live Ollama server with llama3.2 pulled and reachable at
OLLAMA_BASE_URL (defaults to http://localhost:11434). Skips gracefully if
unavailable, so it never blocks a CI run that has no live model configured.

Run directly for a human-readable report:
    DATABASE_URL="" ./.venv/bin/python3 -m pytest tests/test_live_structured_reliability.py -v -s
"""

import os

import httpx
import pytest

from src.agents.intent_agent import IntentResolutionAgent
from src.agents.query_normalizer import normalize
from src.agents.query_plan import Entity, ExtremeSelector, GroupingDimension, Operation, SortField
from src.agents.query_validator import QueryPlanValidationError, QueryPlanValidator
from src.retrieval.structured_sql_builder import StructuredSQLBuilder

PARAPHRASES = [
    "How many students are in each class?",
    "Count students by class",
    "Class sizes",
    "Students per class",
]
RUNS_PER_PARAPHRASE = 10
EXPECTED = {"entity": Entity.STUDENTS, "operation": Operation.COUNT, "group_by": GroupingDimension.BY_CLASS}


def _ollama_available() -> bool:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        return httpx.get(f"{base_url}/api/version", timeout=2.0).status_code == 200
    except Exception:
        return False


class _NoOpDB:
    def execute(self, sql):
        return []


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_live_structured_reliability_report():
    agent = IntentResolutionAgent()
    validator = QueryPlanValidator(_NoOpDB())

    results = {}
    for paraphrase in PARAPHRASES:
        valid_count = 0
        correct_count = 0
        sqls = set()
        for _ in range(RUNS_PER_PARAPHRASE):
            try:
                plan = await agent.resolve_structured(paraphrase, context={"school_id": 56})
            except Exception:
                continue  # technical failure this run -- not counted as valid
            if not plan.can_answer:
                continue  # model declined -- not counted as valid
            try:
                resolved = validator.validate(plan, school_id=56)
            except QueryPlanValidationError:
                continue  # invalid plan -- correctly caught, not counted as valid
            valid_count += 1
            canonical = normalize(plan, resolved)
            is_correct = (
                canonical.entity == EXPECTED["entity"]
                and canonical.operation == EXPECTED["operation"]
                and canonical.group_by == EXPECTED["group_by"]
            )
            if is_correct:
                correct_count += 1
                sqls.add(StructuredSQLBuilder.build(canonical))

        results[paraphrase] = {
            "valid_rate": valid_count / RUNS_PER_PARAPHRASE,
            "correct_rate": correct_count / RUNS_PER_PARAPHRASE,
            "distinct_sql_among_correct": len(sqls),
        }

    print("\n\n=== Structured intent live reliability report ===")
    print(f"{'Paraphrase':<40} {'Valid%':>8} {'Correct%':>10} {'Distinct SQL':>14}")
    for paraphrase, r in results.items():
        print(f"{paraphrase:<40} {r['valid_rate']*100:>7.0f}% {r['correct_rate']*100:>9.0f}% {r['distinct_sql_among_correct']:>14}")

    # Non-gating in spirit (a low correct_rate is a prompt-quality signal to
    # act on, not a build failure) -- but SQL-consistency among the runs
    # that DID land on the correct semantic plan must always be 1, since
    # that's Category A's determinism guarantee re-confirmed under real
    # model output, not a probabilistic property.
    for paraphrase, r in results.items():
        assert r["distinct_sql_among_correct"] <= 1, (
            f"{paraphrase!r}: multiple correct-plan runs produced DIFFERENT SQL -- "
            f"this would be a real determinism bug, not model variance."
        )


# ── Extreme-value / ranking reliability measurement ─────────────────────────
#
# Distinguishes "lowest/highest with no number" (must resolve to extreme=...,
# no sort/limit) from "explicit N" (must resolve to sort=aggregate_value +
# the user's own stated limit) -- see query_plan.py's ExtremeSelector and
# SortField.AGGREGATE_VALUE docstrings. Measures the actual end-user
# experience: one retry-with-feedback pass, exactly as
# query_lifecycle.py's _try_structured_resolution performs, not raw
# first-shot output.

RANKING_PARAPHRASES = {
    "lowest_no_number": [
        "Which students have the lowest attendance?",
        "Who has the lowest attendance?",
        "Which student has the worst attendance record?",
    ],
    "highest_no_number": [
        "Who has the highest attendance?",
        "Which students have the best attendance?",
    ],
    "explicit_n": [
        "Show the 5 students with the lowest attendance.",
        "List the 3 students with the highest attendance.",
    ],
}
RANKING_RUNS_PER_PARAPHRASE = 5


def _wants_lowest_attendance(canonical) -> bool:
    """"Lowest attendance" means the lowest attendance rate, defined as the
    percentage of records with status=present -- and ONLY that. A prior
    version of this oracle also
    accepted "highest ABSENT percentage" as an equivalent encoding; that was
    incorrect and has been removed (see
    tests/test_metric_semantics.py::test_lowest_present_rate_and_highest_absent_rate_can_disagree_on_the_ranking
    for a concrete demonstration) -- STATUS has FOUR members (present/
    absent/late/excused), not two, so %present and %absent are only
    complements when late/excused are zero for every student being
    compared, which nothing guarantees. Accepting the absent-based form
    here would silently let a materially different ranking pass as
    "correct"."""
    numerator = canonical.percentage_of.numerator
    return canonical.extreme == ExtremeSelector.LOWEST and numerator.value == "present"


def _wants_highest_attendance(canonical) -> bool:
    """Mirror of _wants_lowest_attendance -- see its docstring for why this
    is present-only, not also accepting an absent-based inversion."""
    numerator = canonical.percentage_of.numerator
    return canonical.extreme == ExtremeSelector.HIGHEST and numerator.value == "present"


def _is_correct_ranking_plan(canonical, category: str) -> bool:
    base_ok = (
        canonical.entity == Entity.ATTENDANCE
        and canonical.operation == Operation.PERCENTAGE
        and canonical.group_by == GroupingDimension.BY_STUDENT
        and canonical.percentage_of is not None
    )
    if not base_ok:
        return False
    if category == "lowest_no_number":
        return _wants_lowest_attendance(canonical) and canonical.sort is None and canonical.limit is None
    if category == "highest_no_number":
        return _wants_highest_attendance(canonical) and canonical.sort is None and canonical.limit is None
    # explicit_n: sort=aggregate_value + a real, model-stated limit (never a
    # fixed expected number -- the paraphrase's own stated count is what
    # must survive, whatever it is) -- and, like the no-number cases, the
    # numerator must be present-based, not an absent-based inversion.
    return (
        canonical.extreme is None
        and canonical.sort is not None
        and canonical.sort.field == SortField.AGGREGATE_VALUE
        and canonical.limit is not None
        and canonical.percentage_of.numerator.value == "present"
    )


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_live_ranking_reliability_report():
    agent = IntentResolutionAgent()
    validator = QueryPlanValidator(_NoOpDB())

    results = {}
    for category, paraphrases in RANKING_PARAPHRASES.items():
        for paraphrase in paraphrases:
            valid_count = 0
            correct_count = 0
            for _ in range(RANKING_RUNS_PER_PARAPHRASE):
                try:
                    plan = await agent.resolve_structured(paraphrase, context={"school_id": 56})
                except Exception:
                    continue
                if not plan.can_answer:
                    continue
                try:
                    resolved = validator.validate(plan, school_id=56)
                except QueryPlanValidationError as exc:
                    # Exactly one retry-with-feedback, matching
                    # query_lifecycle.py's actual fallback behavior.
                    try:
                        plan = await agent.resolve_structured_with_feedback(paraphrase, {"school_id": 56}, str(exc))
                    except Exception:
                        continue
                    if not plan.can_answer:
                        continue
                    try:
                        resolved = validator.validate(plan, school_id=56)
                    except QueryPlanValidationError:
                        continue  # fails closed to clarification in production -- never wrong SQL
                valid_count += 1
                canonical = normalize(plan, resolved)
                if _is_correct_ranking_plan(canonical, category):
                    correct_count += 1

            results[(category, paraphrase)] = {
                "valid_rate": valid_count / RANKING_RUNS_PER_PARAPHRASE,
                "correct_rate": correct_count / RANKING_RUNS_PER_PARAPHRASE,
            }

    print("\n\n=== Extreme-value/ranking live reliability report (informational, non-gating) ===")
    print(f"{'Category':<20} {'Paraphrase':<45} {'Valid%':>8} {'Correct%':>10}")
    for (category, paraphrase), r in results.items():
        print(f"{category:<20} {paraphrase:<45} {r['valid_rate']*100:>7.0f}% {r['correct_rate']*100:>9.0f}%")

    # Purely informational -- no assertion on correct_rate. A low rate here
    # is llama3.2's already-documented free-text/semantic non-determinism
    # (see docs/memory: don't chase this with more few-shots without an
    # explicit decision to do so) surfacing through the NEW ranking
    # vocabulary too; the validator's job (proven in test_query_validator.py
    # and test_structured_sql_builder.py) is to fail closed to a
    # clarification on every wrong plan, never to emit incorrect SQL -- and
    # it does, unconditionally, regardless of how often the model gets there.
