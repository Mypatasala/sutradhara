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
from src.agents.query_lifecycle import QueryLifecycleAgent
from src.agents.query_normalizer import normalize
from src.agents.query_plan import Entity, ExtremeSelector, GroupingDimension, Operation, RelativeDate, SortField
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


# ── Plain-total vs BY_CLASS grouping reliability (Part A1 regression) ───────
#
# Root cause investigation (2026-09-01): "How many students are in my
# school?" deterministically (4/4 live runs, BEFORE this fix) resolved to
# entity=students, group_by=by_class -- the structured prompt stated the
# correct "leave group_by unset for a plain total" rule in prose, but had NO
# worked example of an ungrouped STUDENTS count, while every existing
# STUDENTS/group_by worked example demonstrated the GROUPED case. Fixed by
# adding contrastive worked examples to _STRUCTURED_SYSTEM_PROMPT. Unlike
# the informational reports above, this is a real regression gate: the
# reported incident showed this failure mode was 100% deterministic at the
# model's configured temperature, not run-to-run noise, so a fix that
# actually works must also reproduce as 100% deterministic, not a
# probabilistic improvement.

PLAIN_TOTAL_PARAPHRASES = [
    "How many students are in my school?",
    "How many students are there?",
]
PLAIN_TOTAL_RUNS_PER_PARAPHRASE = 4


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_plain_student_count_never_gains_group_by():
    """Regression gate for the exact reported incident: a plain "how many
    students" question, with no per-class breakdown requested, must resolve
    to group_by=NONE on every run -- never silently default to BY_CLASS just
    because the entity happens to support that grouping."""
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for paraphrase in PLAIN_TOTAL_PARAPHRASES:
        for i in range(PLAIN_TOTAL_RUNS_PER_PARAPHRASE):
            plan = await agent.resolve_structured(paraphrase, context)
            assert plan.can_answer, f"{paraphrase!r} run {i+1}: model declined to answer"
            assert plan.entity == Entity.STUDENTS, f"{paraphrase!r} run {i+1}: entity={plan.entity}"
            assert plan.operation == Operation.COUNT, f"{paraphrase!r} run {i+1}: operation={plan.operation}"
            assert plan.group_by == GroupingDimension.NONE, (
                f"{paraphrase!r} run {i+1}: group_by={plan.group_by} -- a plain total question "
                f"must never gain a per-class breakdown grouping it didn't ask for."
            )


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_students_in_each_class_still_retains_by_class_grouping():
    """The contrastive worked example added for the fix above must not
    overcorrect the pre-existing, already-correct grouped case in the
    opposite direction."""
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for i in range(PLAIN_TOTAL_RUNS_PER_PARAPHRASE):
        plan = await agent.resolve_structured("How many students are in each class?", context)
        assert plan.can_answer, f"run {i+1}: model declined to answer"
        assert plan.entity == Entity.STUDENTS
        assert plan.operation == Operation.COUNT
        assert plan.group_by == GroupingDimension.BY_CLASS, (
            f"run {i+1}: group_by={plan.group_by} -- an explicit per-class breakdown request "
            f"must still retain BY_CLASS grouping."
        )


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_teacher_count_question_has_no_group_by_regression():
    """USERS has no supported_groupings at all, so this should be
    structurally unaffected by the fix above -- confirmed live, not just
    assumed, since the fix touched a shared section of the prompt."""
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for i in range(PLAIN_TOTAL_RUNS_PER_PARAPHRASE):
        plan = await agent.resolve_structured("How many teachers are in my school?", context)
        assert plan.can_answer, f"run {i+1}: model declined to answer"
        assert plan.entity == Entity.USERS, f"run {i+1}: entity={plan.entity}"
        assert plan.operation == Operation.COUNT
        assert plan.group_by == GroupingDimension.NONE


# ── "How many classes are there?" (Part A2 -- Outcome B) ────────────────
#
# Investigation finding (2026-09-01): prompt-only attempts to route "how
# many classes" to unresolved_reason=out_of_scope did NOT converge (three
# escalating prompt interventions each tested 4/4 deterministic against the
# live model, all still landing on entity=students/group_by=by_class).
#
# Resolved (2026-09-02) via Outcome B: a first-class Entity.SCHOOL_CLASSES
# was registered (COUNT-only, direct `school_classes` table, no joins --
# see query_registry.py). Product-terminology investigation confirmed "class"
# means the grade level alone (school_classes) in myPatasala's own UI/API
# vocabulary ("Select Class" / "Select Section" as distinct controls), never
# a grade+section combination -- so this entity, not class_sections, is what
# "how many classes" must resolve to. An isolated live experiment (adding
# the entity's bullet + a contrastive worked-example block to a full copy of
# the real prompt, verified byte-identical to production after integration)
# proved 100% deterministic correct entity selection across 6 runs each for
# "how many classes are there?", "how many classes are in my school?", plus
# no regression on "how many students are in each class?", "how many
# students are in my school?", teacher counts, and ranking questions --
# before this was implemented in production.
#
# The one hard invariant from the original investigation still holds and is
# still the thing asserted below: a "how many classes" question must NEVER
# resolve to entity=students, group_by=by_class and be silently presented as
# a class count (a different number, students-per-class, mislabeled).

@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_how_many_classes_resolves_to_the_school_classes_entity():
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for paraphrase in ["How many classes are there?", "How many classes are in my school?"]:
        for i in range(PLAIN_TOTAL_RUNS_PER_PARAPHRASE):
            plan = await agent.resolve_structured(paraphrase, context)
            assert plan.can_answer, f"{paraphrase!r} run {i+1}: model declined to answer"
            assert plan.entity == Entity.SCHOOL_CLASSES, (
                f"{paraphrase!r} run {i+1}: entity={plan.entity} -- expected the dedicated "
                f"school_classes entity, never a substitute."
            )
            assert plan.operation == Operation.COUNT
            is_students_by_class_substitute = (
                plan.entity == Entity.STUDENTS and plan.group_by == GroupingDimension.BY_CLASS
            )
            assert not is_students_by_class_substitute, (
                f"{paraphrase!r} run {i+1}: resolved to entity=students, group_by=by_class -- "
                f"this answers 'how many students per class', a DIFFERENT number, and must not "
                f"be presented as a class count."
            )


# ── ATTENDANCE: "show me attendance" (Part B) + LAST_30_DAYS (Part A) ──────
#
# Investigation finding (2026-09-02): "Show me attendance for the last 30
# days" deterministically failed closed to a generic clarification --
# ATTENDANCE had no LIST-capable shape at all (structurally unanswerable,
# not model unreliability), and separately "last 30 days" had no matching
# RelativeDate member and silently resolved to LAST_WEEK even for the
# already-working COUNT/PERCENTAGE variants -- a silent correctness bug.
# Resolved by registering Operation.LIST for ATTENDANCE (registry-owned,
# minimal display shape: student name, date, status) and adding
# RelativeDate.LAST_30_DAYS (a true rolling 30-day window). A third,
# related defect was also traced and fixed here: a plain COUNT with no
# status named was live-observed inventing an unrequested
# filters=[{"field": "status", "value": "present"}] -- gated below too.

PLAIN_TOTAL_RUNS = 4


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_show_attendance_last_30_days_resolves_to_list_with_correct_date_range():
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for paraphrase in ["Show me attendance for the last 30 days.", "Show attendance for the last 30 days."]:
        for i in range(PLAIN_TOTAL_RUNS):
            plan = await agent.resolve_structured(paraphrase, context)
            assert plan.can_answer, f"{paraphrase!r} run {i+1}: model declined to answer"
            assert plan.entity == Entity.ATTENDANCE, f"{paraphrase!r} run {i+1}: entity={plan.entity}"
            assert plan.operation == Operation.LIST, f"{paraphrase!r} run {i+1}: operation={plan.operation}"
            assert plan.date_range == RelativeDate.LAST_30_DAYS, (
                f"{paraphrase!r} run {i+1}: date_range={plan.date_range} -- must never silently "
                f"substitute last_week or any other window for an explicit 'last 30 days'."
            )


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_show_attendance_this_month_still_uses_this_month_not_last_30_days():
    """The new LAST_30_DAYS member must not overcorrect an unrelated,
    already-correct date phrase in the opposite direction."""
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for i in range(PLAIN_TOTAL_RUNS):
        plan = await agent.resolve_structured("Show me attendance this month.", context)
        assert plan.can_answer, f"run {i+1}: model declined to answer"
        assert plan.entity == Entity.ATTENDANCE
        assert plan.date_range == RelativeDate.THIS_MONTH, f"run {i+1}: date_range={plan.date_range}"


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_attendance_count_last_30_days_never_invents_a_status_filter():
    """Regression gate for the exact previously-observed defect: a plain
    COUNT of attendance records with no status named must resolve with
    filters=[] -- never a silently-invented status=present."""
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for i in range(PLAIN_TOTAL_RUNS):
        plan = await agent.resolve_structured("How many attendance records are there in the last 30 days?", context)
        assert plan.can_answer, f"run {i+1}: model declined to answer"
        assert plan.entity == Entity.ATTENDANCE
        assert plan.operation == Operation.COUNT
        assert plan.date_range == RelativeDate.LAST_30_DAYS, f"run {i+1}: date_range={plan.date_range}"
        assert plan.filters == [], (
            f"run {i+1}: filters={plan.filters} -- a plain count with no status named must never "
            f"gain an invented status filter; empty filters means every status is counted."
        )


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_attendance_percentage_last_30_days_retains_correct_semantics():
    """PERCENTAGE's existing explicit numerator contract must be completely
    unaffected by LIST or LAST_30_DAYS being added.

    Contract update (2026-09-03 architectural placement review): calls bare
    resolve_structured() directly, with no retry -- and that method now
    genuinely guarantees ranking-field coherence on every plan it returns
    (see its own docstring), so this test now also asserts `extreme is
    None`, observing that canonical contract directly rather than only via
    the full lifecycle. This is a real, deliberate strengthening of what
    this test checks, not a loosening: a single raw call can still hit an
    unrelated, pre-existing raw-model reliability gap (garbage/incomplete
    output, e.g. entity=None) that retry-with-feedback recovers from in
    production -- that gap is out of scope for this test and is not
    weakened or hidden by anything here; it is a known, separate,
    unresolved issue (see the investigation this test's history documents),
    not something this assertion papers over."""
    agent = IntentResolutionAgent()
    context = {"school_id": 56}

    for paraphrase in [
        "What is my attendance percentage for the last 30 days?",
        "What is the attendance percentage for the last 30 days?",
    ]:
        for i in range(PLAIN_TOTAL_RUNS):
            plan = await agent.resolve_structured(paraphrase, context)
            assert plan.can_answer, f"{paraphrase!r} run {i+1}: model declined to answer"
            assert plan.entity == Entity.ATTENDANCE
            assert plan.operation == Operation.PERCENTAGE, f"{paraphrase!r} run {i+1}: operation={plan.operation}"
            assert plan.date_range == RelativeDate.LAST_30_DAYS, f"{paraphrase!r} run {i+1}: date_range={plan.date_range}"
            assert plan.percentage_of is not None
            assert plan.percentage_of.numerator.value == "present"
            assert plan.extreme is None, (
                f"{paraphrase!r} run {i+1}: extreme={plan.extreme} -- resolve_structured() now "
                f"guarantees ranking-field coherence on every returned plan; this question has "
                f"group_by=NONE, so extreme must never survive."
            )


# ── Ranking-field leakage: deterministic pre-validation fix ─────────────────
#
# Root-cause investigation (2026-09-02): non-ranking ATTENDANCE questions
# (and, confirmed via git stash, the PRE-EXISTING "What is my attendance
# percentage?" -- present before any of today's LAST_30_DAYS/ATTENDANCE-LIST
# work) deterministically produced a spurious `extreme` (or, less often,
# `sort=aggregate_value`+`limit`) with group_by=NONE -- a combination
# QueryPlanValidator already, correctly, rejects. Two targeted prompt-only
# attempts showed unstable, non-convergent improvement, so the fix is
# architectural instead: `clear_incoherent_ranking_fields` (query_plan.py)
# deterministically clears exactly these two fields, using the EXACT SAME
# structural precondition (`is_ranking_capable`, also query_plan.py) the
# validator itself already enforces. Applied inside intent_agent.
# resolve_structured() itself (2026-09-03 architectural placement review)
# -- the single canonical parsing boundary every caller goes through
# (direct callers, resolve_structured_with_feedback's retry, and the full
# QueryLifecycleAgent pipeline all observe the identical, already-repaired
# contract), not a higher-level lifecycle-only patch.
#
# These tests exercise the FULL pipeline (QueryLifecycleAgent.run()) to
# prove the fix reaches all the way through in practice, not because the
# fix itself is lifecycle-scoped -- see test_ranking_field_sanitization.py
# for direct proof that resolve_structured() itself already guarantees
# this contract, independent of query_lifecycle.py.

RANKING_LEAKAGE_CONTEXT = {
    "tenant": "patasala", "role": "ADMIN", "email": "admin@school1.com",
    "user_id": "e2ce1839-8791-4601-ba37-5568f45b0e08", "school_id": 56,
}
RANKING_LEAKAGE_RUNS = 3


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_non_ranking_attendance_questions_never_fail_closed_from_leaked_ranking_fields():
    """The exact previously-failing non-ranking questions -- must reach a
    real answer (never a clarification) through the full pipeline, every
    run. This is the direct regression test for the reported bug."""
    agent = QueryLifecycleAgent()
    queries = [
        "Show me attendance for the last 30 days.",
        "Show attendance for the last 30 days.",
        "Show me attendance this month.",
        "What is my attendance percentage?",
        "What is the attendance percentage for the last 30 days?",
        "How many attendance records are there in the last 30 days?",
        "What is the attendance?",
    ]
    for query in queries:
        for i in range(RANKING_LEAKAGE_RUNS):
            result = await agent.run(query, RANKING_LEAKAGE_CONTEXT, None)
            assert result.get("clarification") is None, (
                f"{query!r} run {i+1}: failed closed to a clarification -- "
                f"{result.get('clarification')}"
            )
            assert result.get("answer") is not None, f"{query!r} run {i+1}: no answer produced"


@pytest.mark.skipif(not _ollama_available(), reason="No live Ollama server reachable -- skipping live reliability measurement.")
@pytest.mark.asyncio
async def test_no_number_ranking_questions_still_resolve_correctly_through_full_pipeline():
    """Legitimate ranking (no stated number) must remain completely
    unaffected by the fix -- proven through the full pipeline, matching the
    non-ranking test above exactly so the two are directly comparable."""
    agent = QueryLifecycleAgent()
    queries = [
        "Which students have the highest attendance?",
        "Which students have the lowest attendance?",
        "Show the bottom 3 students by attendance.",
    ]
    for query in queries:
        for i in range(RANKING_LEAKAGE_RUNS):
            result = await agent.run(query, RANKING_LEAKAGE_CONTEXT, None)
            assert result.get("clarification") is None, (
                f"{query!r} run {i+1}: failed closed to a clarification -- "
                f"{result.get('clarification')}"
            )
            assert result.get("answer") is not None, f"{query!r} run {i+1}: no answer produced"
            intent = result.get("intent") or {}
            assert "GROUP BY" in (intent.get("sql") or ""), (
                f"{query!r} run {i+1}: expected a grouped ranking query, got sql={intent.get('sql')!r}"
            )
