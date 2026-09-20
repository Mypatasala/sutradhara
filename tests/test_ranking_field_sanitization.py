"""
Deterministic unit tests for the ranking-field leakage fix (root-cause
investigation, 2026-09-02, architectural placement reviewed 2026-09-03):
non-ranking ATTENDANCE questions were live-observed producing a spurious
`extreme` (or `sort=aggregate_value` + `limit`) despite `group_by=NONE` --
a combination QueryPlanValidator already, correctly, rejects. Confirmed
PRE-EXISTING via git stash (not introduced by any of today's LAST_30_DAYS/
ATTENDANCE-LIST work). Two targeted prompt-only attempts showed unstable,
non-convergent improvement, so the fix is architectural instead:
`is_ranking_capable` and `clear_incoherent_ranking_fields` both live in
query_plan.py (a pure QueryPlan-shape utility, zero DB/registry
dependency) -- both the validator's rejection checks and the repair use
the exact same condition, so a plan is never silently repaired under a
looser rule than the one that would otherwise reject it. The repair is
applied at the single canonical boundary (intent_agent.resolve_structured())
every caller goes through -- direct callers, resolve_structured_with_
feedback's retry, and the full QueryLifecycleAgent pipeline all observe
the identical, already-repaired contract; see intent_agent.py's
resolve_structured() docstring.

No live LLM/DB involved -- these are pure-function tests.
"""

from src.agents.query_plan import (
    ComparisonFilter,
    Entity,
    ExtremeSelector,
    FilterField,
    GroupingDimension,
    Operation,
    PercentageSpec,
    QueryPlan,
    RelativeDate,
    SortField,
    SortSpec,
    clear_incoherent_ranking_fields,
    is_ranking_capable,
)


# ── is_ranking_capable ───────────────────────────────────────────────────────

def test_ranking_capable_requires_both_group_by_and_aggregate_operation():
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT)
    assert is_ranking_capable(plan) is True


def test_not_ranking_capable_when_group_by_is_none():
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.NONE)
    assert is_ranking_capable(plan) is False


def test_not_ranking_capable_when_operation_is_list_even_with_group_by():
    plan = QueryPlan(entity=Entity.ATTENDANCE, operation=Operation.LIST, group_by=GroupingDimension.BY_STUDENT)
    assert is_ranking_capable(plan) is False


def test_ranking_capable_for_count_grouped_by_class():
    """Not attendance-specific -- STUDENTS/COUNT/BY_CLASS is equally
    "ranking capable" by this same structural definition, even though
    nothing in today's registry actually sets extreme for it."""
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_CLASS)
    assert is_ranking_capable(plan) is True


# ── clear_incoherent_ranking_fields: the exact observed defects ────────────

def test_clears_spurious_extreme_on_ungrouped_percentage():
    """The exact literal reproduction: 'What is my attendance percentage?'
    (confirmed live, pre-existing on the untouched baseline) -- extreme set
    with group_by=NONE."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.NONE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.HIGHEST,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared.extreme is None
    assert cleared.operation == Operation.PERCENTAGE  # nothing else touched
    assert cleared.group_by == GroupingDimension.NONE
    assert cleared.percentage_of == plan.percentage_of


def test_clears_spurious_extreme_on_ungrouped_count():
    """'How many attendance records are there in the last 30 days?' --
    extreme set on a plain COUNT with group_by=NONE."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.NONE,
        date_range=RelativeDate.LAST_30_DAYS, extreme=ExtremeSelector.LOWEST,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared.extreme is None
    assert cleared.date_range == RelativeDate.LAST_30_DAYS  # untouched


def test_clears_spurious_extreme_on_list_operation():
    """'Show attendance for the last 30 days.' -- extreme set on operation
    =list, which AGGREGATE_OPERATIONS never includes at all."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.LIST, group_by=GroupingDimension.NONE,
        date_range=RelativeDate.LAST_30_DAYS, extreme=ExtremeSelector.LOWEST,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared.extreme is None
    assert cleared.operation == Operation.LIST


def test_clears_dangling_limit_and_sort_when_extreme_is_cleared():
    """A non-ranking-capable plan with extreme + limit set and sort=None
    (e.g. 'highest attendance' collapsed onto a plain, ungrouped operation
    that also picked up a spurious limit) previously left `limit` behind
    with no `sort` -- passing validation and reaching the SQL builder as
    an unordered LIMIT. extreme, limit, and sort must all be cleared
    together."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.NONE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.HIGHEST, limit=3,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared.extreme is None
    assert cleared.limit is None
    assert cleared.sort is None


def test_clears_spurious_aggregate_value_sort_and_its_limit_on_list_operation():
    """'Show the 5 students with the lowest attendance.' -- observed
    producing operation=list combined with sort=aggregate_value + limit=5,
    an invalid combination (AGGREGATE_VALUE sort requires an aggregate
    operation). Both sort and its accompanying limit must be cleared
    together -- a bare limit left behind with no sort would be equally
    meaningless."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.LIST, group_by=GroupingDimension.BY_STUDENT,
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"), limit=5,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared.sort is None
    assert cleared.limit is None
    assert cleared.group_by == GroupingDimension.BY_STUDENT  # only sort/limit cleared, not group_by


# ── Must NEVER touch a genuinely valid ranking plan ──────────────────────────

def test_never_touches_valid_extreme_ranking_plan():
    """'Which students have the lowest attendance?' -- group_by=by_student
    + operation=percentage IS ranking-capable; extreme must survive
    completely unchanged."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared is plan  # identity: no-op, not just equal -- proves nothing was rebuilt
    assert cleared.extreme == ExtremeSelector.LOWEST


def test_never_touches_valid_explicit_n_ranking_plan():
    """'Show the 5 students with the lowest attendance.' -- the CORRECT
    shape (percentage + by_student + sort=aggregate_value + limit) must
    survive completely unchanged."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="asc"), limit=5,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared is plan
    assert cleared.sort == plan.sort
    assert cleared.limit == 5


def test_never_touches_physical_column_sort_even_with_group_by_none():
    """'Show my latest report card.' -- operation=list, sort=issue_date
    (a PHYSICAL column, not aggregate_value), group_by=NONE. This is a
    completely valid, unrelated shape (report_cards has no groupings at
    all) that must never be touched -- the aggregate-value branch is the
    only one that can ever fire."""
    plan = QueryPlan(
        entity=Entity.REPORT_CARDS, operation=Operation.LIST, group_by=GroupingDimension.NONE,
        sort=SortSpec(field=SortField.ISSUE_DATE, direction="desc"), limit=1,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared is plan
    assert cleared.sort.field == SortField.ISSUE_DATE
    assert cleared.limit == 1


def test_never_touches_a_plan_with_neither_extreme_nor_sort():
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.NONE)
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared is plan


def test_clears_both_extreme_and_aggregate_sort_independently_when_both_present():
    """Defensive case: even though the validator itself already rejects
    extreme+sort combined together, the sanitizer must not assume they're
    mutually exclusive on the way in -- both fields get the same
    independent treatment."""
    plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.LIST, group_by=GroupingDimension.NONE,
        extreme=ExtremeSelector.HIGHEST,
        sort=SortSpec(field=SortField.AGGREGATE_VALUE, direction="desc"), limit=3,
    )
    cleared = clear_incoherent_ranking_fields(plan)
    assert cleared.extreme is None
    assert cleared.sort is None
    assert cleared.limit is None


# ── Wiring proof: resolve_structured() itself applies the repair ────────────
#
# Everything above tests clear_incoherent_ranking_fields() in isolation --
# these tests instead prove the ACTUAL WIRING claimed in intent_agent.py's
# resolve_structured() docstring: that the repair is genuinely applied at
# that single canonical boundary, for both a direct call and the retry-with-
# -feedback path (which delegates to the same method). The underlying LLM
# call is mocked (no live Ollama needed -- deterministic, fast, always run
# in CI) so this exercises the REAL resolve_structured()/resolve_structured_
# with_feedback() code path, not just the pure function in isolation.

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.intent_agent import IntentResolutionAgent


def _fake_model_returning(plan: QueryPlan):
    """A minimal stand-in for a langchain chat model configured via
    `self.models`, matching the exact interface resolve_structured() calls:
    `model.with_structured_output(QueryPlan)` -> object with async
    `.ainvoke(prompt)` -> QueryPlan."""
    structured_model = MagicMock()
    structured_model.ainvoke = AsyncMock(return_value=plan)
    fake_model = MagicMock()
    fake_model.with_structured_output = MagicMock(return_value=structured_model)
    return fake_model


@pytest.mark.asyncio
async def test_resolve_structured_itself_clears_an_incoherent_raw_plan():
    """Direct resolve_structured() caller: the exact live-observed defect
    shape (extreme set on group_by=NONE) must come back repaired, proving
    the repair is applied INSIDE resolve_structured(), not left to whoever
    calls it."""
    incoherent_plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.NONE,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.HIGHEST,
    )
    agent = IntentResolutionAgent()
    agent.models = [_fake_model_returning(incoherent_plan)]

    result = await agent.resolve_structured("What is my attendance percentage?", {"school_id": 56})

    assert result.extreme is None
    assert result.operation == Operation.PERCENTAGE  # nothing else touched
    assert result.percentage_of == incoherent_plan.percentage_of


@pytest.mark.asyncio
async def test_resolve_structured_leaves_a_valid_ranking_plan_untouched():
    """Direct resolve_structured() caller, the negative case: a genuinely
    valid ranking plan must come back completely unchanged."""
    valid_plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.PERCENTAGE, group_by=GroupingDimension.BY_STUDENT,
        percentage_of=PercentageSpec(numerator=ComparisonFilter(field=FilterField.STATUS, value="present")),
        extreme=ExtremeSelector.LOWEST,
    )
    agent = IntentResolutionAgent()
    agent.models = [_fake_model_returning(valid_plan)]

    result = await agent.resolve_structured("Which students have the lowest attendance?", {"school_id": 56})

    assert result.extreme == ExtremeSelector.LOWEST


@pytest.mark.asyncio
async def test_resolve_structured_with_feedback_retry_also_clears_an_incoherent_plan():
    """The retry-with-feedback path: resolve_structured_with_feedback()
    delegates to resolve_structured() internally, so the SAME repair must
    apply to whatever plan the retry attempt produces -- proven here with
    its own mocked model, not just inferred from reading the delegation."""
    incoherent_retry_plan = QueryPlan(
        entity=Entity.ATTENDANCE, operation=Operation.COUNT, group_by=GroupingDimension.NONE,
        date_range=RelativeDate.LAST_30_DAYS, extreme=ExtremeSelector.LOWEST,
    )
    agent = IntentResolutionAgent()
    agent.models = [_fake_model_returning(incoherent_retry_plan)]

    result = await agent.resolve_structured_with_feedback(
        "How many attendance records are there in the last 30 days?",
        {"school_id": 56},
        "extreme requires a grouped aggregate result (group_by set and operation one of "
        "['average', 'count', 'percentage', 'sum']).",
    )

    assert result.extreme is None
    assert result.operation == Operation.COUNT
    assert result.date_range == RelativeDate.LAST_30_DAYS
