"""
Integration coverage for the grounding wiring in QueryLifecycleAgent._summarize
(see tests/test_summarization_grounding.py for the underlying pure-function
unit tests) and for _try_structured_resolution's result_kind/aggregate_alias
derivation (Part 3's "does the pipeline already carry enough metadata to
distinguish scalar vs grouped vs list results" question -- yes, from
plan.operation/plan.group_by alone, proven here without any live LLM/DB).

Mocks intent_agent.summarize directly, exactly like test_self_profile_
summarize.py, so these assert real _summarize behavior, not the HTTP layer.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.query_lifecycle import QueryLifecycleAgent
from src.agents.query_plan import (
    Entity,
    GroupingDimension,
    Operation,
    QueryPlan,
)


@pytest.fixture()
def orchestrator():
    return QueryLifecycleAgent()


# ── _summarize: the LLM's wrong number is corrected before reaching the UI ──
#
# The original reported incident ("How many students are in my school?" ->
# "There are 10 students", from 9 grouped {class_name, count: 5} rows
# summing to 45) is now prevented one layer upstream: A1's prompt fix means
# this exact question resolves to result_kind="scalar_aggregate" (group_by
# unset), not "grouped_aggregate", so it is the scalar test below --
# test_summarize_corrects_a_wrong_scalar_count -- that is now this
# incident's literal regression test, not a grouped-rows one. See
# test_summarization_grounding.py's module docstring for why a
# "grouped_aggregate -> sum" backstop was deliberately removed rather than
# kept as defense-in-depth: it was itself a free-text heuristic and it
# live-broke a correct breakdown answer.

@pytest.mark.asyncio
async def test_summarize_corrects_a_wrong_scalar_count(orchestrator):
    """The literal regression test for the original incident, now correctly
    exercised as the scalar_aggregate case (see comment above): the SQL
    already returns exactly one authoritative row; if the LLM's answer
    states a different number, it must be corrected."""
    data = [{"count": 45}]
    state = {
        "query": "How many students are in my school?",
        "sql": "SELECT COUNT(*) AS count FROM students",
        "data": data,
        "context": {"user_id": "u1", "email": "admin@school1.com"},
        "result_kind": "scalar_aggregate",
        "aggregate_alias": "count",
    }
    with patch.object(
        orchestrator.intent_agent, "summarize",
        new=AsyncMock(return_value="There are **10** students in my school."),
    ):
        result = await orchestrator._summarize(state)

    assert "**45**" in result["answer"]
    assert "**10**" not in result["answer"]


@pytest.mark.asyncio
async def test_summarize_never_touches_a_grouped_aggregate_answer(orchestrator):
    """A grouped_aggregate result means a breakdown was explicitly
    requested -- grounding must be a complete no-op for it, even when the
    LLM's answer contains a number that doesn't match any row (there is no
    single authoritative number the plan asked for, so nothing to check
    against). This is the deliberate, narrower replacement for the removed
    prose-counting heuristic (see test_summarization_grounding.py)."""
    data = [{"class_name": f"Grade {i}", "count": 5} for i in range(9)]
    state = {
        "query": "How many students are in each class?",
        "sql": "SELECT ... GROUP BY ...",
        "data": data,
        "context": {},
        "result_kind": "grouped_aggregate",
        "aggregate_alias": "count",
    }
    hallucinated = "There are **10** students in my school."
    with patch.object(
        orchestrator.intent_agent, "summarize",
        new=AsyncMock(return_value=hallucinated),
    ):
        result = await orchestrator._summarize(state)

    # Left completely untouched -- correctness here is guaranteed upstream,
    # by the QueryPlan classification (A1) never mis-grouping a plain total
    # in the first place, not by inspecting this text.
    assert result["answer"] == hallucinated


@pytest.mark.asyncio
async def test_summarize_leaves_already_correct_answer_untouched(orchestrator):
    data = [{"count": 45}]
    state = {
        "query": "How many students are in my school?",
        "sql": "SELECT COUNT(*) AS count FROM students",
        "data": data,
        "context": {},
        "result_kind": "scalar_aggregate",
        "aggregate_alias": "count",
    }
    with patch.object(
        orchestrator.intent_agent, "summarize",
        new=AsyncMock(return_value="There are **45** students in your school."),
    ):
        result = await orchestrator._summarize(state)

    assert result["answer"] == "There are **45** students in your school."


@pytest.mark.asyncio
async def test_summarize_never_collapses_a_correctly_rendered_breakdown_into_a_bare_total(orchestrator):
    """Regression case caught live during Part 3 validation: "How many
    students are in each class?" legitimately asks for a breakdown (A1's
    contrastive fix keeps group_by=by_class for exactly this phrasing) --
    an LLM answer that correctly renders one row per group must reach the
    caller completely unchanged, never collapsed into a bare total."""
    data = [{"class_name": f"Grade {i}", "count": 5} for i in range(9)]
    correct_breakdown_answer = (
        "| Class | Students |\n| --- | --- |\n"
        + "\n".join(f"| Grade {i} | **5** |" for i in range(9))
    )
    state = {
        "query": "How many students are in each class?",
        "sql": "SELECT ... GROUP BY ...",
        "data": data,
        "context": {},
        "result_kind": "grouped_aggregate",
        "aggregate_alias": "count",
    }
    with patch.object(
        orchestrator.intent_agent, "summarize",
        new=AsyncMock(return_value=correct_breakdown_answer),
    ):
        result = await orchestrator._summarize(state)

    assert result["answer"] == correct_breakdown_answer
    assert "**45**" not in result["answer"]


@pytest.mark.asyncio
async def test_summarize_is_a_no_op_for_list_results(orchestrator):
    """A list result has no single groundable number -- grounding must never
    touch a genuine roster/list answer."""
    data = [{"first_name": "A", "last_name": "B"}, {"first_name": "C", "last_name": "D"}]
    state = {
        "query": "List all students in Grade 5.",
        "sql": "SELECT ...",
        "data": data,
        "context": {},
        "result_kind": "list",
        "aggregate_alias": None,
    }
    with patch.object(
        orchestrator.intent_agent, "summarize",
        new=AsyncMock(return_value="| first_name | last_name |\n| --- | --- |\n| A | B |\n| C | D |"),
    ):
        result = await orchestrator._summarize(state)

    assert result["answer"] == "| first_name | last_name |\n| --- | --- |\n| A | B |\n| C | D |"


@pytest.mark.asyncio
async def test_summarize_is_a_no_op_for_the_legacy_free_text_path(orchestrator):
    """result_kind/aggregate_alias are simply absent from state on the
    legacy free-text path -- grounding must be inert, exactly as before this
    change (state.get(...) returns None either way)."""
    data = [{"count": 45}]
    state = {
        "query": "How many students are there",
        "sql": "SELECT COUNT(*) FROM students",
        "data": data,
        "context": {},
        # no result_kind / aggregate_alias key at all
    }
    with patch.object(
        orchestrator.intent_agent, "summarize",
        new=AsyncMock(return_value="There are **99** students."),
    ):
        result = await orchestrator._summarize(state)

    # Not touched -- legacy path carries no plan metadata to ground against.
    assert result["answer"] == "There are **99** students."


# ── _try_structured_resolution: result_kind/aggregate_alias derivation ──────
# Proves the plan metadata already available (operation, group_by) is
# sufficient to classify the four result shapes -- no column-name
# inspection anywhere in this derivation.

@pytest.mark.asyncio
async def test_plain_count_plan_is_classified_scalar_aggregate(orchestrator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.NONE)
    with patch.object(orchestrator.intent_agent, "resolve_structured", new=AsyncMock(return_value=plan)):
        result = await orchestrator._try_structured_resolution({"query": "How many students are there?", "context": {"school_id": 56}})

    assert result["result_kind"] == "scalar_aggregate"
    assert result["aggregate_alias"] == "count"
    assert "GROUP BY" not in result["sql"]


@pytest.mark.asyncio
async def test_grouped_count_plan_is_classified_grouped_aggregate(orchestrator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.COUNT, group_by=GroupingDimension.BY_CLASS)
    with patch.object(orchestrator.intent_agent, "resolve_structured", new=AsyncMock(return_value=plan)):
        result = await orchestrator._try_structured_resolution({"query": "How many students are in each class?", "context": {"school_id": 56}})

    assert result["result_kind"] == "grouped_aggregate"
    assert result["aggregate_alias"] == "count"
    assert "GROUP BY" in result["sql"]


@pytest.mark.asyncio
async def test_school_classes_count_plan_is_classified_scalar_aggregate(orchestrator):
    """The new class-count entity (Part A2) is a plain COUNT with no
    grouping support -- must ground exactly like any other scalar aggregate,
    generic to the entity (no STUDENTS-specific logic anywhere in this
    path)."""
    plan = QueryPlan(entity=Entity.SCHOOL_CLASSES, operation=Operation.COUNT)
    with patch.object(orchestrator.intent_agent, "resolve_structured", new=AsyncMock(return_value=plan)):
        result = await orchestrator._try_structured_resolution({"query": "How many classes are there?", "context": {"school_id": 56}})

    assert result["result_kind"] == "scalar_aggregate"
    assert result["aggregate_alias"] == "count"
    assert result["sql"] == "SELECT COUNT(*) AS count FROM school_classes"


@pytest.mark.asyncio
async def test_summarize_grounds_school_classes_count(orchestrator):
    state = {
        "query": "How many classes are there?",
        "sql": "SELECT COUNT(*) AS count FROM school_classes",
        "data": [{"count": 10}],
        "context": {},
        "result_kind": "scalar_aggregate",
        "aggregate_alias": "count",
    }
    with patch.object(
        orchestrator.intent_agent, "summarize",
        new=AsyncMock(return_value="There are **45** classes in your school."),
    ):
        result = await orchestrator._summarize(state)

    assert "**10**" in result["answer"]
    assert "**45**" not in result["answer"]


@pytest.mark.asyncio
async def test_list_plan_is_classified_list_with_no_aggregate_alias(orchestrator):
    plan = QueryPlan(entity=Entity.STUDENTS, operation=Operation.LIST)
    with patch.object(orchestrator.intent_agent, "resolve_structured", new=AsyncMock(return_value=plan)):
        result = await orchestrator._try_structured_resolution({"query": "List all students.", "context": {"school_id": 56}})

    assert result["result_kind"] == "list"
    assert result["aggregate_alias"] is None
