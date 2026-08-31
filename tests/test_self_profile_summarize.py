"""Dedicated coverage for QueryLifecycleAgent._summarize()'s self-profile
pre-filter and context-threading (see the Principal Engineer Review finding
in docs/architecture/Patasala-OPA-Policy-Status.md "Open item": this logic
was previously verified only by manual, live testing).

These are unit tests against `_summarize` directly, not the HTTP layer --
`intent_agent.summarize` is mocked so we can assert exactly what `data` and
`context` it was called with, which is the actual behavior under test.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.query_lifecycle import QueryLifecycleAgent


@pytest.fixture()
def orchestrator():
    # No env patching needed/wanted here: these tests mock
    # intent_agent.summarize directly (see below), so real LLM provider setup
    # in IntentResolutionAgent.__init__ never gets exercised either way, and
    # patching os.getenv would also affect unrelated env reads elsewhere in
    # the same constructor chain (e.g. SchemaProvider's DB_STATEMENT_TIMEOUT_
    # SECONDS) since `patch("...os.getenv", ...)` replaces the process-wide
    # os.getenv function, not just this module's usage of it.
    return QueryLifecycleAgent()


def _rows_school_56():
    return [
        {"id": "e2ce1839-8791-4601-ba37-5568f45b0e08", "email": "admin@school1.com", "first_name": "Admin"},
        {"id": "aaaaaaaa-1111-2222-3333-444444444444", "email": "other@school1.com", "first_name": "Other"},
        {"id": "bbbbbbbb-1111-2222-3333-444444444444", "email": "third@school1.com", "first_name": "Third"},
    ]


@pytest.mark.asyncio
async def test_self_profile_query_selects_matching_row_by_id(orchestrator):
    context = {"user_id": "e2ce1839-8791-4601-ba37-5568f45b0e08", "email": "admin@school1.com"}
    state = {
        "query": "Show me my profile details.",
        "sql": "SELECT id, email, first_name FROM users",
        "data": _rows_school_56(),
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == [{"id": "e2ce1839-8791-4601-ba37-5568f45b0e08", "email": "admin@school1.com", "first_name": "Admin"}]


@pytest.mark.asyncio
async def test_self_profile_query_selects_matching_row_by_email_when_id_absent(orchestrator):
    rows = [
        {"email": "admin@school1.com", "first_name": "Admin"},
        {"email": "other@school1.com", "first_name": "Other"},
    ]
    context = {"user_id": "some-id-not-present-in-rows", "email": "admin@school1.com"}
    state = {
        "query": "about me",
        "sql": "SELECT email, first_name FROM users",
        "data": rows,
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == [{"email": "admin@school1.com", "first_name": "Admin"}]


@pytest.mark.asyncio
async def test_self_profile_query_selects_matching_row_by_user_id_field(orchestrator):
    # teacher_profiles-shaped data: identity column is `user_id`, not `id`.
    rows = [
        {"user_id": "caller-id", "designation": "Senior Teacher"},
        {"user_id": "someone-else", "designation": "Junior Teacher"},
    ]
    context = {"user_id": "caller-id"}
    state = {
        "query": "what is my designation",  # note: does NOT match the narrow regex
        "sql": "SELECT user_id, designation FROM teacher_profiles",
        "data": rows,
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    # "what is my designation" doesn't match _SELF_REFERENCE_RE (only
    # "profile/details/information/info/account/record"), so no narrowing
    # happens here -- this test documents that boundary, not a match case.
    called_data = mock_summarize.call_args.args[2]
    assert called_data == rows


@pytest.mark.asyncio
async def test_my_details_phrasing_triggers_narrowing_by_user_id_field():
    orchestrator = QueryLifecycleAgent()
    rows = [
        {"user_id": "caller-id", "designation": "Senior Teacher"},
        {"user_id": "someone-else", "designation": "Junior Teacher"},
    ]
    context = {"user_id": "caller-id"}
    state = {
        "query": "show me my details",
        "sql": "SELECT user_id, designation FROM teacher_profiles",
        "data": rows,
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == [{"user_id": "caller-id", "designation": "Senior Teacher"}]


@pytest.mark.asyncio
async def test_who_am_i_phrasing_triggers_narrowing(orchestrator):
    context = {"user_id": "e2ce1839-8791-4601-ba37-5568f45b0e08"}
    state = {
        "query": "Who am I?",
        "sql": "SELECT id, email FROM users",
        "data": _rows_school_56(),
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert len(called_data) == 1
    assert called_data[0]["id"] == "e2ce1839-8791-4601-ba37-5568f45b0e08"


@pytest.mark.asyncio
async def test_unrelated_broader_query_does_not_alter_data(orchestrator):
    """'students in my school' contains the word 'my' but is NOT a self-
    profile question -- must not be narrowed, per _SELF_REFERENCE_RE's
    deliberately narrow scope."""
    context = {"user_id": "e2ce1839-8791-4601-ba37-5568f45b0e08"}
    rows = _rows_school_56()
    state = {
        "query": "Show me all students in my school",
        "sql": "SELECT id, email, first_name FROM users",
        "data": rows,
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == rows


@pytest.mark.asyncio
async def test_aggregate_query_with_my_school_phrasing_untouched(orchestrator):
    context = {"user_id": "e2ce1839-8791-4601-ba37-5568f45b0e08"}
    rows = [{"count": 98}]
    state = {
        "query": "how many students are in my school",
        "sql": "SELECT COUNT(*) as count FROM students",
        "data": rows,
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == rows


@pytest.mark.asyncio
async def test_self_profile_query_with_no_matching_identity_leaves_data_unchanged(orchestrator):
    """If a self-referential question is asked but NONE of the returned rows
    match the caller's own identity, the pre-filter must NOT drop all data
    (which would look like "nothing exists") -- it must leave `data` exactly
    as it was and let intent_agent.summarize's own Caller Identity backstop
    (or an honest 'no match' answer) handle it instead."""
    context = {"user_id": "someone-not-in-this-result-set", "email": "nomatch@example.com"}
    rows = _rows_school_56()
    state = {
        "query": "show me my profile details",
        "sql": "SELECT id, email, first_name FROM users",
        "data": rows,
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == rows  # unchanged, NOT emptied


@pytest.mark.asyncio
async def test_single_row_data_never_narrowed_even_with_self_phrasing(orchestrator):
    """The len(data) > 1 guard means a single-row result (e.g. a policy
    filter that's already self-scoped) is never touched by this logic at
    all, matching/relying on data already being correct as-is."""
    context = {"user_id": "someone-else-entirely"}
    rows = [{"id": "e2ce1839-8791-4601-ba37-5568f45b0e08", "email": "admin@school1.com"}]
    state = {
        "query": "show me my profile details",
        "sql": "SELECT id, email FROM users WHERE id = 'e2ce1839-8791-4601-ba37-5568f45b0e08'",
        "data": rows,
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == rows


@pytest.mark.asyncio
async def test_self_profile_query_without_context_does_not_crash_or_alter_data(orchestrator):
    state = {
        "query": "show me my profile details",
        "sql": "SELECT id, email FROM users",
        "data": _rows_school_56(),
        "context": None,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_data = mock_summarize.call_args.args[2]
    assert called_data == _rows_school_56()


# ── Context threading into summarize() ──────────────────────────────────────

@pytest.mark.asyncio
async def test_context_is_threaded_into_summarize_call(orchestrator):
    context = {"user_id": "e2ce1839-8791-4601-ba37-5568f45b0e08", "email": "admin@school1.com"}
    state = {
        "query": "how many students are there",
        "sql": "SELECT COUNT(*) FROM students",
        "data": [{"count": 15986}],
        "context": context,
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    mock_summarize.assert_called_once()
    called_context = mock_summarize.call_args.args[3]
    assert called_context == context


@pytest.mark.asyncio
async def test_missing_context_threads_empty_dict_into_summarize():
    orchestrator = QueryLifecycleAgent()
    state = {
        "query": "how many students are there",
        "sql": "SELECT COUNT(*) FROM students",
        "data": [{"count": 15986}],
        # no "context" key at all
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="ok")) as mock_summarize:
        await orchestrator._summarize(state)

    called_context = mock_summarize.call_args.args[3]
    assert called_context == {}


@pytest.mark.asyncio
async def test_summarize_return_value_becomes_answer(orchestrator):
    state = {
        "query": "how many students are there",
        "sql": "SELECT COUNT(*) FROM students",
        "data": [{"count": 15986}],
        "context": {},
    }
    with patch.object(orchestrator.intent_agent, "summarize", new=AsyncMock(return_value="There are **15986** students.")):
        result = await orchestrator._summarize(state)

    assert result == {"answer": "There are **15986** students."}
