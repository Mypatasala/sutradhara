"""
Deterministic contract tests for IntentResolutionAgent's structured-output
prompt text (_STRUCTURED_SYSTEM_PROMPT) -- these assert on the PROMPT
STRING itself, never invoking any LLM, so they prove the model-facing
documentation for a field exists without empirically testing whether any
particular model actually follows it (that is exactly what
tests/test_live_structured_reliability.py's live-Ollama tests measure
instead, and is explicitly out of scope here).

Explicit date/date-range support (2026-09-05): the backend
(QueryPlan.explicit_start_date/explicit_end_date, QueryPlanValidator,
query_normalizer, StructuredSQLBuilder) landed and was fully tested before
the prompt ever mentioned these fields at all -- a fully backend-supported
but practically unreachable capability, since schema-constrained decoding
only tells the model a field CAN be populated, never WHEN or HOW. These
tests close that gap by proving the prompt text itself documents the
contract, mirroring the existing DATE_RANGE section's worked-example style.
"""

from src.agents.intent_agent import IntentResolutionAgent

_PROMPT = IntentResolutionAgent._STRUCTURED_SYSTEM_PROMPT


def test_prompt_mentions_both_explicit_date_field_names():
    assert "explicit_start_date" in _PROMPT
    assert "explicit_end_date" in _PROMPT


def test_prompt_documents_strict_iso_format():
    assert "YYYY-MM-DD" in _PROMPT


def test_prompt_documents_single_date_as_start_equals_end():
    assert "start == end" in _PROMPT or "SAME date" in _PROMPT


def test_prompt_documents_mutual_exclusivity_with_date_range():
    """The prompt already contains an unrelated "mutually exclusive"
    sentence (extreme vs sort/limit) -- this must find a DIFFERENT
    occurrence that actually names both date mechanisms, not just match the
    phrase's first/only appearance."""
    start = 0
    found = False
    while True:
        idx = _PROMPT.find("mutually exclusive", start)
        if idx == -1:
            break
        surrounding = _PROMPT[max(0, idx - 200): idx + 50]
        if ("explicit_start_date" in surrounding or "explicit_end_date" in surrounding) and "date_range" in surrounding:
            found = True
            break
        start = idx + 1
    assert found, "no 'mutually exclusive' sentence found naming both explicit dates and date_range"


def test_prompt_includes_a_single_explicit_date_worked_example():
    assert 'explicit_start_date="2026-08-15", explicit_end_date="2026-08-15"' in _PROMPT


def test_prompt_includes_an_explicit_date_range_worked_example():
    assert 'explicit_start_date="2026-08-01", explicit_end_date="2026-08-15"' in _PROMPT


def test_prompt_does_not_introduce_free_form_relative_date_phrases():
    """Guard against scope creep: the prompt addition must stay strict
    YYYY-MM-DD, never natural-language relative phrasing like "next
    Tuesday" or "the 15th" -- that would require actual date-arithmetic
    parsing this pipeline deliberately keeps out of the model's hands (see
    RelativeDate's own closed-vocabulary design in query_plan.py)."""
    assert "next tuesday" not in _PROMPT.lower()
    assert "the 15th" not in _PROMPT.lower()
