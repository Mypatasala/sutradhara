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


# ── HOMEWORK SUBJECT reachability (2026-09-07) ───────────────────────────────
# The backend (query_registry.py's HOMEWORK.lookup_filter_fields[SUBJECT])
# landed and was fully tested before the prompt ever mentioned it -- the
# same "fully supported but practically unreachable" gap as explicit dates
# above, just for a filter field instead of a schema field. These tests
# isolate the HOMEWORK entity bullet specifically (not just search the
# whole prompt) so a match against COURSE_SCHEDULE's pre-existing, unrelated
# subject mention can never make these pass by accident.

def _homework_bullet() -> str:
    """Extracts just the "- homework -- ..." bullet's text, up to the next
    "\\n- " bullet marker, so assertions below can prove the wording is
    actually ATTACHED to homework, not merely present somewhere else in the
    prompt (e.g. COURSE_SCHEDULE's own, pre-existing subject mention)."""
    start = _PROMPT.index("- homework --")
    end = _PROMPT.index("\n- ", start + 1)
    return _PROMPT[start:end]


def test_prompt_homework_bullet_documents_subject_filter():
    bullet = _homework_bullet()
    assert "subject" in bullet


def test_prompt_homework_subject_described_as_dynamic_lookup_not_fixed_list():
    bullet = _homework_bullet()
    assert "dynamic lookup" in bullet
    assert "not a fixed list" in bullet


def test_prompt_homework_bullet_still_documents_status_filter_unchanged():
    """Regression: the pre-existing status filter documentation and its
    allowed-value list must survive this addition unchanged."""
    bullet = _homework_bullet()
    assert "status" in bullet
    assert "pending/submitted/graded/late" in bullet
