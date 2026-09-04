"""
Deterministic unit tests for the summarization grounding boundary added to
fix the "How many students are in my school?" -> "There are 10 students"
incident: the summarizer LLM, given the exact correct 9 grouped rows
({class_name, count}) summing to 45, deterministically (5/5 live runs)
answered with a number (10) that matched none of them -- proving a prompt
instruction alone ("don't invent numbers") cannot guarantee correctness.

_compute_deterministic_aggregate derives the one groundable number PURELY
from QueryPlan-derived metadata (result_kind, aggregate_alias) -- never by
inspecting the LLM's answer text, guessing from column names, or summing
arbitrary numeric fields.

Principal-engineer review (2026-09-02) removed an earlier version's
"grouped_aggregate -> sum across groups" backstop, which decided WHETHER to
override the LLM's answer by counting bolded numbers in its prose -- itself
a free-text semantic heuristic, and one that live-broke a correct 9-row
breakdown answer by collapsing it to a bare total. The current contract is
narrower and fully structural: only "scalar_aggregate" (group_by=NONE) has
a single well-defined number at all -- the plan itself already declares
that shape, so there is nothing to infer from prose. A "grouped_aggregate"
plan means a breakdown was explicitly requested (group_by is only ever set
for that reason -- see the structured prompt's worked examples); the
correct answer IS the breakdown, and this function intentionally never
touches it.

No live LLM/DB involved -- these are pure-function tests.
"""

from decimal import Decimal

from src.agents.query_lifecycle import _compute_deterministic_aggregate, _ground_numeric_answer


# ── _compute_deterministic_aggregate ─────────────────────────────────────────

def test_scalar_aggregate_returns_the_single_row_value():
    data = [{"count": 45}]
    assert _compute_deterministic_aggregate(data, "scalar_aggregate", "count") == 45


def test_scalar_aggregate_percentage_returns_the_single_row_value():
    data = [{"percentage": Decimal("87.50")}]
    assert _compute_deterministic_aggregate(data, "scalar_aggregate", "percentage") == Decimal("87.50")


def test_grouped_aggregate_count_is_never_grounded():
    """A grouped COUNT means a breakdown was explicitly requested (group_by
    is only ever set for that reason) -- the breakdown itself IS the
    correct answer, so this must return None (nothing to ground/override),
    never a summed total nobody asked for. This is the exact case a prior
    version's prose-counting heuristic got wrong live (see module
    docstring): it collapsed a correct 9-row breakdown into a bare total."""
    data = [{"class_name": f"Grade {i}", "count": 5} for i in range(9)]
    assert _compute_deterministic_aggregate(data, "grouped_aggregate", "count") is None


def test_grouped_aggregate_percentage_is_never_grounded():
    """Per-group percentages/averages have no single meaningful total
    either -- must return None (nothing to ground against)."""
    data = [
        {"student_name": "Alice", "percentage": Decimal("40.00")},
        {"student_name": "Bob", "percentage": Decimal("85.00")},
    ]
    assert _compute_deterministic_aggregate(data, "grouped_aggregate", "percentage") is None


def test_list_result_has_no_groundable_number():
    data = [{"first_name": "A", "last_name": "B"}, {"first_name": "C", "last_name": "D"}]
    assert _compute_deterministic_aggregate(data, "list", None) is None


def test_no_aggregate_alias_means_nothing_to_ground_legacy_path():
    """The legacy free-text path never sets result_kind/aggregate_alias --
    this must be a complete no-op for it, exactly as before this change."""
    data = [{"count": 45}]
    assert _compute_deterministic_aggregate(data, None, None) is None


def test_empty_data_has_no_groundable_number():
    assert _compute_deterministic_aggregate([], "scalar_aggregate", "count") is None


def test_error_result_has_no_groundable_number():
    data = [{"error": "Could not safely determine the authorized scope for this query."}]
    assert _compute_deterministic_aggregate(data, "scalar_aggregate", "count") is None


def test_scalar_aggregate_never_reads_an_arbitrary_numeric_column():
    """Only ever reads the column identified by aggregate_alias (itself
    derived from plan.operation, not a name guess) -- a coincidentally
    numeric column with a different name must never be picked instead."""
    data = [{"count": 45, "room_number": 101}]
    assert _compute_deterministic_aggregate(data, "scalar_aggregate", "count") == 45


# ── _ground_numeric_answer ───────────────────────────────────────────────────

def test_correct_answer_is_left_untouched():
    answer = "There are **45** students in your school."
    assert _ground_numeric_answer(answer, 45) == answer


def test_the_exact_original_incident_is_corrected_in_place():
    """The literal reproduction of the reported bug: given the real grouped
    data (sums to 45), the live model deterministically produced this exact
    wrong answer across 5/5 repeated runs. Grounding must surgically fix the
    one wrong number while preserving the LLM's own formatting/wording."""
    hallucinated = "### Number of Students in My School\n\nThere are **10** students in my school."
    grounded = _ground_numeric_answer(hallucinated, 45)
    assert grounded == "### Number of Students in My School\n\nThere are **45** students in my school."


def test_wrong_single_bolded_number_is_replaced_surgically():
    answer = "There are **12** students enrolled this term."
    assert _ground_numeric_answer(answer, 45) == "There are **45** students enrolled this term."


def test_zero_bolded_numbers_falls_back_to_minimal_statement():
    answer = "Quite a few students attend this school."
    assert _ground_numeric_answer(answer, 45) == "**45**"


def test_multiple_bolded_numbers_cannot_be_disambiguated_falls_back():
    """Ambiguous which bolded number was meant to be the answer -- must not
    guess which one to replace; falls back rather than risk leaving a wrong
    number standing next to the correct one."""
    answer = "There were **12** students last year and **18** this year."
    assert _ground_numeric_answer(answer, 45) == "**45**"


def test_comma_formatted_correct_number_is_recognized():
    answer = "There are **1,045** students across the district."
    assert _ground_numeric_answer(answer, 1045) == answer


def test_decimal_percentage_value_is_recognized_when_correct():
    answer = "Your attendance is **87.5**% this term."
    assert _ground_numeric_answer(answer, Decimal("87.5")) == answer


# ── Principal-engineer review (2026-09-02): decimal-precision tolerance ─────
# MariaDB returns PERCENTAGE as a high-precision Decimal (verified live:
# "50.47619", not a clean "50.5") -- reachable today via a plain, ungrouped
# "what percentage..." question (operation=percentage with group_by unset is
# a valid QueryPlanValidator-accepted shape). A bare string-equality check
# would treat a correct, reasonably-rounded LLM answer as "wrong" and
# corrupt it into the ugly raw value -- reproduced live before this fix.

def test_reasonably_rounded_percentage_is_recognized_as_correct():
    """The exact live-reproduced defect: MariaDB's raw 50.47619 rounded by
    the LLM to a natural 50.48% must NOT be treated as wrong."""
    answer = "Your attendance percentage is **50.48**%."
    assert _ground_numeric_answer(answer, Decimal("50.47619")) == answer


def test_zero_decimal_rounded_percentage_is_recognized_as_correct():
    answer = "Your attendance percentage is about **50**%."
    assert _ground_numeric_answer(answer, Decimal("50.47619")) == answer


def test_exact_high_precision_percentage_is_still_recognized():
    answer = "Your attendance percentage is **50.47619**%."
    assert _ground_numeric_answer(answer, Decimal("50.47619")) == answer


def test_genuinely_wrong_percentage_is_still_corrected_despite_tolerance():
    """The rounding tolerance must never let an actually wrong number slip
    through -- 60% is not a rounding of 50.47619% at any precision."""
    answer = "Your attendance percentage is **60**%."
    assert _ground_numeric_answer(answer, Decimal("50.47619")) == "Your attendance percentage is **50.47619**%."


def test_genuinely_wrong_integer_count_is_not_excused_by_tolerance():
    """Integer COUNTs have no fractional part to round away -- a wrong
    integer must still be corrected, exactly as before this fix."""
    answer = "There are **44** students."
    assert _ground_numeric_answer(answer, 45) == "There are **45** students."
