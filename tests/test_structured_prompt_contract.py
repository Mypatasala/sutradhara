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


# ── HOMEWORK BY_SUBJECT grouping reachability (2026-09-08) ──────────────────
# The backend (query_registry.py's HOMEWORK.supported_groupings gaining
# BY_SUBJECT, reusing the same homework.subject column the existing SUBJECT
# lookup filter already validates) landed alongside this prompt change --
# these tests isolate the HOMEWORK bullet specifically, same reasoning as
# the SUBJECT-filter tests above, and specifically prove the filter and the
# grouping are documented as two DIFFERENT things, not conflated.

def test_prompt_homework_bullet_documents_by_subject_grouping():
    bullet = _homework_bullet()
    assert "by_subject" in bullet


def test_prompt_homework_bullet_distinguishes_subject_filter_from_by_subject_grouping():
    """The bullet must show both the ONE-named-subject filter shape and the
    every-subject breakdown grouping shape, clearly distinguished -- not
    just mention "by_subject" once with no worked example."""
    bullet = _homework_bullet()
    assert "How many homework assignments are there for Mathematics?" in bullet
    assert '"field": "subject", "value": "Mathematics"' in bullet
    assert "How many homework assignments are there by subject?" in bullet
    assert "group_by=by_subject" in bullet


def test_prompt_grouping_line_by_subject_now_includes_homework():
    """Regression/update: the shared GROUPING line's by_subject annotation
    must now name both entities that support it, not just course_schedule."""
    idx = _PROMPT.index("GROUPING (group_by):")
    grouping_line = _PROMPT[idx: _PROMPT.index("\n\n", idx)]
    assert "by_subject (course_schedule or homework)" in grouping_line


# ── USERS COUNT/ROLE reachability (2026-09-07) ───────────────────────────────
# The backend (query_registry.py's USERS.supported_operations gaining COUNT,
# and USERS.lookup_filter_fields[ROLE]) landed in ee8d3e4 (P0-1), but the
# prompt still claimed "Supports: list only" -- not merely silent like the
# HOMEWORK SUBJECT gap above, but actively WRONG, which could steer the
# model away from COUNT even where it might otherwise have guessed
# correctly. These tests isolate the USERS bullet specifically, same
# reasoning as _homework_bullet() above.

def _users_bullet() -> str:
    """Extracts just the "- users -- ..." bullet's text (now multi-line,
    including its worked example), up to the next "\\n- " bullet marker."""
    start = _PROMPT.index("- users --")
    end = _PROMPT.index("\n- ", start + 1)
    return _PROMPT[start:end]


def test_prompt_users_bullet_documents_count_operation():
    bullet = _users_bullet()
    assert "count" in bullet


def test_prompt_users_bullet_no_longer_claims_list_only():
    """Positive regression check: the stale, actively-wrong "list only"
    claim must be gone, not merely supplemented."""
    assert "list only" not in _users_bullet()


def test_prompt_users_bullet_documents_role_filter():
    bullet = _users_bullet()
    assert "role" in bullet


def test_prompt_users_role_described_as_dynamic_lookup_not_fixed_list():
    bullet = _users_bullet()
    assert "dynamic lookup" in bullet
    assert "not a fixed list" in bullet


def test_prompt_users_bullet_still_documents_profile_fields_unchanged():
    """Regression: the pre-existing profile-field wording must survive this
    addition unchanged."""
    bullet = _users_bullet()
    assert "staff/self profile fields (name, email, phone, department)" in bullet


def test_prompt_users_includes_teacher_count_worked_example():
    """The exact motivating case: 'How many teachers are there?' must map
    to entity=users, operation=count, filters=[role=teacher], and this must
    be attached to the USERS bullet, not found elsewhere in the prompt."""
    bullet = _users_bullet()
    assert "How many teachers are there?" in bullet
    assert "entity=users" in bullet
    assert "operation=count" in bullet
    assert '"field": "role", "value": "teacher"' in bullet


# ── REPORT_CARDS AVERAGE reachability (2026-09-07, Phase 3) ─────────────────
# The backend (query_registry.py's REPORT_CARDS.supported_operations gaining
# AVERAGE + numeric_agg_fields[OVERALL_PERCENTAGE], and
# structured_sql_builder.py's AVG(...) branch) landed in Phase 1/2, but the
# prompt never mentioned aggregate_target or average at all -- the same
# "fully supported but practically unreachable" gap as HOMEWORK SUBJECT and
# USERS ROLE/COUNT above. These tests isolate the REPORT_CARDS bullet
# specifically, same reasoning as _homework_bullet()/_users_bullet() above.

def _report_cards_bullet() -> str:
    """Extracts just the "- report_cards -- ..." bullet's text (now
    multi-line, including its worked example), up to the next "\\n- "
    bullet marker."""
    start = _PROMPT.index("- report_cards --")
    end = _PROMPT.index("\n- ", start + 1)
    return _PROMPT[start:end]


def test_prompt_report_cards_bullet_documents_average_operation():
    bullet = _report_cards_bullet()
    assert "average" in bullet


def test_prompt_report_cards_bullet_documents_aggregate_target_required():
    bullet = _report_cards_bullet()
    assert "aggregate_target" in bullet
    assert "requires aggregate_target" in bullet


def test_prompt_report_cards_bullet_documents_overall_percentage_as_only_target():
    bullet = _report_cards_bullet()
    assert "overall_percentage" in bullet
    assert "ONLY supported aggregate_target" in bullet


def test_prompt_report_cards_bullet_still_documents_list_and_sort_unchanged():
    """Regression: the pre-existing list/sort-by-issue_date wording must
    survive this addition unchanged."""
    bullet = _report_cards_bullet()
    assert "a student's own report cards" in bullet
    assert "sort by" in bullet and "issue_date" in bullet
    assert 'sort issue_date desc, limit 1' in bullet


def test_prompt_report_cards_includes_average_grade_worked_example():
    """The exact motivating case: 'What is the average grade?' must map to
    entity=report_cards, operation=average, aggregate_target=
    overall_percentage, and this must be attached to the REPORT_CARDS
    bullet, not found elsewhere in the prompt."""
    bullet = _report_cards_bullet()
    assert "What is the average grade?" in bullet
    assert "entity=report_cards" in bullet
    assert "operation=average" in bullet
    assert "aggregate_target=overall_percentage" in bullet


def test_prompt_report_cards_does_not_expose_sum_as_supported():
    """SUM must remain unsupported and undocumented -- confirms the prompt
    change didn't accidentally also expose it."""
    bullet = _report_cards_bullet()
    assert "sum" not in bullet.lower()


def test_prompt_operations_line_documents_average_and_its_target_requirement():
    """The shared OPERATIONS line (not entity-specific) must also mention
    average and that it requires aggregate_target, mirroring how percentage
    documents its own required companion field (percentage_of) there."""
    idx = _PROMPT.index("OPERATIONS:")
    operations_line = _PROMPT[idx: _PROMPT.index("\n\n", idx)]
    assert "average" in operations_line
    assert "aggregate_target" in operations_line
    assert "sum is not supported" in operations_line


def test_prompt_percentage_and_count_documentation_unaffected_by_average_addition():
    """Regression: the pre-existing percentage_of/COUNT documentation in
    the shared OPERATIONS line must survive this addition unchanged."""
    idx = _PROMPT.index("OPERATIONS:")
    operations_line = _PROMPT[idx: _PROMPT.index("\n\n", idx)]
    assert "count, list, percentage" in operations_line
    assert "requires percentage_of" in operations_line


# ── Remaining backend-implemented-but-prompt-unreachable capabilities ───────
# (2026-09-07, post-AVERAGE audit follow-up): YESTERDAY/LAST_7_DAYS,
# ATTENDANCE.BY_STATUS, HOMEWORK.BY_STATUS, REPORT_CARDS.COUNT/BY_TERM,
# COURSE_SCHEDULE.COUNT/BY_DAY_OF_WEEK were all already registry/builder-
# supported (see query_registry.py) but never mentioned in the prompt --
# the same "fully supported but practically unreachable" gap as every prior
# reachability fix in this file. Prompt-only change; no backend/registry/
# validator/builder/security file touched.

def _attendance_bullet() -> str:
    start = _PROMPT.index("- attendance --")
    end = _PROMPT.index("\n- ", start + 1)
    return _PROMPT[start:end]


def _course_schedule_bullet() -> str:
    start = _PROMPT.index("- course_schedule --")
    end = _PROMPT.index("\n- ", start + 1)
    return _PROMPT[start:end]


def _courses_bullet() -> str:
    """COURSES is currently the LAST entity bullet (no trailing "\\n- "
    marker) -- bounded instead by the blank line before OPERATIONS:."""
    start = _PROMPT.index("- courses --")
    end = _PROMPT.index("\n\n", start)
    return _PROMPT[start:end]


def _date_range_section() -> str:
    start = _PROMPT.index("DATE_RANGE:")
    end = _PROMPT.index("\n\n", start)
    return _PROMPT[start:end]


# -- 1/2: YESTERDAY / LAST_7_DAYS in DATE_RANGE --

def test_prompt_date_range_documents_yesterday():
    section = _date_range_section()
    assert "yesterday" in section


def test_prompt_date_range_documents_last_7_days():
    section = _date_range_section()
    assert "last_7_days" in section


def test_prompt_date_range_last_7_days_distinguished_from_last_week():
    """Same silent-conflation guard already proven for last_30_days --
    last_7_days must be explicitly distinguished from last_week in the
    prompt text, not just listed."""
    section = _date_range_section()
    assert "last_week" in section and "last_7_days" in section
    # "last_7_days" appears twice (the enumeration, then its own worked
    # explanation) -- find the occurrence whose nearby text actually
    # explains the distinction, not just the first (enumeration) match.
    start = 0
    found = False
    while True:
        idx = section.find("last_7_days", start)
        if idx == -1:
            break
        if "last_week" in section[idx: idx + 300]:
            found = True
            break
        start = idx + 1
    assert found, "no last_7_days occurrence found near a last_week distinction"


def test_prompt_date_range_existing_values_unaffected():
    """Regression: every pre-existing enumerated value and the last_30_days
    worked-example wording must survive unchanged."""
    section = _date_range_section()
    for value in ["all_time", "today", "this_week", "last_week", "this_month",
                  "last_month", "this_year", "last_year", "last_30_days"]:
        assert value in section
    assert "rolling 30-day window" in section


# -- 3: ATTENDANCE.BY_STATUS --

def test_prompt_attendance_bullet_documents_by_status_grouping():
    bullet = _attendance_bullet()
    assert "by_status" in bullet


def test_prompt_attendance_bullet_includes_by_status_worked_example():
    bullet = _attendance_bullet()
    assert "How many attendance records are there by status?" in bullet
    assert "group_by=by_status" in bullet


def test_prompt_attendance_bullet_still_documents_by_student_and_status_filter_unchanged():
    """Regression: pre-existing status-filter and by_student wording must
    survive this addition unchanged."""
    bullet = _attendance_bullet()
    assert "present/absent/late/excused" in bullet
    assert "by_student" in bullet
    assert "operation=list shows individual attendance records" in bullet


# -- 4: HOMEWORK.BY_STATUS --

def test_prompt_homework_bullet_documents_by_status_grouping():
    bullet = _homework_bullet()
    assert "by_status" in bullet


def test_prompt_homework_bullet_includes_by_status_worked_example():
    bullet = _homework_bullet()
    assert "How many homework assignments are pending vs graded?" in bullet
    assert "group_by=by_status" in bullet


# -- ASSIGNMENTS Phase 1 (2026-09-08): COUNT, LIST, status filter, --
# BY_STATUS grouping. Reachability tests mirror _homework_bullet()'s
# pattern; wording assertions specifically prove the bullet describes
# row-level semantics ("rows in the assignments table"), never a
# per-student assignment count.

def _assignments_bullet() -> str:
    start = _PROMPT.index("- assignments --")
    end = _PROMPT.index("\n- ", start + 1)
    return _PROMPT[start:end]


def test_prompt_assignments_bullet_describes_row_level_semantics_not_per_student():
    bullet = _assignments_bullet()
    assert "rows in the assignments table" in bullet
    assert "NOT per-student" in bullet


def test_prompt_assignments_bullet_documents_count():
    bullet = _assignments_bullet()
    assert "count" in bullet


def test_prompt_assignments_bullet_documents_list():
    bullet = _assignments_bullet()
    assert "list" in bullet


def test_prompt_assignments_bullet_documents_status_filter():
    bullet = _assignments_bullet()
    assert "not_started/in_progress/submitted/graded/overdue" in bullet


def test_prompt_assignments_bullet_documents_by_status_grouping():
    bullet = _assignments_bullet()
    assert "by_status" in bullet
    assert "How many assignments are there by status?" in bullet
    assert "group_by=by_status" in bullet


def test_prompt_assignments_bullet_states_no_grade_or_average_support():
    """Scope guard: no date filtering and no grade/points/average -- see
    query_registry.py's ASSIGNMENTS entry. Phase 1 also stated "no subject/
    course filter" here; Phase 2 (2026-09-10) added exactly that filter --
    see the SUBJECT-specific tests below for its own scope guard (no
    by_subject grouping)."""
    bullet = _assignments_bullet()
    assert "No date filter" in bullet
    assert "grade/points/average support" in bullet


# -- ASSIGNMENTS SUBJECT filter (Phase 2, 2026-09-10) -- reachability tests
# mirror _homework_bullet()'s SUBJECT-filter pattern; wording assertions
# specifically prove course/subject equivalence is documented and no
# by_subject grouping is exposed.

def test_prompt_assignments_bullet_documents_subject_filter():
    bullet = _assignments_bullet()
    assert "by subject" in bullet


def test_prompt_assignments_bullet_documents_course_subject_equivalence():
    bullet = _assignments_bullet()
    assert '"subject" and "course" are the same thing' in bullet


def test_prompt_assignments_bullet_includes_subject_filter_worked_example():
    bullet = _assignments_bullet()
    assert "Show Mathematics assignments." in bullet
    assert '"field": "subject", "value": "Mathematics"' in bullet


def test_prompt_assignments_bullet_does_not_expose_by_subject_grouping():
    """Scope guard: Phase 2 adds a SUBJECT filter, not a BY_SUBJECT
    grouping -- the bullet explicitly disclaims it (rather than merely
    omitting it) so the model isn't left to guess. No worked example uses
    group_by=by_subject for assignments."""
    bullet = _assignments_bullet()
    assert "no by_subject grouping" in bullet
    assert "group_by=by_subject" not in bullet


def test_prompt_grouping_line_documents_assignments_by_status():
    idx = _PROMPT.index("GROUPING (group_by):")
    grouping_line = _PROMPT[idx: _PROMPT.index("\n\n", idx)]
    assert "assignments" in grouping_line


# -- COURSES Phase 1 (2026-09-10): COUNT and LIST only. Reachability tests
# mirror _assignments_bullet()'s pattern; the scope-guard test proves no
# unsupported capability (term/section/instructor/name/code filtering,
# sorting, enrollment_count) is documented.

def test_prompt_courses_bullet_documents_count_and_list():
    bullet = _courses_bullet()
    assert "count" in bullet
    assert "list" in bullet


def test_prompt_courses_bullet_includes_count_and_list_worked_examples():
    bullet = _courses_bullet()
    assert "How many courses are offered?" in bullet
    assert "List the courses." in bullet


def test_prompt_courses_bullet_distinguishes_from_course_schedule():
    """Scope guard: courses (the offered-course entity) must not be
    conflated with course_schedule (the timetable) or with a per-student
    enrollment count -- both explicitly disclaimed in the bullet."""
    bullet = _courses_bullet()
    assert "NOT course_schedule" in bullet
    assert "NOT a per-student enrollment count" in bullet


def test_prompt_courses_bullet_documents_no_unsupported_capability():
    """Scope guard: term/section/instructor/name/code filtering, sorting,
    and enrollment_count must never be documented as a SUPPORTED capability
    for courses this phase -- see query_registry.py's COURSES entry for why
    each is excluded. "enrollment" itself legitimately appears once, in the
    bullet's own disclaimer that courses is NOT a per-student enrollment
    count -- checked for the column name specifically instead."""
    bullet = _courses_bullet()
    for unsupported in ("term", "semester", "section", "instructor", "sort", "enrollment_count"):
        assert unsupported not in bullet.lower()


# -- 5/6: REPORT_CARDS.COUNT / BY_TERM --

def test_prompt_report_cards_bullet_documents_count_operation():
    bullet = _report_cards_bullet()
    assert "list, count, average" in bullet


def test_prompt_report_cards_bullet_documents_by_term_grouping():
    bullet = _report_cards_bullet()
    assert "by_term" in bullet


def test_prompt_report_cards_bullet_includes_count_by_term_worked_example():
    bullet = _report_cards_bullet()
    assert "How many report cards were issued each term?" in bullet
    assert "group_by=by_term" in bullet


def test_prompt_report_cards_average_wording_still_intact():
    """Regression: this task's edits must not disturb the existing AVERAGE
    documentation verified by the Phase 3 tests above."""
    bullet = _report_cards_bullet()
    assert "average" in bullet
    assert "aggregate_target=overall_percentage" in bullet
    assert "ONLY supported aggregate_target value" in bullet
    assert "no gpa" in bullet


# -- REPORT_CARDS TERM filter (Phase 1, 2026-09-10) -- reachability tests
# mirror _homework_bullet()'s SUBJECT-filter pattern; wording assertions
# specifically prove TERM is documented as a dynamic lookup (real per-school
# data), never a fixed enum/vocabulary.

def test_prompt_report_cards_bullet_documents_term_filter():
    bullet = _report_cards_bullet()
    assert "filter by term" in bullet


def test_prompt_report_cards_term_described_as_dynamic_lookup_not_fixed_list():
    bullet = _report_cards_bullet()
    assert "dynamic lookup filter" in bullet
    assert "not\n  a fixed list" in bullet or "not a fixed list" in bullet


def test_prompt_report_cards_bullet_distinguishes_term_filter_from_by_term_grouping():
    """The bullet must show both the ONE-named-term filter shape and the
    every-term breakdown grouping shape, clearly distinguished -- not just
    mention "term" once with no worked example, same FILTER-vs-GROUPING
    proof already established for HOMEWORK.SUBJECT/BY_SUBJECT."""
    bullet = _report_cards_bullet()
    assert "Show Term 2 report cards." in bullet
    assert '"field": "term", "value": "Term 2"' in bullet
    assert "How many report cards were issued each term?" in bullet
    assert "group_by=by_term" in bullet


def test_prompt_report_cards_bullet_includes_term_filter_combined_with_average_worked_example():
    """The real motivating case ('average grade in Term 2') must be
    reachable, not just a bare filter -- proves TERM composes with AVERAGE
    in the documented worked examples, not merely in the backend."""
    bullet = _report_cards_bullet()
    assert "What is the average grade in Term 2?" in bullet
    assert "aggregate_target=overall_percentage, filters=" in bullet


def test_prompt_report_cards_bullet_does_not_expose_term_as_fixed_vocabulary():
    """Scope guard: no hardcoded term value list (e.g. 'Term 1, Term 2,
    Final') may ever appear in the bullet -- term values are real,
    per-school, dynamically-validated data, never a closed enum the model
    is told to pick from."""
    bullet = _report_cards_bullet()
    assert "Term 1, Term 2" not in bullet
    assert "Term 1/Term 2" not in bullet


# -- 7/8: COURSE_SCHEDULE.COUNT / BY_DAY_OF_WEEK --

def test_prompt_course_schedule_bullet_documents_count_operation():
    bullet = _course_schedule_bullet()
    assert "list, count" in bullet
    assert "list only" not in bullet


def test_prompt_course_schedule_bullet_documents_by_day_of_week_grouping():
    bullet = _course_schedule_bullet()
    assert "by_day_of_week" in bullet


def test_prompt_course_schedule_bullet_includes_count_worked_example():
    bullet = _course_schedule_bullet()
    assert "How many classes are scheduled on Mondays?" in bullet
    assert '"field": "day_of_week", "value": "Monday"' in bullet


def test_prompt_course_schedule_bullet_still_documents_subject_and_day_filter_unchanged():
    """Regression: pre-existing day_of_week/subject filter and by_subject
    grouping wording must survive this addition unchanged."""
    bullet = _course_schedule_bullet()
    assert "day_of_week" in bullet
    assert "by subject" in bullet
    assert "by_subject" in bullet
    assert "timetable" in bullet


# -- COURSE_SCHEDULE start_time sorting (2026-09-10): the registry/validator/
# builder already fully supported SortField.START_TIME for this entity, but
# the structured prompt never documented it -- these tests prove the
# capability is now actually reachable, not merely mechanically possible.

def test_prompt_course_schedule_bullet_documents_start_time_sort():
    bullet = _course_schedule_bullet()
    assert "start_time" in bullet


def test_prompt_course_schedule_bullet_includes_start_time_sort_worked_example():
    bullet = _course_schedule_bullet()
    assert "Show today's schedule in order." in bullet
    assert '"field": "start_time", "direction": "asc"' in bullet


def test_prompt_course_schedule_bullet_start_time_sort_implies_no_new_filter_or_grouping():
    """Scope guard: the new sort wording must not add or imply any new
    filter/grouping capability -- day_of_week/subject filters and
    by_subject/by_day_of_week groupings remain the only ones documented."""
    bullet = _course_schedule_bullet()
    assert "group by_subject, or by_day_of_week" in bullet
    assert "filter\n  by day_of_week, or by subject" in bullet


# -- 10/11: shared GROUPING line now annotates entity scope --

def test_prompt_grouping_line_annotates_by_status_and_by_day_of_week_and_by_term_scope():
    idx = _PROMPT.index("GROUPING (group_by):")
    grouping_line = _PROMPT[idx: _PROMPT.index("\n\n", idx)]
    assert "by_status (attendance, homework, or assignments)" in grouping_line
    assert "by_day_of_week\n(course_schedule only)" in grouping_line or "by_day_of_week (course_schedule only)" in grouping_line
    assert "by_term (report_cards only)" in grouping_line


def test_prompt_grouping_line_existing_annotations_unaffected():
    """Regression: pre-existing by_class/by_student annotations must
    survive unchanged."""
    idx = _PROMPT.index("GROUPING (group_by):")
    grouping_line = _PROMPT[idx: _PROMPT.index("\n\n", idx)]
    assert "by_class (students only)" in grouping_line
    assert "by_student (attendance only)" in grouping_line


# ── Grouped COUNT ranking/extreme reachability (2026-09-07) ────────────────
# The backend/lifecycle already supports COUNT+group_by+extreme for
# ATTENDANCE.BY_STATUS, REPORT_CARDS.BY_TERM, and COURSE_SCHEDULE.
# BY_DAY_OF_WEEK (see the regression tests in test_extreme_selection.py and
# test_summarize_grounding_integration.py), but every pre-existing RANKING
# worked example was attendance/percentage/by_student-specific -- the model
# had no textual basis to generalize extreme to a different operation/
# grouping. These tests isolate the RANKING section specifically.

def _ranking_section() -> str:
    """Extracts the full RANKING section, from its own heading up to the
    next top-level "DATE_RANGE:" section marker."""
    start = _PROMPT.index("RANKING --")
    end = _PROMPT.index("DATE_RANGE:")
    return _PROMPT[start:end]


def test_prompt_ranking_section_documents_attendance_by_status_extreme():
    section = _ranking_section()
    assert "Which attendance status has the most records?" in section
    assert "entity=attendance" in section
    assert "operation=count" in section
    assert "group_by=by_status" in section
    assert "extreme=highest" in section


def test_prompt_ranking_section_documents_report_cards_by_term_extreme():
    section = _ranking_section()
    assert "Which term had the most report cards?" in section
    assert "entity=report_cards" in section
    assert "group_by=by_term" in section
    assert "extreme=highest" in section


def test_prompt_ranking_section_documents_course_schedule_by_day_of_week_extreme():
    section = _ranking_section()
    assert "Which day of the week has the most scheduled classes?" in section
    assert "entity=course_schedule" in section
    assert "group_by=by_day_of_week" in section
    assert "extreme=highest" in section


def test_prompt_ranking_section_generalizes_beyond_attendance_percentage():
    """The new examples must be introduced as a generalization, not just
    three more isolated cases -- otherwise the model has no reason to
    extend the pattern to a fourth, not-yet-demonstrated entity/grouping
    later."""
    section = _ranking_section()
    assert "not limited to attendance/percentage/by_student" in section


def test_prompt_ranking_section_does_not_imply_average_ranking_is_new():
    """Guard against scope creep: this task only documents COUNT ranking
    over the three new groupings -- it must not introduce or imply new
    AVERAGE-ranking guidance (REPORT_CARDS.AVERAGE+extreme was already
    correct and tested before this change; this task doesn't touch it)."""
    section = _ranking_section()
    assert "operation=average" not in section


def test_prompt_ranking_section_existing_examples_unaffected():
    """Regression: all five pre-existing worked examples (attendance/
    percentage/by_student extreme, explicit sort+limit, and the no-ranking
    contrast case) must survive this addition verbatim."""
    section = _ranking_section()
    assert 'Q: "Which students have the lowest attendance?"' in section
    assert 'Q: "Who has the highest attendance?"' in section
    assert 'Q: "Show the 5 students with the lowest attendance."' in section
    assert 'Q: "List the 3 students with the highest attendance."' in section
    assert 'Q: "What is my attendance percentage?"' in section
    assert "extreme and sort/limit are mutually exclusive" in section


def test_prompt_ranking_section_rule_text_unaffected():
    """Regression: the core extreme-vs-sort+limit rule paragraph (the two
    bullet points above the worked examples) must survive unchanged."""
    section = _ranking_section()
    assert '"lowest/highest" vs "top/bottom N" are DIFFERENT questions, never guess a number' in section
    assert "Never invent a limit when no number was stated" in section
