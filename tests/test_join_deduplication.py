"""
Direct unit tests on StructuredSQLBuilder._collect_joins -- deliberately
bypassing QueryPlan/QueryPlanValidator entirely. The LIST + group_by
validator rule (see test_query_validator.py) makes the originally-reproduced
wrapping plan (operation=list, group_by=by_subject, a subject filter) itself
unreachable through the normal pipeline now -- but the builder-level fix
must still be proven directly, since a future entity with an aggregate
operation and this same join-overlap shape would hit it again otherwise.
"""

from src.agents.query_plan import (
    ComparisonFilter,
    Entity,
    FilterField,
    GroupingDimension,
    LookupFilterField,
    Operation,
    QueryPlan,
    RelativeDate,
)
from src.agents.query_registry import REGISTRY, EntityMeta, JoinStep, LookupFilterFieldMeta
from src.retrieval.structured_sql_builder import _collect_joins, _join_sql


def test_duplicate_courses_join_deduplicated():
    """The exact previously-reproduced case: COURSE_SCHEDULE's BY_SUBJECT
    grouping path and a SUBJECT lookup filter's main_query_join_path both
    require course_schedule -> courses. Constructing the plan directly
    (bypassing the validator, which now rejects the wrapping LIST+group_by
    combination) to prove the builder itself never emits the join twice."""
    meta = REGISTRY[Entity.COURSE_SCHEDULE]
    plan = QueryPlan.model_construct(
        entity=Entity.COURSE_SCHEDULE,
        operation=Operation.LIST,  # bypassing validation entirely via model_construct
        group_by=GroupingDimension.BY_SUBJECT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="Mathematics")],
        display_fields=[], percentage_of=None, date_range=RelativeDate.ALL_TIME, sort=None, limit=None,
        distinct=False, can_answer=True, unresolved_reason=None, clarification_question=None,
    )

    joins = _collect_joins(plan, meta, meta.table)
    joins_sql = _join_sql(joins)

    assert joins_sql.count("JOIN courses") == 1, f"expected exactly one JOIN courses, got: {joins_sql!r}"
    assert joins_sql == " JOIN courses ON course_schedule.course_id = courses.id"


def test_two_different_joins_to_the_same_target_table_are_both_preserved():
    """Regression proving the deduplication is keyed on FULL join identity
    (anchor + table + both columns), not target-table name alone: two
    genuinely different relationships that both happen to join the same
    target table must both survive, unmerged."""
    fake_meta = EntityMeta(
        table="course_schedule",
        supported_operations={Operation.LIST},
        supported_groupings={
            # Two different grouping dimensions, both joining "courses" --
            # but via DIFFERENT columns, i.e. genuinely different join
            # semantics, not the same relationship expressed twice.
            GroupingDimension.BY_SUBJECT: REGISTRY[Entity.COURSE_SCHEDULE].supported_groupings[GroupingDimension.BY_SUBJECT],
        },
        lookup_filter_fields={
            LookupFilterField.SUBJECT: LookupFilterFieldMeta(
                column="courses.name",
                lookup_table="courses",
                lookup_column="name",
                # Deliberately a DIFFERENT join condition than BY_SUBJECT's
                # own (different left_column) -- a different relationship to
                # the same target table, e.g. a hypothetical "recommended
                # alternate course" FK rather than the scheduled course FK.
                main_query_join_path=[JoinStep(table="courses", left_column="alternate_course_id", right_column="id")],
                existence_check_join_path=[],
                school_id_column="school_id",
            ),
        },
    )
    plan = QueryPlan.model_construct(
        entity=Entity.COURSE_SCHEDULE,
        operation=Operation.LIST,
        group_by=GroupingDimension.BY_SUBJECT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="Mathematics")],
        display_fields=[], percentage_of=None, date_range=RelativeDate.ALL_TIME, sort=None, limit=None,
        distinct=False, can_answer=True, unresolved_reason=None, clarification_question=None,
    )

    joins = _collect_joins(plan, fake_meta, fake_meta.table)
    joins_sql = _join_sql(joins)

    # Both relationships to "courses" must be present -- via course_id (the
    # grouping's own join) AND via alternate_course_id (the filter's own,
    # genuinely different join) -- neither one dropped.
    assert joins_sql.count("JOIN courses") == 2, f"expected both distinct joins preserved, got: {joins_sql!r}"
    assert "ON course_schedule.course_id = courses.id" in joins_sql
    assert "ON course_schedule.alternate_course_id = courses.id" in joins_sql


def test_identical_join_from_group_by_and_filter_deduplicated_but_order_preserved():
    """Sanity check on ordering: when the grouping path and a filter's join
    path really are identical, the single surviving join appears exactly
    once, in the position the grouping path (evaluated first) put it."""
    meta = REGISTRY[Entity.COURSE_SCHEDULE]
    plan = QueryPlan.model_construct(
        entity=Entity.COURSE_SCHEDULE, operation=Operation.LIST, group_by=GroupingDimension.BY_SUBJECT,
        filters=[ComparisonFilter(field=FilterField.SUBJECT, value="Mathematics")],
        display_fields=[], percentage_of=None, date_range=RelativeDate.ALL_TIME, sort=None, limit=None,
        distinct=False, can_answer=True, unresolved_reason=None, clarification_question=None,
    )
    joins = _collect_joins(plan, meta, meta.table)
    assert len(joins) == 1
    anchor, step = joins[0]
    assert anchor == "course_schedule"
    assert step.table == "courses"
