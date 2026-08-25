from typing import Dict, List, Optional


class SemanticMapper:
    """
    Maps business concepts (logical entities) to physical schema elements.
    Provides pre-approved join paths, column sensitivity labels, and enriched
    schema context for the intent resolution LLM.
    """

    # Business concept synonyms → physical table name
    CONCEPT_TABLE_MAP: Dict[str, str] = {
        "grade": "report_cards",
        "grades": "report_cards",
        "marks": "report_cards",
        "scores": "report_cards",
        "result": "report_cards",
        "results": "report_cards",
        "attendance": "attendance",
        "present": "attendance",
        "absent": "attendance",
        "student": "students",
        "students": "students",
        "pupil": "students",
        "teacher": "teachers",
        "teachers": "teachers",
        "instructor": "teachers",
        "course": "courses",
        "courses": "courses",
        "subject": "courses",
        "subjects": "courses",
        "class": "timetable",
        "classes": "timetable",
        "timetable": "timetable",
        "schedule": "timetable",
        "assignment": "assignments",
        "assignments": "assignments",
        "homework": "assignments",
        "enrollment": "enrollments",
        "enrollments": "enrollments",
        "enrolments": "enrollments",
        "user": "users",
        "users": "users",
        "department": "departments",
        "departments": "departments",
    }

    # Pre-approved safe join paths — prevents cartesian products and hallucinated joins
    SAFE_JOIN_PATHS: List[Dict[str, str]] = [
        {"from": "students",     "to": "users",       "on": "students.user_id = users.id"},
        {"from": "teachers",     "to": "users",       "on": "teachers.user_id = users.id"},
        {"from": "attendance",   "to": "students",    "on": "attendance.student_id = students.user_id"},
        {"from": "enrollments",  "to": "students",    "on": "enrollments.student_id = students.user_id"},
        {"from": "enrollments",  "to": "courses",     "on": "enrollments.course_id = courses.id"},
        {"from": "courses",      "to": "teachers",    "on": "courses.teacher_id = teachers.user_id"},
        {"from": "report_cards", "to": "students",    "on": "report_cards.student_id = students.user_id"},
        {"from": "report_cards", "to": "courses",     "on": "report_cards.course_id = courses.id"},
        {"from": "timetable",    "to": "courses",     "on": "timetable.course_id = courses.id"},
        {"from": "assignments",  "to": "courses",     "on": "assignments.course_id = courses.id"},
        {"from": "teachers",     "to": "departments", "on": "teachers.department_id = departments.id"},
    ]

    # Columns that carry PII and must never be surfaced to unauthorised roles
    SENSITIVE_COLUMNS: Dict[str, List[str]] = {
        "users":    ["email"],
        "students": ["parent_id"],
    }

    def resolve_concept(self, concept: str) -> Optional[str]:
        """Maps a loose business concept to its canonical physical table name."""
        return self.CONCEPT_TABLE_MAP.get(concept.lower())

    def get_join_path(self, from_table: str, to_table: str) -> Optional[Dict[str, str]]:
        """Returns the pre-approved join spec between two tables, or None if not whitelisted."""
        for path in self.SAFE_JOIN_PATHS:
            if path["from"] == from_table and path["to"] == to_table:
                return path
        return None

    def enrich_schema_summary(self, schema_summary: str) -> str:
        """Appends semantic hints to the raw schema summary before passing to the LLM."""
        hints = [
            "",
            "SEMANTIC HINTS (business concept → physical table):",
        ]
        for concept, table in self.CONCEPT_TABLE_MAP.items():
            hints.append(f"  - '{concept}' → '{table}'")

        hints += [
            "",
            "PRE-APPROVED JOIN PATHS (use only these joins):",
        ]
        for path in self.SAFE_JOIN_PATHS:
            hints.append(f"  - {path['from']} JOIN {path['to']} ON {path['on']}")

        return schema_summary + "\n".join(hints)
