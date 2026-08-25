package sutradhara.roles.parent

test_parent_attendance_allowed if {
    decision.authorized == true
    decision.filter == "student_id = 42"
} with input as {
    "user": {"role": "parent", "id": 10, "linked_student_id": 42},
    "action": "select",
    "target_table": "attendance"
}

test_parent_report_cards_allowed if {
    decision.authorized == true
    decision.filter == "student_id = 42"
} with input as {
    "user": {"role": "parent", "id": 10, "linked_student_id": 42},
    "action": "select",
    "target_table": "report_cards"
}

test_parent_courses_allowed if {
    decision.authorized == true
    decision.filter == ""
} with input as {
    "user": {"role": "parent", "id": 10, "linked_student_id": 42},
    "action": "select",
    "target_table": "courses"
}

test_parent_users_denied if {
    decision.authorized == false
} with input as {
    "user": {"role": "parent", "id": 10, "linked_student_id": 42},
    "action": "select",
    "target_table": "users"
}

test_parent_without_student_denied if {
    decision.authorized == false
} with input as {
    "user": {"role": "parent", "id": 10},
    "action": "select",
    "target_table": "attendance"
}

test_parent_write_denied if {
    decision.authorized == false
} with input as {
    "user": {"role": "parent", "id": 10, "linked_student_id": 42},
    "action": "insert",
    "target_table": "attendance"
}
