package sutradhara.roles.parent

import future.keywords.if

default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Parent access restricted to their linked student's records only."
}

allowed_actions := {"select", "aggregate"}

# Parents can view their linked student's attendance
decision := {
    "authorized": true,
    "columns": ["id", "student_id", "date", "status"],
    "filter": sprintf("student_id = %v", [input.user.linked_student_id])
} if {
    input.target_table == "attendance"
    input.action in allowed_actions
    input.user.linked_student_id
}

# Parents can view their linked student's report cards
decision := {
    "authorized": true,
    "columns": ["id", "student_id", "course_id", "grade", "comments"],
    "filter": sprintf("student_id = %v", [input.user.linked_student_id])
} if {
    input.target_table == "report_cards"
    input.action in allowed_actions
    input.user.linked_student_id
}

# Parents can view basic student profile (no parent_id exposed)
decision := {
    "authorized": true,
    "columns": ["user_id", "grade_level"],
    "filter": sprintf("user_id = %v", [input.user.linked_student_id])
} if {
    input.target_table == "students"
    input.action in allowed_actions
    input.user.linked_student_id
}

# Parents can view all courses (context for grades/attendance)
decision := {
    "authorized": true,
    "columns": ["id", "name", "description"],
    "filter": ""
} if {
    input.target_table == "courses"
    input.action in allowed_actions
}

# Parents can view their student's enrollments
decision := {
    "authorized": true,
    "columns": ["student_id", "course_id", "enrollment_date"],
    "filter": sprintf("student_id = %v", [input.user.linked_student_id])
} if {
    input.target_table == "enrollments"
    input.action in allowed_actions
    input.user.linked_student_id
}
