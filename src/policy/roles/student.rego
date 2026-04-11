package sutradhara.roles.student

import future.keywords.if

default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Student access restricted to own records."
}

allowed_actions := {"select", "aggregate"}

# Students can see their own user profile
decision := {
    "authorized": true,
    "columns": ["id", "name", "email", "role"],
    "filter": sprintf("id = %v", [input.user.id])
} if {
    input.target_table == "users"
    input.action in allowed_actions
}

# Students can see their own attendance
decision := {
    "authorized": true,
    "columns": ["id", "student_id", "date", "status"],
    "filter": sprintf("student_id = %v", [input.user.id])
} if {
    input.target_table == "attendance"
    input.action in allowed_actions
}

# Students can see their own report cards
decision := {
    "authorized": true,
    "columns": ["id", "student_id", "course_id", "grade", "comments"],
    "filter": sprintf("student_id = %v", [input.user.id])
} if {
    input.target_table == "report_cards"
    input.action in allowed_actions
}

# Students can see all courses (read-only)
decision := {
    "authorized": true,
    "columns": ["id", "name", "description", "teacher_id", "department_id"],
    "filter": "" # No row filter for course list
} if {
    input.target_table == "courses"
    input.action in allowed_actions
}
