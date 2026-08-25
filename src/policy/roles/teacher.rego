package sutradhara.roles.teacher

import future.keywords.if

default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Teacher access restricted to courses and assigned students."
}

allowed_actions := {"select", "aggregate"}

# Teachers can see all users (masked columns for PII)
decision := {
    "authorized": true,
    "columns": ["id", "name", "role", "department_id"],
    "filter": ""
} if {
    input.target_table == "users"
    input.action in allowed_actions
}

# Teachers can see attendance for students in THEIR courses only
decision := {
    "authorized": true,
    "columns": ["id", "student_id", "date", "status"],
    "filter": sprintf("student_id IN (SELECT student_id FROM enrollments WHERE course_id IN (SELECT id FROM courses WHERE teacher_id = %v))", [input.user.id])
} if {
    input.target_table == "attendance"
    input.action in allowed_actions
}

# Teachers can see courses they teach
decision := {
    "authorized": true,
    "columns": ["id", "name", "description", "teacher_id"],
    "filter": sprintf("teacher_id = %v", [input.user.id])
} if {
    input.target_table == "courses"
    input.action in allowed_actions
}
