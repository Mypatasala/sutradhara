package sutradhara.roles.admin

import future.keywords.if

default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Admin access denied: table not in allowed list."
}

allowed_actions := {"select", "aggregate"}

allowed_tables := {
    "users", "students", "teachers", "courses",
    "enrollments", "attendance", "report_cards",
    "assignments", "timetable", "departments"
}

# Teachers table: scoped to the admin's department
decision := {
    "authorized": true,
    "columns": [],
    "filter": sprintf("department_id = %v", [input.user.department_id])
} if {
    input.target_table == "teachers"
    input.action in allowed_actions
    input.user.department_id
}

# Departments table: only the admin's own department row
decision := {
    "authorized": true,
    "columns": [],
    "filter": sprintf("id = %v", [input.user.department_id])
} if {
    input.target_table == "departments"
    input.action in allowed_actions
    input.user.department_id
}

# All other allowed tables: full school-wide read access (no row filter)
decision := {
    "authorized": true,
    "columns": [],
    "filter": ""
} if {
    input.target_table in (allowed_tables - {"teachers", "departments"})
    input.action in allowed_actions
}

