package sutradhara.roles.admin

import future.keywords.if

default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Admin access restricted to department records."
}

allowed_actions := {"select", "aggregate"}

# Admins can see everything in their department
decision := {
    "authorized": true,
    "columns": [], # Empty list = all columns in our sanitizer logic (or handled by DB)
    "filter": sprintf("department_id = %v", [input.user.department_id])
} if {
    input.action in allowed_actions
}
