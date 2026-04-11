package sutradhara.roles.principal

import future.keywords.if

default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Principal access denied (read-only allowed)."
}

allowed_actions := {"select", "aggregate"}

# Principals have global read access
decision := {
    "authorized": true,
    "columns": [], # Full column access
    "filter": ""   # No row filter
} if {
    input.action in allowed_actions
}
