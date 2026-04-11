package sutradhara.main

import data.sutradhara.roles.student
import data.sutradhara.roles.teacher
import data.sutradhara.roles.admin
import data.sutradhara.roles.principal

# Modular decision routing
decision := student.decision if {
    input.user.role == "student"
}

decision := teacher.decision if {
    input.user.role == "teacher"
}

decision := admin.decision if {
    input.user.role == "admin"
}

decision := principal.decision if {
    input.user.role == "principal"
}

# Default deny if no role matches or error occurs
default decision = {
    "authorized": false,
    "columns": [],
    "filter": "1=0",
    "error": "Invalid or missing role"
}
