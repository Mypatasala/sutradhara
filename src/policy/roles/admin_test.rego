package sutradhara.roles.admin_test

import data.sutradhara.roles.admin
import future.keywords.if

test_admin_access_department if {
    res := admin.decision with input as {
        "user": {"id": 10, "role": "admin", "department_id": 1},
        "target_table": "students",
        "action": "select"
    }
    res.authorized == true
    res.filter == "department_id = 1"
}

test_admin_deny_no_action if {
    res := admin.decision with input as {
        "user": {"id": 10, "role": "admin", "department_id": 1},
        "target_table": "students",
        "action": "delete"
    }
    res.authorized == false
}
