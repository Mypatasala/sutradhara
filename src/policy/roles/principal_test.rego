package sutradhara.roles.principal_test

import data.sutradhara.roles.principal
import future.keywords.if

test_principal_global_access if {
    res := principal.decision with input as {
        "user": {"id": 1, "role": "principal"},
        "target_table": "any_table",
        "action": "select"
    }
    res.authorized == true
    res.filter == ""
}
