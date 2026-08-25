package sutradhara.roles.teacher_test

import data.sutradhara.roles.teacher
import future.keywords.if

test_teacher_access_users_masked if {
    res := teacher.decision with input as {
        "user": {"id": 1001, "role": "teacher"},
        "target_table": "users",
        "action": "select"
    }
    res.authorized == true
    res.columns == ["id", "name", "role", "department_id"]
}

test_teacher_access_attendance_rebac if {
    res := teacher.decision with input as {
        "user": {"id": 1001, "role": "teacher"},
        "target_table": "attendance",
        "action": "select"
    }
    res.authorized == true
    contains(res.filter, "teacher_id = 1001")
}

test_teacher_access_own_courses if {
    res := teacher.decision with input as {
        "user": {"id": 1001, "role": "teacher"},
        "target_table": "courses",
        "action": "select"
    }
    res.authorized == true
    res.filter == "teacher_id = 1001"
}

test_teacher_deny_fees if {
    res := teacher.decision with input as {
        "user": {"id": 1001, "role": "teacher"},
        "target_table": "fee_payments",
        "action": "select"
    }
    res.authorized == false
}
