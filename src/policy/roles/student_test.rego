package sutradhara.roles.student_test

import data.sutradhara.roles.student
import future.keywords.if

test_student_access_own_attendance if {
    res := student.decision with input as {
        "user": {"id": 500, "role": "student"},
        "target_table": "attendance",
        "action": "select"
    }
    res.authorized == true
    res.filter == "student_id = 500"
}

test_student_access_courses_no_filter if {
    res := student.decision with input as {
        "user": {"id": 500, "role": "student"},
        "target_table": "courses",
        "action": "select"
    }
    res.authorized == true
    res.filter == ""
}

test_student_deny_other_table if {
    res := student.decision with input as {
        "user": {"id": 500, "role": "student"},
        "target_table": "fee_payments",
        "action": "select"
    }
    res.authorized == false
}
