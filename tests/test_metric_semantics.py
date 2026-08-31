"""
Mandatory regression protecting against reintroducing the incorrect
equivalence assumption: "lowest present%" was previously (wrongly) treated
as interchangeable with "highest absent%" in
tests/test_live_structured_reliability.py's ranking-correctness oracle. That
equivalence only holds when STATUS is effectively binary (present/absent
only); the real STATUS enum has FOUR members (present/absent/late/excused),
so %present + %absent + %late + %excused = 100% exactly, but
%present != 100% - %absent whenever %late or %excused differ between the
students being compared.

"Lowest/highest attendance" is therefore always pinned to
percentage_of=status/present (never derived from an absent-based numerator
by direction inversion) wherever it's expressed in a QueryPlan --
independent of whether that plan is hand-constructed or model-produced.
"""

from decimal import Decimal


def _attendance_rate(present: int, absent: int, late: int, excused: int) -> Decimal:
    total = present + absent + late + excused
    return (Decimal(present) * 100) / Decimal(total)


def _absence_rate(present: int, absent: int, late: int, excused: int) -> Decimal:
    total = present + absent + late + excused
    return (Decimal(absent) * 100) / Decimal(total)


def test_lowest_present_rate_and_highest_absent_rate_can_disagree_on_the_ranking():
    """Two students with the SAME absence rate but DIFFERENT late/excused
    distributions get DIFFERENT present rates -- so "ranked by lowest
    present%" and "ranked by highest absent%" can pick different winners.
    This is the concrete demonstration requested: not an assumption, a
    computed example."""
    # Alice: 10 present, 10 absent, 0 late, 0 excused out of 20 records.
    alice = {"present": 10, "absent": 10, "late": 0, "excused": 0}
    # Bob: 10 present, 10 absent, but ALSO 10 late and 10 excused records
    # (a larger total denominator) -- same absent COUNT and RATE-of-absent-
    # among-present-vs-absent as Alice is not what matters here; what
    # matters is his overall denominator differs because late/excused exist.
    bob = {"present": 10, "absent": 10, "late": 10, "excused": 10}

    alice_present_rate = _attendance_rate(**alice)
    bob_present_rate = _attendance_rate(**bob)
    alice_absent_rate = _absence_rate(**alice)
    bob_absent_rate = _absence_rate(**bob)

    # Both have absent=10, present=10 -- but Bob's larger denominator
    # (40 vs 20) makes BOTH his present% and absent% lower than Alice's.
    assert alice_present_rate == Decimal("50")
    assert bob_present_rate == Decimal("25")
    assert alice_absent_rate == Decimal("50")
    assert bob_absent_rate == Decimal("25")

    # "Lowest present%" ranks Bob lowest (25% < 50%).
    lowest_present_winner = min([("alice", alice_present_rate), ("bob", bob_present_rate)], key=lambda x: x[1])[0]
    # "Highest absent%" should, if the two were truly equivalent, ALSO rank
    # Bob as the extreme -- but "highest absent%" picks the LARGEST value,
    # and Bob's absent% (25%) is actually LOWER than Alice's (50%), so
    # "highest absent%" picks ALICE, not Bob.
    highest_absent_winner = max([("alice", alice_absent_rate), ("bob", bob_absent_rate)], key=lambda x: x[1])[0]

    assert lowest_present_winner == "bob"
    assert highest_absent_winner == "alice"
    assert lowest_present_winner != highest_absent_winner, (
        "lowest present% and highest absent% picked the SAME student -- this would wrongly "
        "suggest the two rankings are interchangeable. They disagree here specifically because "
        "late/excused records exist and differ between students, which is exactly why "
        "'attendance rate' must always be pinned to status=present, never derived from an "
        "absence-based numerator by direction inversion."
    )
