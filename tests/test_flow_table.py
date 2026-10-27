"""
Section 2.3 Step 2: "A tiny deterministic sequence ... is worked through by
hand for prediction, installation, hit/miss, eviction, TTL expiry, and
Packet-In outcomes at each step; the implementation is trusted on generated
traffic only once its event log matches this trace exactly."

Sequence: A -> B -> C -> A -> B -> C, table capacity = 2, idle_timeout = 100
(large, so TTL never fires here -- capacity eviction is isolated on purpose;
a second test below isolates TTL expiry with capacity effectively unlimited).

Hand-worked trace (capacity=2, LRU eviction), all installs REACTIVE
(install-on-miss, capacity applies to every install regardless of strategy):

  t=0  A arrives -> miss (table {})            -> install A          -> table {A}
  t=1  B arrives -> miss (table {A})           -> install B          -> table {A,B}
  t=2  C arrives -> miss (table {A,B})         -> at capacity, evict LRU=A -> install C -> table {B,C}
  t=3  A arrives -> miss (table {B,C})         -> at capacity, evict LRU=B -> install A -> table {C,A}
  t=4  B arrives -> miss (table {C,A})         -> at capacity, evict LRU=C -> install B -> table {A,B}
  t=5  C arrives -> miss (table {A,B})         -> at capacity, evict LRU=A -> install C -> table {B,C}

Every arrival is a miss because capacity=2 can never hold all 3 signatures
that keep cycling -- this is the intended "thrashing" edge case the hand
trace is designed to exercise.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from table_sim.flow_table import FlowTable


def test_hand_traced_capacity_eviction():
    table = FlowTable(capacity=2, idle_timeout=100.0)
    sequence = ["A", "B", "C", "A", "B", "C"]

    expected_hits = [False, False, False, False, False, False]
    expected_evicted = [None, None, "A", "B", "C", "A"]

    got_hits, got_evicted = [], []
    for t, sig in enumerate(sequence):
        m = table.match(sig, t)
        got_hits.append(m.hit)
        if not m.hit:
            res = table.install(sig, t)
            got_evicted.append(res.evicted_signature)
        else:
            got_evicted.append(None)

    assert got_hits == expected_hits, f"hits mismatch: got {got_hits}, expected {expected_hits}"
    assert got_evicted == expected_evicted, f"evictions mismatch: got {got_evicted}, expected {expected_evicted}"
    assert len(table) == 2
    print("test_hand_traced_capacity_eviction: PASS")


def test_hand_traced_ttl_expiry():
    """
    Sequence: A@t=0, A@t=3 (hit, refreshes TTL), then nothing touches A
    until t=20. idle_timeout=5 -> A must have expired by t=20 (last_used=3,
    expiry=8). Table capacity set high so capacity eviction never interferes.
    """
    table = FlowTable(capacity=10, idle_timeout=5.0)

    m0 = table.match("A", 0.0)
    assert m0.hit is False
    table.install("A", 0.0)

    m1 = table.match("A", 3.0)
    assert m1.hit is True, "A should still be resident and hit at t=3 (installed at t=0, timeout=5)"

    expired_before = "A" in table
    assert expired_before is True

    table.expire(t=20.0)  # last_used=3, natural_expiry=8, so by t=20 it must be gone
    assert "A" not in table, "A should have expired by t=20 (last_used=3 + idle_timeout=5 = 8)"

    m2 = table.match("A", 20.0)
    assert m2.hit is False, "A was TTL-expired, so this must be a miss, not a hit"
    print("test_hand_traced_ttl_expiry: PASS")


def test_harmful_eviction_analysis_confirms_harm():
    """
    Minimal scripted scenario for the harmful-eviction definition itself:
    a proactive install of B evicts A (capacity_lru) while A would still
    have been "alive"; A then reappears (as a real flow arrival) before its
    natural expiry and misses + triggers a controller interaction ->
    harmful=True by definition (i, ii, iii all satisfied).
    """
    from shared.schema import FlowEvent
    from table_sim.flow_table import analyze_harmful_evictions

    events = [
        # A is installed reactively at t=0, idle_timeout=5 -> natural_expiry=5
        FlowEvent(run_id="t", strategy="markov", tau=0.5, scenario="baseline", seed=1,
                   repetition_index=0, flow_id=0, t_arrival=0.0, signature="A",
                   decision="reactive_install", match_type="miss", controller_interaction=True),
        # B is proactively installed at t=1, evicting A. A's natural_expiry
        # at the moment of eviction was 5 (0 + idle_timeout=5).
        FlowEvent(run_id="t", strategy="markov", tau=0.5, scenario="baseline", seed=1,
                   repetition_index=0, flow_id=1, t_arrival=1.0, signature="B",
                   decision="proactive_install", installed_proactively=True,
                   match_type="hit", caused_eviction=True, evicted_signature="A",
                   eviction_cause="capacity_lru", evicted_natural_expiry=5.0),
        # A reappears at t=3, BEFORE its would-have-been natural expiry (5) -> miss, controller interaction
        FlowEvent(run_id="t", strategy="markov", tau=0.5, scenario="baseline", seed=1,
                   repetition_index=0, flow_id=2, t_arrival=3.0, signature="A",
                   decision="reactive_install", match_type="miss", controller_interaction=True),
    ]
    analyze_harmful_evictions(events)
    assert events[1].harmful_eviction is True
    print("test_harmful_eviction_analysis_confirms_harm: PASS")


def test_harmful_eviction_analysis_rejects_late_return():
    """Same shape, but A returns AFTER its natural expiry -> not harmful (it would have expired anyway)."""
    from shared.schema import FlowEvent
    from table_sim.flow_table import analyze_harmful_evictions

    events = [
        FlowEvent(run_id="t", strategy="markov", tau=0.5, scenario="baseline", seed=1,
                   repetition_index=0, flow_id=0, t_arrival=0.0, signature="A",
                   decision="reactive_install", match_type="miss", controller_interaction=True),
        FlowEvent(run_id="t", strategy="markov", tau=0.5, scenario="baseline", seed=1,
                   repetition_index=0, flow_id=1, t_arrival=1.0, signature="B",
                   decision="proactive_install", installed_proactively=True,
                   match_type="hit", caused_eviction=True, evicted_signature="A",
                   eviction_cause="capacity_lru", evicted_natural_expiry=5.0),
        # A reappears at t=9, AFTER natural_expiry=5 -> not harmful
        FlowEvent(run_id="t", strategy="markov", tau=0.5, scenario="baseline", seed=1,
                   repetition_index=0, flow_id=2, t_arrival=9.0, signature="A",
                   decision="reactive_install", match_type="miss", controller_interaction=True),
    ]
    analyze_harmful_evictions(events)
    assert events[1].harmful_eviction is False
    print("test_harmful_eviction_analysis_rejects_late_return: PASS")


if __name__ == "__main__":
    test_hand_traced_capacity_eviction()
    test_hand_traced_ttl_expiry()
    test_harmful_eviction_analysis_confirms_harm()
    test_harmful_eviction_analysis_rejects_late_return()
    print("\nAll hand-traced ground-truth tests passed.")
