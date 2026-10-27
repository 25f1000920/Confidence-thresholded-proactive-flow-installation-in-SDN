import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.schema import FlowEvent
from shared.event_log import EventLogWriter, read_events_list
from shared.live_merge import merge_latency


def _ev(fid, t, sig, **kw):
    base = dict(run_id="r", strategy="markov", tau=0.5, scenario="table_pressure", seed=1,
                repetition_index=0, flow_id=fid, t_arrival=t, signature=sig)
    base.update(kw)
    return FlowEvent(**base)


def test_merge_fills_latency_and_harmful_flag():
    d = tempfile.mkdtemp()
    ev_path, recv_path = os.path.join(d, "e.jsonl"), os.path.join(d, "r.csv")
    events = [
        _ev(0, 0.0, "A", match_type="miss", controller_interaction=True),
        _ev(1, 1.0, "B", match_type="hit", decision="proactive_install", installed_proactively=True,
            caused_eviction=True, evicted_signature="A", eviction_cause="capacity_lru",
            evicted_natural_expiry=5.0),
        _ev(2, 3.0, "A", match_type="miss", controller_interaction=True),
    ]
    with EventLogWriter(ev_path) as w:
        for e in events:
            w.write(e)
    with open(recv_path, "w") as f:
        f.write("0,10.000000,10.008000\n1,10.100000,10.100400\n")  # flow 2 never arrived

    stats = merge_latency(ev_path, recv_path)
    assert stats == {"n_events": 3, "n_with_latency": 2, "n_recv_records": 2}, stats
    out = {e.flow_id: e for e in read_events_list(ev_path)}
    assert abs(out[0].flow_setup_latency_ms - 8.0) < 1e-6
    assert abs(out[1].flow_setup_latency_ms - 0.4) < 1e-6
    assert out[2].flow_setup_latency_ms is None
    assert out[1].harmful_eviction is True      # A evicted at t=1, came back as a miss at t=3 < 5
    print("test_merge_fills_latency_and_harmful_flag: PASS")


def test_rewriting_does_not_duplicate_lines():
    d = tempfile.mkdtemp()
    ev_path, recv_path = os.path.join(d, "e.jsonl"), os.path.join(d, "r.csv")
    with EventLogWriter(ev_path) as w:
        w.write(_ev(0, 0.0, "A", match_type="miss", controller_interaction=True))
    open(recv_path, "w").write("0,1.0,1.001\n")
    merge_latency(ev_path, recv_path)
    merge_latency(ev_path, recv_path)
    assert len(read_events_list(ev_path)) == 1
    print("test_rewriting_does_not_duplicate_lines: PASS")


if __name__ == "__main__":
    test_merge_fills_latency_and_harmful_flag()
    test_rewriting_does_not_duplicate_lines()
    print("\nAll live_merge tests passed.")
