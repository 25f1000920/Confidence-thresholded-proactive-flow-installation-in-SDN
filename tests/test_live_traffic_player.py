"""
Tests the scheduling/timing half of scripts/live_traffic_player.py with a
FAKE clock and a recording send_fn -- no real sockets, no Mininet, no wall-
clock sleeping. This is exactly why play_traffic() was split from
LiveSender in the first place.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.live_traffic_player import play_traffic, check_timing_accuracy, host_ip


class FakeClock:
    """A controllable clock: sleep_fn advances it exactly by the requested amount."""
    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        assert dt >= 0
        self.t += dt


def test_host_ip_matches_topology_convention():
    assert host_ip("h1") == "10.0.0.1"
    assert host_ip("h16") == "10.0.0.16"
    print("test_host_ip_matches_topology_convention: PASS")


def test_play_traffic_dispatches_in_scheduled_order_with_fake_clock():
    clock = FakeClock()
    sent = []

    def fake_send(src_host, dst_ip, dst_port):
        sent.append((src_host, dst_ip, dst_port, clock.now()))

    log = play_traffic(
        scenario="baseline", seed=1000, n_flows=200, send_fn=fake_send,
        sleep_fn=clock.sleep, clock_fn=clock.now,
    )

    assert len(sent) == 200
    assert len(log) == 200

    # With a fake clock that advances by EXACTLY the requested sleep, actual
    # dispatch time must equal scheduled time -- this isolates "is the
    # scheduling math right" from "is the real OS scheduler noisy".
    for sig, sched_t, actual_t in log:
        assert abs(sched_t - actual_t) < 1e-9, f"scheduling drift with a fake clock: {sched_t} vs {actual_t}"

    # dispatch order must be non-decreasing in scheduled time (generators
    # already guarantee this, but the player must not reorder them)
    times = [t for _, t, _ in log]
    assert times == sorted(times)

    # every dispatched dst_ip must be well-formed per the topology convention
    for src_host, dst_ip, dst_port, _ in sent:
        assert dst_ip.startswith("10.0.0.")
        assert 0 < dst_port < 65536
        assert src_host.startswith("h")

    print("test_play_traffic_dispatches_in_scheduled_order_with_fake_clock: PASS")


def test_check_timing_accuracy_passes_on_perfect_log():
    log = [(None, 1.0, 1.0), (None, 2.0, 2.001)]
    check_timing_accuracy(log, max_mean_drift_ms=5.0)  # should not raise
    print("test_check_timing_accuracy_passes_on_perfect_log: PASS")


def test_check_timing_accuracy_raises_on_bad_drift():
    log = [(None, 1.0, 1.5), (None, 2.0, 2.6)]  # 500-600ms drift
    try:
        check_timing_accuracy(log, max_mean_drift_ms=5.0)
        raise AssertionError("expected RuntimeError for excessive drift")
    except RuntimeError:
        pass
    print("test_check_timing_accuracy_raises_on_bad_drift: PASS")


def test_unknown_scenario_rejected():
    try:
        play_traffic("not_a_real_scenario", seed=1, n_flows=10, send_fn=lambda *a: None)
        raise AssertionError("expected ValueError for unknown scenario")
    except ValueError:
        pass
    print("test_unknown_scenario_rejected: PASS")


if __name__ == "__main__":
    test_host_ip_matches_topology_convention()
    test_play_traffic_dispatches_in_scheduled_order_with_fake_clock()
    test_check_timing_accuracy_passes_on_perfect_log()
    test_check_timing_accuracy_raises_on_bad_drift()
    test_unknown_scenario_rejected()
    print("\nAll live-traffic-player scheduling tests passed.")
