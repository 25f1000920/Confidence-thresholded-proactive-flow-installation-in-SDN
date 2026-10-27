"""
Real-time traffic player for live runs.

Deliberately split in two:
  - `play_traffic(...)`: pure scheduling/timing logic, parameterized over a
    `send_fn` callback, a `sleep_fn`, and a `clock_fn`. Depends on nothing
    Mininet-specific, so it is fully unit-testable in this sandbox with a
    fake clock and a recording send_fn -- see tests/test_live_traffic_player.py.
  - `LiveSender`: the actual Mininet-facing implementation of `send_fn`,
    talking to each host's `host_agent.py` over a UNIX datagram socket.
    This half is NOT testable here (needs real per-namespace sockets) but
    is intentionally tiny and mechanical, to keep untested surface area small.
"""
from __future__ import annotations
import os
import socket
import sys
import time
from typing import Callable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic.generators import GENERATORS
from shared import config as cfg
from shared.schema import FlowSignature

SendFn = Callable[[str, str, int], None]  # (src_host_name, dst_ip, dst_port) -> None


def host_ip(host_name: str) -> str:
    """Matches topo/mininet_topo.py's '10.0.0.<i>' convention and
    controller/ryu_app.py's _lookup_out_port assumption (host index == port)."""
    idx = int(host_name.lstrip("h"))
    return f"10.0.0.{idx}"


def play_traffic(
    scenario: str,
    seed: int,
    n_flows: int,
    send_fn: SendFn,
    rate: float = cfg.FLOW_RATE_FLOWS_PER_SEC,
    speed_factor: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> List[Tuple[FlowSignature, float, float]]:
    """
    Generates `n_flows` of `scenario` traffic (seeded by `seed`, same
    generator every offline replay of this scenario would use) and
    dispatches each flow to `send_fn` at its scheduled arrival time.

    Returns [(signature, scheduled_t, actual_t_since_start), ...] so timing
    accuracy (scheduled vs. actual) can be checked after the run -- large,
    systematic drift here would mean the live run's inter-arrival times no
    longer match what the offline-replayed scenarios assumed, which would
    quietly invalidate any baseline/cold_start/drift/unstructured vs.
    table_pressure comparison.

    `speed_factor` > 1.0 compresses elapsed time (useful for a fast pilot);
    keep it at 1.0 for the real experiment matrix.
    """
    if scenario not in GENERATORS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {list(GENERATORS)}")

    signatures, timestamps = GENERATORS[scenario](seed, n_flows, rate=rate)

    t_start = clock_fn()
    log: List[Tuple[FlowSignature, float, float]] = []
    for sig, t_sched in zip(signatures, timestamps):
        target = t_start + t_sched / speed_factor
        now = clock_fn()
        if target > now:
            sleep_fn(target - now)
        send_fn(sig.src_host, host_ip(sig.dst_host), sig.dst_port)
        log.append((sig, t_sched, clock_fn() - t_start))
    return log


def check_timing_accuracy(log: List[Tuple[FlowSignature, float, float]],
                           max_mean_drift_ms: float = 5.0) -> None:
    """
    Raises if actual dispatch times drifted too far from scheduled times on
    average -- run this after every live scenario before trusting its data.
    A little drift is expected (sleep() granularity, GIL, agent dispatch);
    large systematic drift usually means the host is overloaded at the
    configured flow rate and FLOW_RATE_FLOWS_PER_SEC needs recalibrating
    down (see scripts/calibrate.py).
    """
    if not log:
        return
    drifts = [abs(actual - sched) for _, sched, actual in log]
    mean_drift_ms = 1000.0 * sum(drifts) / len(drifts)
    if mean_drift_ms > max_mean_drift_ms:
        raise RuntimeError(
            f"Mean scheduling drift {mean_drift_ms:.2f}ms exceeds {max_mean_drift_ms}ms "
            f"-- traffic rate is likely too high for this host; recalibrate FLOW_RATE_FLOWS_PER_SEC."
        )


class LiveSender:
    """
    Talks to each host's host_agent.py (scripts/host_agent.py) over a UNIX
    datagram control socket at `<sock_dir>/agent_<host_name>.sock`.

    Usage on your Mininet machine (see scripts/run_live_experiment.py for
    the full driver that also starts the agents and the Mininet network):

        sender = LiveSender()
        play_traffic(scenario, seed, n_flows, sender.send)
    """

    def __init__(self, sock_dir: str = "/tmp"):
        self.sock_dir = sock_dir
        self._ctrl = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._next_flow_id = 0  # increments per send; matches generator order

    def _agent_path(self, host_name: str) -> str:
        return os.path.join(self.sock_dir, f"agent_{host_name}.sock")

    def send(self, src_host: str, dst_ip: str, dst_port: int) -> None:
        flow_id = self._next_flow_id
        self._next_flow_id += 1
        msg = f"{dst_ip} {dst_port} {flow_id}\n".encode()
        self._ctrl.sendto(msg, self._agent_path(src_host))
