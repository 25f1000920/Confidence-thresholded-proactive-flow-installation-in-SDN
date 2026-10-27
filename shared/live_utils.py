"""
Helpers shared by the live-run pieces (host agent, Ryu controller, merge
script). Kept dependency-free so they can be unit-tested without Ryu/Mininet.

Naming: the traffic generators and predictors identify hosts as "h1".."h16",
while packets on the wire carry IPs "10.0.0.1".."10.0.0.16" (see
topo/mininet_topo.py). Every signature the controller logs or predicts MUST
use the "hN" form, or the predictors (trained on generator output) would
never recognise a live flow.

Payload: each UDP packet carries "<flow_id>,<t_send>" so the receiving host
can compute true end-to-end latency and the controller can log the same
flow_id, letting the two be joined exactly afterwards.
"""
from __future__ import annotations
import re
from typing import Optional, Tuple

_PAYLOAD_RE = re.compile(rb"^(\d+),(\d+(?:\.\d+)?)")


def ip_to_host(ip: str) -> str:
    return "h" + ip.rsplit(".", 1)[-1]


def host_to_ip(host: str) -> str:
    return "10.0.0." + host.lstrip("h")


def host_to_port(host: str) -> int:
    """Switch port of host hN in the flat topology (links are added h1..hN in order)."""
    return int(host.lstrip("h"))


def build_payload(flow_id: int, t_send: float) -> bytes:
    return f"{flow_id},{t_send:.6f}".encode()


def parse_payload(data: bytes) -> Optional[Tuple[int, float]]:
    """Returns (flow_id, t_send), or None if malformed. Tolerates trailing padding bytes."""
    m = _PAYLOAD_RE.match(bytes(data))
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))
