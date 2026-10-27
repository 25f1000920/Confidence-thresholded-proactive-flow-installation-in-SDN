"""
Post-processing for live runs. The controller log has everything EXCEPT
latency (the controller can't see when a packet reached its destination), and
it doesn't run the harmful-eviction analysis. This module:

  1. reads the receiving hosts' CSV ("flow_id,t_send,t_recv" per packet),
  2. sets flow_setup_latency_ms = (t_recv - t_send) * 1000 on each controller
     event with the same flow_id (None if that packet never arrived), and
  3. runs table_sim.flow_table.analyze_harmful_evictions on the full run,

then rewrites the event log in place. After this, a live log has exactly the
same fields as an offline-replay log.
"""
from __future__ import annotations
from typing import Dict, Tuple

from shared.event_log import read_events_list, EventLogWriter
from table_sim.flow_table import analyze_harmful_evictions


def load_recv(recv_path: str) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    with open(recv_path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            try:
                out[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except ValueError:
                continue
    return out


def merge_latency(events_path: str, recv_path: str) -> dict:
    events = read_events_list(events_path)
    recv = load_recv(recv_path)

    matched = 0
    for ev in events:
        r = recv.get(ev.flow_id)
        if r is not None:
            ev.flow_setup_latency_ms = (r[1] - r[0]) * 1000.0
            matched += 1
        else:
            ev.flow_setup_latency_ms = None

    analyze_harmful_evictions(events)

    with EventLogWriter(events_path) as w:  # truncates, then rewrites
        for ev in sorted(events, key=lambda e: e.flow_id):
            w.write(ev)

    return {"n_events": len(events), "n_with_latency": matched, "n_recv_records": len(recv)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python3 -m shared.live_merge <events.jsonl> <recv.csv>")
        sys.exit(1)
    print(merge_latency(sys.argv[1], sys.argv[2]))
