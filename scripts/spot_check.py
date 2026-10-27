"""
Spot checks (Research Plan Section 2.4, Quality assurance: "a manual spot-check
tracing a small number of individual flows per scenario through the logs").

Two parts:

1. AUTOMATIC CHECKS over every log:
   - offline: re-simulates the flow table from each log's own decisions and
     confirms every flow's hit/miss and eviction count is exactly what the
     table rules imply (catches any bookkeeping bug in replay);
   - live: flow ids complete, every flow has a latency, occupancy never above
     capacity, harmful evictions only on proactive evictions, and hits faster
     than misses.

2. A READABLE TRACE of a few flows per scenario (one run each), written to
   results/spot_checks.txt, for you to follow by hand: what was predicted, with
   what confidence, whether it was installed, whether the flow then hit, and
   which rule got evicted.

    python3 scripts/spot_check.py
    python3 scripts/spot_check.py --flows 20 --seed 1003
"""
from __future__ import annotations
import argparse
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import config as cfg  # noqa: E402
from shared.event_log import read_events_list  # noqa: E402
from table_sim.flow_table import FlowTable  # noqa: E402


def check_offline(events, capacity: int) -> list[str]:
    """Rebuild the table from the log's own decisions (replay order: proactive install,
    then the flow arrives, then a reactive install on a miss) and compare."""
    problems = []
    t = FlowTable(capacity=capacity, idle_timeout=cfg.IDLE_TIMEOUT_SEC)
    for e in events:
        n_ev = 0
        if e.installed_proactively:
            n_ev += t.install(e.predicted_signature, e.t_arrival).caused_eviction
        m = t.match(e.signature, e.t_arrival)
        if m.hit != (e.match_type == "hit"):
            problems.append(f"flow {e.flow_id}: log says {e.match_type}, table rules say "
                            f"{'hit' if m.hit else 'miss'}")
        if not m.hit:
            n_ev += t.install(e.signature, e.t_arrival).caused_eviction
        if n_ev != e.n_evictions:
            problems.append(f"flow {e.flow_id}: log says {e.n_evictions} evictions, table rules say {n_ev}")
        if len(problems) >= 3:
            break
    return problems


def check_live(events, capacity: int) -> list[str]:
    problems = []
    ids = sorted(e.flow_id for e in events)
    if ids != list(range(len(ids))):
        problems.append("flow ids are not exactly 0..n-1 (missing or duplicated flows)")
    if any(e.flow_setup_latency_ms is None for e in events):
        problems.append("some flows have no latency")
    if any((e.table_occupancy_after or 0) > capacity for e in events):
        problems.append(f"table occupancy above capacity {capacity}")
    if any(e.harmful_eviction and e.evicted_by != "proactive" for e in events):
        problems.append("a harmful eviction that was not caused by a proactive install")
    hits = [e.flow_setup_latency_ms for e in events if e.match_type == "hit" and e.flow_setup_latency_ms is not None]
    miss = [e.flow_setup_latency_ms for e in events if e.match_type == "miss" and e.flow_setup_latency_ms is not None]
    if hits and miss and st.median(hits) >= st.median(miss):
        problems.append("hits are not faster than misses (median latency)")
    return problems


def trace(events, n_flows: int, live: bool) -> str:
    """A window of flows around the first eviction (or the start), as plain text."""
    events = sorted(events, key=lambda e: e.flow_id)
    first = next((i for i, e in enumerate(events) if e.caused_eviction), 0)
    start = max(0, first - n_flows // 2)
    e0 = events[0]
    lines = [f"--- {e0.scenario} | {e0.strategy} tau={e0.tau:.1f} | seed {e0.seed} | flows {start}-{start + n_flows - 1}",
             ("    (live: the prediction on a row is for the NEXT flow)" if live
              else "    (offline: the prediction on a row was made just BEFORE this flow arrived)"),
             f"    {'flow':>5} {'t (s)':>7}  {'arriving flow':17} {'predicted':17} {'conf':>5}  "
             f"{'installed?':10} {'result':6} {'eviction':28} {'harmful':7}"]
    for e in events[start:start + n_flows]:
        conf = f"{e.confidence:.2f}" if e.confidence is not None else "-"
        ev = ""
        if e.n_evictions:
            ev = f"{e.n_evictions}x, {e.evicted_by}: {e.evicted_signature}"
        harm = {True: "YES", False: "no", None: ""}[e.harmful_eviction]
        lines.append(f"    {e.flow_id:>5} {e.t_arrival:7.3f}  {e.signature:17} {str(e.predicted_signature or '-'):17} "
                     f"{conf:>5}  {'proactive' if e.installed_proactively else '-':10} {e.match_type:6} {ev:28} {harm:7}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs")
    p.add_argument("--flows", type=int, default=12, help="flows per trace")
    p.add_argument("--seed", type=int, default=cfg.EVALUATION_SEEDS[0])
    p.add_argument("--out", default="results/spot_checks.txt")
    args = p.parse_args()

    off = sorted(glob.glob(os.path.join(args.logs, "offline", "*", "*.jsonl")))
    liv = sorted(glob.glob(os.path.join(args.logs, "live", "*.jsonl")))
    if not off and not liv:
        sys.exit(f"No logs found under {args.logs}/ -- run this from the project folder.")

    print("Automatic checks")
    all_ok = True
    for label, files, fn, cap in (("offline", off, check_offline, cfg.TABLE_CAPACITY),
                                  ("live", liv, check_live, cfg.TABLE_PRESSURE_CAPACITY)):
        if not files:
            continue
        bad = []
        for f in files:
            probs = fn(sorted(read_events_list(f), key=lambda e: e.flow_id), cap)
            if probs:
                bad.append((f, probs))
        all_ok &= not bad
        print(f"  {label:8} {len(files):4} logs checked: "
              f"{'all consistent' if not bad else f'{len(bad)} with problems'}")
        for f, probs in bad[:5]:
            print(f"     {os.path.basename(f)}: {probs[0]}")

    traces = []
    for scen_dir in sorted(glob.glob(os.path.join(args.logs, "offline", "*"))):
        f = os.path.join(scen_dir, f"markov__tau0.3__seed{args.seed}.jsonl")
        if os.path.exists(f):
            traces.append(trace(read_events_list(f), args.flows, live=False))
    f = os.path.join(args.logs, "live", f"markov__tau0.3__table_pressure__seed{args.seed}.jsonl")
    if os.path.exists(f):
        traces.append(trace(read_events_list(f), args.flows, live=True))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("Spot-check traces: Markov, tau = 0.3, one run per scenario.\n"
                 "Check by hand for a few rows: a flow is a hit only if its rule is in the table "
                 "(installed earlier and not evicted or idle for 5 s); a proactive install into a "
                 "full table evicts the least-recently-used rule.\n\n")
        fh.write("\n\n".join(traces) + "\n")
    print(f"\n{len(traces)} traces written to {args.out}")
    print(f"\nOverall: {'PASS' if all_ok else 'PROBLEMS FOUND -- see above'}")


if __name__ == "__main__":
    main()
