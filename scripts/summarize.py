"""
Summarise every event log into one table (pure Python, no extra packages).

    python3 scripts/summarize.py                 # reads logs/, writes results/summary.csv
    python3 scripts/summarize.py --logs logs/live

One output row per (mode, scenario, strategy, tau): the mean and standard
deviation over seeds of

    hit_rate            share of flows that matched an installed rule
    proactive_installs  proactive installs per run
    precision           share of proactive installs whose predicted flow then actually arrived next
    evictions           rules evicted per run (sum of n_evictions)
    harmful_evictions   proactive evictions whose flow came back before its timeout, as a miss
    latency_ms          mean flow-setup latency (live: measured; offline: synthetic 8 ms / 0.5 ms model)
    p95_latency_ms      95th-percentile latency
    hit_latency_ms / miss_latency_ms

Also prints a short table per scenario and warns about anything suspicious
(missing latencies, unequal seed counts, logs from before the eviction fix).
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.event_log import read_events_list  # noqa: E402

METRICS = ["hit_rate", "proactive_installs", "precision", "evictions", "harmful_evictions",
           "latency_ms", "p95_latency_ms", "hit_latency_ms", "miss_latency_ms"]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def run_metrics(events, mode: str) -> dict:
    events = sorted(events, key=lambda e: e.flow_id)
    n = len(events)
    lat = [e.flow_setup_latency_ms for e in events if e.flow_setup_latency_ms is not None]
    pro = [i for i, e in enumerate(events) if e.installed_proactively]
    # Offline replay predicts flow i BEFORE it arrives, so event i's prediction targets flow i.
    # Live, the controller can only act on flow i's Packet-In, so event i's prediction targets flow i+1.
    shift = 1 if mode == "live" else 0
    correct = sum(1 for i in pro if i + shift < n and events[i].predicted_signature == events[i + shift].signature)
    old_format = any(e.caused_eviction and e.evicted_by is None for e in events)
    return {
        "n_flows": n,
        "missing_latency": n - len(lat),
        "hit_rate": sum(e.match_type == "hit" for e in events) / n,
        "proactive_installs": len(pro),
        "precision": (correct / len(pro)) if pro else None,
        # logs made before the fix have no n_evictions; fall back to (undercounted) rows
        "evictions": sum(bool(e.caused_eviction) for e in events) if old_format else sum(e.n_evictions for e in events),
        "harmful_evictions": sum(bool(e.harmful_eviction) for e in events),
        "latency_ms": _mean(lat),
        "p95_latency_ms": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)] if lat else None,
        "hit_latency_ms": _mean([e.flow_setup_latency_ms for e in events if e.match_type == "hit"]),
        "miss_latency_ms": _mean([e.flow_setup_latency_ms for e in events if e.match_type == "miss"]),
        "old_format": old_format,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs", help="folder searched recursively for *.jsonl event logs")
    p.add_argument("--out", default="results/summary.csv")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs, "**", "*.jsonl"), recursive=True))
    if not files:
        sys.exit(f"No *.jsonl logs found under {args.logs}/ -- run this from the project folder.")

    groups = defaultdict(list)
    warnings = []
    n_old = 0
    for f in files:
        events = read_events_list(f)
        if not events:
            warnings.append(f"empty log: {f}")
            continue
        e0 = events[0]
        mode = "live" if os.sep + "live" in os.path.abspath(f) else "offline"
        m = run_metrics(events, mode)
        if mode == "live" and m["missing_latency"]:
            warnings.append(f"{m['missing_latency']} flows without latency in {f}")
        if m["old_format"]:
            n_old += 1
        groups[(mode, e0.scenario, e0.strategy, e0.tau)].append(m)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    for (mode, scenario, strategy, tau), runs in sorted(groups.items()):
        row = {"mode": mode, "scenario": scenario, "strategy": strategy, "tau": tau, "n_seeds": len(runs)}
        for k in METRICS:
            vals = [r[k] for r in runs if r[k] is not None]
            row[k + "_mean"] = round(st.mean(vals), 4) if vals else ""
            row[k + "_sd"] = round(st.stdev(vals), 4) if len(vals) > 1 else ""
        rows.append(row)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if n_old:
        warnings.append(f"{n_old} logs were made before the eviction-count fix, so their "
                        "eviction counts are too low (rerun with the current code)")
    seeds = {r["n_seeds"] for r in rows}
    if len(seeds) > 1:
        warnings.append(f"groups have different numbers of seeds {sorted(seeds)} -- some runs are missing")

    def fmt(v, pct=False):
        if v == "":
            return "-"
        return f"{100 * v:5.1f}%" if pct else f"{v:7.2f}"

    for scenario in sorted({r["scenario"] for r in rows}):
        for mode in ("offline", "live"):
            sel = [r for r in rows if r["scenario"] == scenario and r["mode"] == mode]
            if not sel:
                continue
            print(f"\n{scenario} ({mode}, {sel[0]['n_seeds']} seeds)")
            print(f"  {'strategy':9}{'tau':>5}{'hit':>8}{'installs':>10}{'precision':>11}"
                  f"{'evictions':>11}{'harmful':>9}{'latency_ms':>12}")
            for r in sel:
                print(f"  {r['strategy']:9}{r['tau']:5.1f}{fmt(r['hit_rate_mean'], True):>8}"
                      f"{fmt(r['proactive_installs_mean']):>10}{fmt(r['precision_mean'], True):>11}"
                      f"{fmt(r['evictions_mean']):>11}{fmt(r['harmful_evictions_mean']):>9}"
                      f"{fmt(r['latency_ms_mean']):>12}")

    print(f"\nRead {len(files)} logs -> {len(rows)} rows written to {args.out}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w_ in warnings[:15]:
            print("  - " + w_)
        if len(warnings) > 15:
            print(f"  ... and {len(warnings) - 15} more")


if __name__ == "__main__":
    main()
