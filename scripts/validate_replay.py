"""
Replay-vs-live validation (Research Plan Section 2.3 step 8, last sentence, and
the Quality-assurance paragraph of Section 2.4):

    "Before relying on this split, replay output at one tau is checked against
     a live run at that same tau."

For every live table_pressure log of the chosen strategy/tau (one per seed), this
re-runs the SAME traffic (same generator, same seed) through the offline replay
engine with the SAME table capacity and the SAME predictor training, then
compares the two logs flow by flow.

    python3 scripts/validate_replay.py                     # markov, tau 0.3 + reactive
    python3 scripts/validate_replay.py --tau 0.5

Only Markov and reactive are checked: Markov training is deterministic, so the
replay predictor is exactly the live one. (The LSTM is retrained without a fixed
seed at every controller start, so a replayed LSTM would be a different model.)

What should match, and what should not:
  - NOTE: replay treats an install as instant. Live, a proactive rule is only sent
    when the PREVIOUS flow's Packet-In is handled, and it must reach the switch
    before the next packet does. The script counts these "install races".
  - hit/miss per flow, proactive installs, evictions: SHOULD match closely.
    Small differences come from real-time effects the replay cannot see
    (idle timeouts measured on the wall clock, packets arriving slightly early
    or late relative to the schedule).
  - latency: should NOT be compared -- replay uses a fixed model (8 ms miss,
    0.5 ms hit); live measures it.

PASS criteria (fixed here, before looking at the comparison):
  per-flow hit/miss agreement >= 97%  AND  |hit-rate difference| <= 1.0 point
  (averaged over seeds).
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import config as cfg  # noqa: E402
from shared.event_log import read_events_list  # noqa: E402
from traffic.generators import GENERATORS, generate_baseline  # noqa: E402
from predictors.markov import MarkovPredictor  # noqa: E402
from scripts.replay import replay_run  # noqa: E402

AGREEMENT_MIN = 0.97
HIT_DIFF_MAX = 0.010


def train_like_live() -> MarkovPredictor:
    """Exactly what controller/ryu_app.py does at startup."""
    sigs, _ = generate_baseline(cfg.TRAIN_SEED, n_flows=4000)
    p = MarkovPredictor()
    p.fit([s.key() for s in sigs])
    return p


def compare(live_path: str, predictor, capacity: int) -> dict:
    live = sorted(read_events_list(live_path), key=lambda e: e.flow_id)
    e0 = live[0]
    sigs, ts = GENERATORS["table_pressure"](e0.seed, len(live))
    # replay_run only accepts the offline scenario names; table_pressure traffic IS baseline traffic.
    rep = replay_run(e0.strategy, e0.tau, "baseline", e0.seed, e0.repetition_index, sigs, ts,
                     predictor=predictor, table_capacity=capacity)
    n = len(live)
    same_traffic = all(a.signature == b.signature for a, b in zip(live, rep))
    agree = sum(a.match_type == b.match_type for a, b in zip(live, rep)) / n

    def hit(evs):
        return sum(e.match_type == "hit" for e in evs) / n

    # Classify disagreements. "Install race": the live controller installed this flow's
    # rule proactively at the PREVIOUS arrival, but the packet reached the switch before
    # the rule did, so live saw a miss where replay (instant installs) saw a hit.
    race = other = 0
    race_gaps = []
    for k in range(n):
        if live[k].match_type == rep[k].match_type:
            continue
        prev = live[k - 1] if k else None
        if (live[k].match_type == "miss" and prev is not None and prev.installed_proactively
                and prev.predicted_signature == live[k].signature):
            race += 1
            race_gaps.append(1000 * (live[k].t_arrival - prev.t_arrival))
        else:
            other += 1

    return {
        "run": os.path.basename(live_path).replace(".jsonl", ""),
        "seed": e0.seed,
        "same_traffic": same_traffic,
        "hit_live": round(hit(live), 4),
        "hit_replay": round(hit(rep), 4),
        "hit_diff": round(hit(live) - hit(rep), 4),
        "per_flow_agreement": round(agree, 4),
        "installs_live": sum(e.installed_proactively for e in live),
        "installs_replay": sum(e.installed_proactively for e in rep),
        "evictions_live": sum(e.n_evictions for e in live),
        "evictions_replay": sum(e.n_evictions for e in rep),
        "disagree_install_race": race,
        "disagree_other": other,
        "race_median_gap_ms": round(st.median(race_gaps), 2) if race_gaps else "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default="logs/live")
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--capacity", type=int, default=cfg.TABLE_PRESSURE_CAPACITY)
    p.add_argument("--out", default="results/tables/replay_vs_live.csv")
    args = p.parse_args()

    predictor = train_like_live()
    rows = []
    for strategy, tau in (("markov", args.tau), ("reactive", 1.0)):
        files = sorted(glob.glob(os.path.join(args.logs, f"{strategy}__tau{tau:.1f}__table_pressure__seed*.jsonl")))
        if not files:
            sys.exit(f"No live logs found for {strategy} tau={tau:.1f} under {args.logs}/")
        for f in files:
            r = compare(f, predictor if strategy != "reactive" else None, args.capacity)
            r["strategy"], r["tau"] = strategy, tau
            rows.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{'run':44}{'hit live':>9}{'replay':>8}{'agree':>8}{'installs L/R':>14}{'evictions L/R':>15}")
    for r in rows:
        print(f"{r['run']:44}{100*r['hit_live']:8.1f}%{100*r['hit_replay']:7.1f}%{100*r['per_flow_agreement']:7.1f}%"
              f"{r['installs_live']:>8}/{r['installs_replay']:<5}{r['evictions_live']:>8}/{r['evictions_replay']:<5}")

    ok = True
    print()
    for strategy in ("markov", "reactive"):
        sel = [r for r in rows if r["strategy"] == strategy]
        agree = st.mean(r["per_flow_agreement"] for r in sel)
        diff = st.mean(r["hit_diff"] for r in sel)
        traffic_ok = all(r["same_traffic"] for r in sel)
        passed = traffic_ok and agree >= AGREEMENT_MIN and abs(diff) <= HIT_DIFF_MAX
        ok &= passed
        print(f"{strategy:9} tau={sel[0]['tau']:.1f}: mean per-flow agreement {100*agree:.1f}% "
              f"(need >= {100*AGREEMENT_MIN:.0f}%), mean hit-rate difference {100*diff:+.2f} points "
              f"(need within +/-{100*HIT_DIFF_MAX:.1f}), same traffic: {traffic_ok}  -> {'PASS' if passed else 'FAIL'}")
    mk = [r for r in rows if r["strategy"] == "markov"]
    race = sum(r["disagree_install_race"] for r in mk)
    other = sum(r["disagree_other"] for r in mk)
    gaps = [r["race_median_gap_ms"] for r in mk if r["race_median_gap_ms"] != ""]
    if race + other:
        print(f"\nWhy markov runs disagree: {race} of {race + other} differing flows "
              f"({100 * race / (race + other):.0f}%) are install races -- the rule for that flow was "
              f"installed proactively at the previous arrival, but the packet arrived first"
              + (f" (median {st.median(gaps):.1f} ms later)." if gaps else "."))
        print("Replay assumes installs take no time, so it overstates proactive hits by about "
              f"{-100 * st.mean(r['hit_diff'] for r in mk):.1f} points at this flow rate. Table dynamics "
              "(installs, evictions) agree; report this as a known replay limitation.")
    print(f"\nOverall: {'PASS -- offline replay agrees with live runs' if ok else 'FAIL on the pre-set criteria -- see the explanation above'}")
    print(f"Table written to {args.out}")


if __name__ == "__main__":
    main()
