"""
Calibration (Research Plan Section 2.3 Step 1 / Section 2.4's precision
discussion). Two modes:

  --mode replay-dry-run   Runs entirely offline, in this sandbox or
                           anywhere with just Python -- no Mininet/Ryu
                           needed. Useful to sanity-check the calibration
                           LOGIC itself (which is what was done here), but
                           its noise-floor number is a synthetic stand-in
                           (see REACTIVE_LATENCY_MS in scripts/replay.py),
                           NOT a real measurement.

  --mode live              The real calibration: requires ryu-manager +
                           Mininet running on YOUR machine (see
                           scripts/run_live_experiment.py). Measures the
                           actual Packet-In -> flow-mod round trip.

Either mode reports, and this script does NOT auto-write into
shared/config.py -- you read the printed recommendation and edit
TABLE_CAPACITY / TABLE_PRESSURE_CAPACITY / FLOW_RATE_FLOWS_PER_SEC /
N_REPETITIONS / TAU_GRID yourself, once, before the main matrix starts.
That extra manual step is intentional: Section 2.3 Step 1 calls this
"locking" the values, and an auto-overwrite would make it too easy to
silently re-lock them mid-experiment.
"""
from __future__ import annotations
import argparse
import statistics
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import config as cfg
from shared.splits import build_split
from traffic.generators import generate_baseline
from predictors.markov import MarkovPredictor
from scripts.replay import replay_run


def estimate_noise_floor(events) -> dict:
    """
    Section 2.2: 'measured Packet-In-to-flow-mod round-trip time and its
    run-to-run variability serve as a noise floor; any proactive "latency
    saving" smaller than this floor is not treated as meaningful.'

    Computed here from a reactive-only run's per-flow latencies on MISSES
    only (hits are near-zero and not informative about controller RTT).
    """
    miss_latencies = [e.flow_setup_latency_ms for e in events if e.match_type == "miss"]
    if not miss_latencies:
        return {"median_ms": None, "iqr_ms": None, "n": 0}
    miss_latencies.sort()
    n = len(miss_latencies)
    median = statistics.median(miss_latencies)
    q1 = miss_latencies[n // 4]
    q3 = miss_latencies[(3 * n) // 4]
    return {"median_ms": median, "iqr_ms": q3 - q1, "n": n}


def estimate_working_set(signatures, idle_timeout: float, timestamps) -> dict:
    """
    Approximates how many DISTINCT signatures are "alive" (would still be
    resident under an unlimited table) at any point in the run, using a
    sliding window of `idle_timeout` seconds. This is the quantity that
    determines whether a given table capacity can create genuine pressure:
    if the working set never exceeds capacity, no eviction (harmful or
    otherwise) can ever occur regardless of tau.
    """
    import bisect
    keys = [s.key() for s in signatures]
    ts = list(timestamps)
    max_live, sum_live = 0, 0
    for i, t in enumerate(ts):
        window_start = t - idle_timeout
        lo = bisect.bisect_left(ts, window_start)
        live = len(set(keys[lo:i + 1]))
        max_live = max(max_live, live)
        sum_live += live
    return {"max_working_set": max_live, "mean_working_set": sum_live / len(ts) if ts else 0}


def recommend_capacity(working_set_stats: dict) -> dict:
    max_ws = working_set_stats["max_working_set"]
    lo, hi = cfg.TABLE_CAPACITY_RANGE
    # For baseline/table_pressure to show a REAL contrast between
    # comfortable and pressured operation: TABLE_CAPACITY should sit above
    # the working set (little/no pressure at reactive-only), and
    # TABLE_PRESSURE_CAPACITY should sit meaningfully below it.
    suggested_normal = max(lo, min(hi, int(max_ws * 1.3)))
    suggested_pressure = max(8, int(max_ws * 0.5))
    return {"suggested_TABLE_CAPACITY": suggested_normal,
            "suggested_TABLE_PRESSURE_CAPACITY": suggested_pressure,
            "observed_max_working_set": max_ws}


def pilot_effect_size(markov_events, lstm_events) -> dict | None:
    """
    Section 2.4's precision discussion: 'the pilot run includes a quick
    paired-difference check between Markov and LSTM latency-savings at a
    mid-range tau; if the pilot's paired spread suggests the true effect is
    likely small, repetitions are set to the upper end of the range (8)
    rather than the lower.' Returns None if lstm_events is None (e.g. torch
    unavailable in this environment) -- run this half on your own machine.
    """
    if lstm_events is None:
        return None
    m_lat = statistics.mean(e.flow_setup_latency_ms for e in markov_events)
    l_lat = statistics.mean(e.flow_setup_latency_ms for e in lstm_events)
    pooled_sd = statistics.pstdev(
        [e.flow_setup_latency_ms for e in markov_events] +
        [e.flow_setup_latency_ms for e in lstm_events]
    )
    cohens_d = (m_lat - l_lat) / pooled_sd if pooled_sd > 0 else 0.0
    return {"markov_mean_latency_ms": m_lat, "lstm_mean_latency_ms": l_lat, "cohens_d": cohens_d}


def run_replay_dry_run(n_train: int, n_eval: int, tau_for_pilot: float, try_lstm: bool):
    print("=== CALIBRATION: replay-dry-run mode (synthetic latency model, NOT real hardware) ===\n")

    split = build_split("baseline", n_train=n_train, n_validation=500, n_eval=n_eval,
                         eval_seeds=cfg.EVALUATION_SEEDS[:1])
    seed = cfg.EVALUATION_SEEDS[0]
    eval_sigs = split.eval_signatures_by_seed[seed]
    eval_ts = split.eval_timestamps_by_seed[seed]

    # 1) noise floor, from reactive-only
    reactive_events = replay_run("reactive", 1.0, "baseline", seed, 0, eval_sigs, eval_ts, predictor=None)
    noise = estimate_noise_floor(reactive_events)
    print(f"Noise floor (reactive-only, misses only): median={noise['median_ms']:.3f}ms  "
          f"IQR={noise['iqr_ms']:.3f}ms  (n={noise['n']})")
    print("  -> ON REAL HARDWARE, replace this with the equivalent live-run numbers before trusting it.\n")

    # 2) working-set / capacity recommendation
    ws = estimate_working_set(eval_sigs, cfg.IDLE_TIMEOUT_SEC, eval_ts)
    rec = recommend_capacity(ws)
    print(f"Working set: max={ws['max_working_set']}  mean={ws['mean_working_set']:.1f}")
    print(f"Suggested TABLE_CAPACITY (comfortable): {rec['suggested_TABLE_CAPACITY']}")
    print(f"Suggested TABLE_PRESSURE_CAPACITY (constrained): {rec['suggested_TABLE_PRESSURE_CAPACITY']}")
    lo, hi = cfg.TABLE_CAPACITY_RANGE
    if not (lo <= rec["suggested_TABLE_CAPACITY"] <= hi):
        print(f"  WARNING: suggested capacity falls outside the locked range {cfg.TABLE_CAPACITY_RANGE}; "
              f"consider adjusting n_states/fanout in traffic/generators.py rather than breaking the range.")
    print()

    # 3) pilot Markov vs LSTM effect size (LSTM optional)
    predictor = MarkovPredictor()
    predictor.fit([s.key() for s in split.train_signatures])
    markov_events = replay_run("markov", tau_for_pilot, "baseline", seed, 0, eval_sigs, eval_ts,
                                predictor=predictor)

    lstm_events = None
    if try_lstm:
        try:
            from predictors.lstm import LSTMPredictor
            lstm_predictor = LSTMPredictor()
            lstm_predictor.fit([s.key() for s in split.train_signatures])
            lstm_events = replay_run("lstm", tau_for_pilot, "baseline", seed, 0, eval_sigs, eval_ts,
                                      predictor=lstm_predictor)
        except ImportError as e:
            print(f"(LSTM pilot skipped: {e})\n")

    effect = pilot_effect_size(markov_events, lstm_events)
    if effect is None:
        print("Pilot Markov-vs-LSTM effect size: SKIPPED (no LSTM/torch available here).")
        print("  -> Run this script with torch installed on your machine before locking N_REPETITIONS.")
    else:
        print(f"Pilot Markov-vs-LSTM effect size at tau={tau_for_pilot}: "
              f"Cohen's d={effect['cohens_d']:.3f}  "
              f"(markov={effect['markov_mean_latency_ms']:.3f}ms, lstm={effect['lstm_mean_latency_ms']:.3f}ms)")
        if abs(effect["cohens_d"]) < 0.5:
            print(f"  -> Effect looks small-to-moderate; per Section 2.4, set N_REPETITIONS=8 (upper end), "
                  f"not {cfg.N_REPETITIONS}.")
        else:
            print(f"  -> Effect looks large; N_REPETITIONS={cfg.N_REPETITIONS} is likely adequate.")

    print("\n=== Reminder: none of shared/config.py was modified. Edit it by hand once you've")
    print("    decided on these numbers -- see the module docstring above. ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["replay-dry-run", "live"], default="replay-dry-run")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=1500)
    p.add_argument("--tau-for-pilot", type=float, default=0.5)
    p.add_argument("--no-lstm", action="store_true", help="skip the LSTM half even if torch is installed")
    args = p.parse_args()

    if args.mode == "replay-dry-run":
        run_replay_dry_run(args.n_train, args.n_eval, args.tau_for_pilot, try_lstm=not args.no_lstm)
    else:
        print("=== CALIBRATION: live mode ===")
        print("Not implemented as a single script -- live calibration is just a short,")
        print("deliberately real run of your normal live pipeline:")
        print("  1. terminal 1: STRATEGY=reactive TAU=1.0 SCENARIO=table_pressure SEED=999 OUT_DIR=logs/live "
              "~/ryu-venv/bin/python scripts/ryu_manager.py --ofp-tcp-listen-port 6653 controller/ryu_app.py")
        print("  2. terminal 2: sudo python3 scripts/run_live_experiment.py "
              "--scenario table_pressure --seed 999 --n-flows 1000")
        print("  3. Then run estimate_noise_floor() and estimate_working_set() from this file")
        print("     (import them) against the resulting logs/<run_id>.jsonl via shared.event_log.read_events.")


if __name__ == "__main__":
    main()
