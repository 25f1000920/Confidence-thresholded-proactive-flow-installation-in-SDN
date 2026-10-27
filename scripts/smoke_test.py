"""
End-to-end smoke test: generate baseline + all offline-replayable traffic,
train the Markov predictor, replay it at a few tau values, and sanity-check
that the aggregate numbers move in the expected direction (higher tau ->
fewer proactive installs; harmful evictions only appear under real pressure).

This is NOT a substitute for Section 2.3's real work procedure -- it is a
fast, cheap way to catch integration bugs (schema drift, off-by-ones, etc.)
before spending Mininet/Ryu wall-clock time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic.generators import generate_baseline, GENERATORS
from predictors.markov import MarkovPredictor
from scripts.replay import replay_run
from shared import config as cfg


def summarize(events, label):
    n = len(events)
    hits = sum(1 for e in events if e.match_type == "hit")
    proactive = sum(1 for e in events if e.installed_proactively)
    evictions = sum(e.n_evictions for e in events)
    harmful = sum(1 for e in events if e.harmful_eviction)
    avg_latency = sum(e.flow_setup_latency_ms for e in events) / n
    print(f"{label:40s} n={n:4d}  hit_rate={hits/n:5.2f}  proactive={proactive:4d}  "
          f"evictions={evictions:3d}  harmful={harmful:3d}  avg_latency_ms={avg_latency:5.2f}")


def main():
    n_train, n_eval = 4000, 1500

    # 1) Train the Markov predictor on TRAIN_SEED traffic (disjoint from eval seeds)
    train_sigs, _ = generate_baseline(cfg.TRAIN_SEED, n_train)
    predictor = MarkovPredictor()
    predictor.fit([s.key() for s in train_sigs])

    # 2) Reactive-only baseline (no predictor) at the first eval seed
    seed = cfg.EVALUATION_SEEDS[0]
    eval_sigs, ts = generate_baseline(seed, n_eval)
    reactive_events = replay_run("reactive", 1.0, "baseline", seed, 0, eval_sigs, ts, predictor=None)
    summarize(reactive_events, "reactive-only / baseline")

    # 3) Markov at a few tau values on the SAME seed (paired comparison, per Section 2.1)
    for tau in [0.0, 0.3, 0.6, 0.9]:
        events = replay_run("markov", tau, "baseline", seed, 0, eval_sigs, ts, predictor=predictor)
        summarize(events, f"markov / baseline / tau={tau}")

    # 4) One stress scenario: unstructured, same tau sweep -- expect near-zero
    #    proactive coverage even at low tau, since a well-behaved Markov
    #    predictor should rarely clear any tau on unseen random transitions.
    print()
    for scenario in ["cold_start", "drift", "unstructured"]:
        gen = GENERATORS[scenario]
        sigs, ts_s = gen(seed, n_eval)
        events = replay_run("markov", 0.3, scenario, seed, 0, sigs, ts_s, predictor=predictor)
        summarize(events, f"markov / {scenario} / tau=0.3")

    # 5) table_pressure: same traffic as baseline, but small capacity + low tau
    print()
    tp_sigs, tp_ts = GENERATORS["table_pressure"](seed, n_eval)
    for tau in [0.0, 0.3, 0.9]:
        events = replay_run("markov", tau, "baseline", seed, 0, tp_sigs, tp_ts, predictor=predictor,
                             table_capacity=cfg.TABLE_PRESSURE_CAPACITY)
        summarize(events, f"markov / table_pressure(cap={cfg.TABLE_PRESSURE_CAPACITY}) / tau={tau}")


if __name__ == "__main__":
    main()
