"""
Runs the FULL offline-replay portion of the experiment matrix:

    strategies {reactive, markov, lstm}
    x TAU_GRID (11 values, only meaningful for markov/lstm)
    x OFFLINE_REPLAY_SCENARIOS (baseline, cold_start, drift, unstructured)
    x EVALUATION_SEEDS (repetitions)

table_pressure is NOT here -- it must be run live (see run_live_matrix.py),
per Section 2.3 Step 8 (an eviction at one tau changes which future flows
miss at a different tau, so it can't be swept in post-processing).

Both predictors are trained ONCE, on baseline traffic only (Section 2.2:
trained once on a fixed seed, reused unchanged across repetitions and tau
values) -- NOT retrained per scenario. cold_start/drift/unstructured
deliberately test how a baseline-trained predictor degrades when the
traffic it sees at evaluation time no longer matches what it learned.

Usage:
    python3 scripts/run_offline_matrix.py                      # everything
    python3 scripts/run_offline_matrix.py --strategies markov  # subset
    python3 scripts/run_offline_matrix.py --dry-run            # tiny, fast, for wiring checks

Output: logs/offline/<scenario>/<strategy>__tau<tau>__seed<seed>.jsonl
        one JSONL event log per (strategy, tau, scenario, seed) point.
        reactive-only is tau-independent, so it is run once per
        (scenario, seed) and logged under tau=1.0 by convention
        (see shared/config.py's RunKey docstring).
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import config as cfg
from shared.splits import build_split
from traffic.generators import GENERATORS
from predictors.markov import MarkovPredictor
from scripts.replay import replay_run


def train_predictors(n_train: int, n_validation: int, use_lstm: bool):
    """Trains on BASELINE traffic only, once, per Section 2.2."""
    split = build_split("baseline", n_train=n_train, n_validation=n_validation, n_eval=1,
                         eval_seeds=[cfg.EVALUATION_SEEDS[0]])
    train_keys = [s.key() for s in split.train_signatures]

    predictors = {}
    markov = MarkovPredictor()
    markov.fit(train_keys)
    predictors["markov"] = markov
    print(f"Trained markov on {len(train_keys)} baseline flows.")

    if use_lstm:
        try:
            from predictors.lstm import LSTMPredictor
            lstm = LSTMPredictor()
            lstm.fit(train_keys)
            predictors["lstm"] = lstm
            print(f"Trained lstm on {len(train_keys)} baseline flows.")
        except ImportError as e:
            print(f"Skipping lstm ({e}). Install torch on this machine to include it.")

    return predictors


def run_matrix(strategies, taus, scenarios, seeds, n_eval: int, out_dir: str, predictors: dict):
    t_start = time.time()
    n_runs = 0

    for scenario in scenarios:
        assert scenario in cfg.OFFLINE_REPLAY_SCENARIOS, f"{scenario} must run live, not here"
        gen = GENERATORS[scenario]
        scenario_dir = os.path.join(out_dir, scenario)

        for rep_idx, seed in enumerate(seeds):
            sigs, ts = gen(seed, n_eval)

            if "reactive" in strategies:
                out_path = os.path.join(scenario_dir, f"reactive__tau1.0__seed{seed}.jsonl")
                replay_run("reactive", 1.0, scenario, seed, rep_idx, sigs, ts,
                           predictor=None, out_path=out_path)
                n_runs += 1

            for strategy in strategies:
                if strategy == "reactive":
                    continue
                predictor = predictors.get(strategy)
                if predictor is None:
                    continue  # e.g. lstm requested but unavailable here
                for tau in taus:
                    out_path = os.path.join(scenario_dir, f"{strategy}__tau{tau:.1f}__seed{seed}.jsonl")
                    replay_run(strategy, tau, scenario, seed, rep_idx, sigs, ts,
                               predictor=predictor, out_path=out_path)
                    n_runs += 1

            print(f"[{time.time()-t_start:6.1f}s] scenario={scenario:14s} seed={seed}  "
                  f"({rep_idx+1}/{len(seeds)} repetitions done)")

    elapsed = time.time() - t_start
    print(f"\nDone: {n_runs} runs written under {out_dir}/ in {elapsed:.1f}s "
          f"({elapsed/max(n_runs,1)*1000:.1f}ms/run avg).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+", default=cfg.STRATEGIES, choices=cfg.STRATEGIES)
    p.add_argument("--scenarios", nargs="+", default=cfg.OFFLINE_REPLAY_SCENARIOS,
                    choices=cfg.OFFLINE_REPLAY_SCENARIOS)
    p.add_argument("--taus", nargs="+", type=float, default=cfg.TAU_GRID)
    p.add_argument("--seeds", nargs="+", type=int, default=cfg.EVALUATION_SEEDS)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-validation", type=int, default=500)
    p.add_argument("--n-eval", type=int, default=1500)
    p.add_argument("--out-dir", default="logs/offline")
    p.add_argument("--no-lstm", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                    help="Tiny fast run (1 tau, 1 scenario, 1 seed, few flows) to check wiring, "
                         "not to collect real data.")
    args = p.parse_args()

    strategies, scenarios, taus, seeds, n_eval = (
        args.strategies, args.scenarios, args.taus, args.seeds, args.n_eval
    )
    if args.dry_run:
        strategies = ["reactive", "markov"]
        scenarios = [scenarios[0]]
        taus = [0.0, 0.5, 1.0]
        seeds = seeds[:1]
        n_eval = 200
        print("--dry-run: tiny wiring check only, NOT a real data-collection run.\n")

    use_lstm = ("lstm" in strategies) and not args.no_lstm
    predictors = train_predictors(args.n_train, args.n_validation, use_lstm=use_lstm)

    run_matrix(strategies, taus, scenarios, seeds, n_eval, args.out_dir, predictors)


if __name__ == "__main__":
    main()
