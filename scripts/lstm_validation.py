"""
LSTM predictor validation -- run on YOUR machine after:
    pip install torch --break-system-packages

Not runnable in this sandbox (no torch here). Mirrors scripts/smoke_test.py
but exercises predictors/lstm.py specifically, plus a couple of checks the
Markov predictor doesn't need (training actually converges; confidence
distribution isn't degenerate).

Usage:
    python3 scripts/lstm_validation.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.splits import build_split
from shared import config as cfg
from scripts.replay import replay_run


def main():
    try:
        from predictors.lstm import LSTMPredictor, LSTMConfig
    except ImportError as e:
        print(f"torch not available: {e}")
        print("Run: pip install torch --break-system-packages")
        sys.exit(1)

    print("Building baseline train/validation/eval split...")
    split = build_split("baseline", n_train=4000, n_validation=1000, n_eval=1500,
                         eval_seeds=cfg.EVALUATION_SEEDS[:2])
    train_keys = [s.key() for s in split.train_signatures]
    val_keys = [s.key() for s in split.validation_signatures]

    print(f"Training LSTM on {len(train_keys)} flows (small, fixed architecture)...")
    predictor = LSTMPredictor(LSTMConfig(epochs=5))
    predictor.fit(train_keys)

    # 1) sanity check: predictor should do noticeably better than chance on
    #    held-out VALIDATION traffic from the same (baseline) world.
    correct, total = 0, 0
    window = cfg.INPUT_WINDOW
    for i in range(len(val_keys) - window - 1):
        hist = val_keys[i:i + window]
        target = val_keys[i + window]
        pred = predictor.predict(hist)
        total += 1
        if pred.signature == target:
            correct += 1
    acc = correct / total if total else 0.0
    print(f"Validation next-token accuracy: {acc:.3f} ({correct}/{total})")
    if acc < 0.05:
        print("WARNING: accuracy looks close to chance -- check LSTMConfig.epochs, "
              "learning rate, or that the vocabulary/window matches training.")

    # 2) confidence distribution shouldn't be degenerate (always ~0 or always ~1)
    confidences = []
    for i in range(0, len(val_keys) - window - 1, 5):
        pred = predictor.predict(val_keys[i:i + window])
        confidences.append(pred.confidence)
    import statistics
    print(f"Confidence on validation: mean={statistics.mean(confidences):.3f} "
          f"stdev={statistics.pstdev(confidences):.3f} "
          f"min={min(confidences):.3f} max={max(confidences):.3f}")

    # 3) end-to-end replay sanity check, same shape as smoke_test.py's markov section
    seed = cfg.EVALUATION_SEEDS[0]
    eval_sigs = split.eval_signatures_by_seed[seed]
    eval_ts = split.eval_timestamps_by_seed[seed]
    for tau in [0.0, 0.3, 0.6, 0.9]:
        events = replay_run("lstm", tau, "baseline", seed, 0, eval_sigs, eval_ts, predictor=predictor)
        n = len(events)
        hits = sum(1 for e in events if e.match_type == "hit")
        proactive = sum(1 for e in events if e.installed_proactively)
        print(f"lstm / baseline / tau={tau}: n={n} hit_rate={hits/n:.2f} proactive={proactive}")

    print("\nIf proactive counts fall as tau rises (like the Markov smoke test), the LSTM")
    print("predictor is wired correctly end-to-end. Compare this file's accuracy/confidence")
    print("numbers against the Markov predictor's before trusting RQ2 results.")


if __name__ == "__main__":
    main()
