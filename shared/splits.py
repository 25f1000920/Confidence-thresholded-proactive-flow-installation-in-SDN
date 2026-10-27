"""
Explicit train/validation/test split (Research Plan Section 2.2: "trained
on traffic drawn from the same generators but kept disjoint from the
traffic used for evaluation ... train/validation/test splits fixed and
disjoint at generation time").

Before this file, "disjoint" was only true by accident of using different
integers as seeds (shared/config.py's TRAIN_SEED, VALIDATION_SEED,
EVALUATION_SEEDS). This module makes it an explicit, asserted contract:
one function that returns all three traffic sets together, refuses to
proceed if any seed collides, and is the ONLY place predictor training
code (predictors/*.fit) and validation/tuning code should pull traffic
from -- so "which seeds were used for what" is never re-decided ad hoc
somewhere else in the codebase.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from shared import config as cfg
from shared.schema import FlowSignature
from traffic.generators import GENERATORS


@dataclass
class TrafficSplit:
    scenario: str
    train_signatures: List[FlowSignature]
    train_timestamps: List[float]
    validation_signatures: List[FlowSignature]
    validation_timestamps: List[float]
    # keyed by evaluation seed, so each repetition's traffic is retrievable by its seed
    eval_signatures_by_seed: dict
    eval_timestamps_by_seed: dict


def build_split(
    scenario: str,
    n_train: int,
    n_validation: int,
    n_eval: int,
    train_seed: int = cfg.TRAIN_SEED,
    validation_seed: int = cfg.VALIDATION_SEED,
    eval_seeds: List[int] | None = None,
) -> TrafficSplit:
    """
    Generates train / validation / evaluation traffic for `scenario` from
    explicitly distinct seeds, and asserts disjointness before returning
    anything -- this is meant to be the single choke point that fails
    loudly (rather than silently leaking) if two seeds ever collide, e.g.
    because someone widens EVALUATION_SEEDS to overlap TRAIN_SEED later.

    NOTE: disjoint seeds do not, by themselves, guarantee the *traffic
    sequences* share no timestep-for-timestep values -- they guarantee
    independent random walks through the (shared, fixed) world graph,
    which is the intended notion of "disjoint" here: no repetition's walk
    is used for both training and evaluating the same predictor instance.
    """
    if scenario not in GENERATORS:
        raise ValueError(f"unknown scenario {scenario!r}")
    eval_seeds = list(eval_seeds if eval_seeds is not None else cfg.EVALUATION_SEEDS)

    all_seeds = [train_seed, validation_seed, *eval_seeds]
    if len(set(all_seeds)) != len(all_seeds):
        dupes = {s for s in all_seeds if all_seeds.count(s) > 1}
        raise ValueError(f"train/validation/eval seeds are not disjoint: {dupes}")

    gen = GENERATORS[scenario]
    train_sigs, train_ts = gen(train_seed, n_train)
    val_sigs, val_ts = gen(validation_seed, n_validation)

    eval_sigs_by_seed, eval_ts_by_seed = {}, {}
    for s in eval_seeds:
        sigs, ts = gen(s, n_eval)
        eval_sigs_by_seed[s] = sigs
        eval_ts_by_seed[s] = ts

    return TrafficSplit(
        scenario=scenario,
        train_signatures=train_sigs, train_timestamps=train_ts,
        validation_signatures=val_sigs, validation_timestamps=val_ts,
        eval_signatures_by_seed=eval_sigs_by_seed, eval_timestamps_by_seed=eval_ts_by_seed,
    )
