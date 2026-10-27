import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.splits import build_split
from shared import config as cfg


def test_split_is_disjoint_and_correct_sizes():
    split = build_split("baseline", n_train=500, n_validation=200, n_eval=300,
                         eval_seeds=cfg.EVALUATION_SEEDS[:3])
    assert len(split.train_signatures) == 500
    assert len(split.validation_signatures) == 200
    assert set(split.eval_signatures_by_seed.keys()) == set(cfg.EVALUATION_SEEDS[:3])
    for seed, sigs in split.eval_signatures_by_seed.items():
        assert len(sigs) == 300
    print("test_split_is_disjoint_and_correct_sizes: PASS")


def test_split_rejects_colliding_seeds():
    try:
        build_split("baseline", n_train=100, n_validation=100, n_eval=100,
                     train_seed=42, validation_seed=42)  # deliberately colliding
        raise AssertionError("expected ValueError for colliding seeds")
    except ValueError:
        pass
    print("test_split_rejects_colliding_seeds: PASS")


def test_split_reproducible_across_calls():
    s1 = build_split("baseline", n_train=200, n_validation=100, n_eval=100,
                      eval_seeds=[cfg.EVALUATION_SEEDS[0]])
    s2 = build_split("baseline", n_train=200, n_validation=100, n_eval=100,
                      eval_seeds=[cfg.EVALUATION_SEEDS[0]])
    assert [s.key() for s in s1.train_signatures] == [s.key() for s in s2.train_signatures]
    print("test_split_reproducible_across_calls: PASS")


if __name__ == "__main__":
    test_split_is_disjoint_and_correct_sizes()
    test_split_rejects_colliding_seeds()
    test_split_reproducible_across_calls()
    print("\nAll split tests passed.")
