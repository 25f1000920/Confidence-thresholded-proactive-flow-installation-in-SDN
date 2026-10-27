"""
Shared, centrally-locked experiment configuration.

Per Research Plan Section 2.2/2.3: the tau grid, table capacity, idle timeout,
topology size, traffic rate, and seeds are fixed ONCE here before any traffic
is generated or any run is executed. Every other module imports from here
rather than hard-coding these values, so a change is visible and auditable
in one place (this also satisfies the "risk 4" mitigation in Section 3.3:
a common, centrally-defined configuration across all modules).

NOTE on placeholder values: fields marked "TO LOCK DURING CALIBRATION" are
intentionally left as a starting point / range. Per Section 2.3 Step 1, run
scripts/calibrate.py (see scripts/) against your real Mininet/Ryu setup to
pick concrete numbers, then hard-set them here and do not change them again
once the main experiment matrix starts (this is what "locking" means).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Strategies (installation policies) and scenarios (traffic conditions)
# ---------------------------------------------------------------------------
STRATEGIES = ["reactive", "markov", "lstm"]

# Scenario names as used consistently across generators, event logs, and
# analysis. "table_pressure" reuses baseline traffic but is run live
# (Section 2.3 Step 8) rather than replayed offline, and is evaluated only
# at low tau against a small fixed table capacity.
SCENARIOS = ["baseline", "cold_start", "drift", "unstructured", "table_pressure"]

# Scenarios whose predictions/decisions do NOT depend on installation policy
# and can therefore be logged once and swept over tau in post-processing.
OFFLINE_REPLAY_SCENARIOS = ["baseline", "cold_start", "drift", "unstructured"]
# Scenario that must be run live, once per tau, because an eviction at one
# tau changes which future flows miss at a different tau.
LIVE_ONLY_SCENARIOS = ["table_pressure"]

assert set(OFFLINE_REPLAY_SCENARIOS) | set(LIVE_ONLY_SCENARIOS) == set(SCENARIOS)


# ---------------------------------------------------------------------------
# The locked tau grid (Section 2.3 Step 1)
# ---------------------------------------------------------------------------
TAU_GRID_FULL: List[float] = [round(0.1 * i, 1) for i in range(11)]  # 0.0 .. 1.0
TAU_GRID_COARSE: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # fallback if time-constrained

# Set this to TAU_GRID_COARSE only if the pilot run shows the full grid is
# infeasible within the time budget (Section 2.3 Step 1 / Section 3.3).
TAU_GRID: List[float] = TAU_GRID_FULL


# ---------------------------------------------------------------------------
# Topology / traffic-rate parameters (Section 2.2) -- TO LOCK DURING CALIBRATION
# ---------------------------------------------------------------------------
N_HOSTS: int = 16                 # flat, single-switch topology
FLOW_RATE_RANGE = (150, 400)      # flows/sec, aggregate target; pick one value after calibration
FLOW_RATE_FLOWS_PER_SEC: float = 250.0  # placeholder pick within range; recalibrate on your machine

IDLE_TIMEOUT_SEC: int = 5         # OpenFlow idle-timeout defining "natural TTL expiry"

# Fixed, constrained flow-table capacity (entries). Locked within 100-250.
TABLE_CAPACITY_RANGE = (100, 250)
TABLE_CAPACITY: int = 150         # placeholder pick; recalibrate on your machine

# Table-pressure scenario deliberately uses a SMALLER capacity to guarantee
# pressure is reachable within a short run even at moderate tau.
TABLE_PRESSURE_CAPACITY: int = 80


# ---------------------------------------------------------------------------
# Repetitions / seeds (Section 2.1, 2.3 Step 1, 2.4)
# ---------------------------------------------------------------------------
N_REPETITIONS: int = 8            # within the locked 5-8 range; raise to 8 if pilot shows a small RQ2 effect
EVALUATION_SEEDS: List[int] = list(range(1000, 1000 + N_REPETITIONS))
TRAIN_SEED: int = 42              # single fixed seed for predictor training data (disjoint from eval seeds)
VALIDATION_SEED: int = 43


# ---------------------------------------------------------------------------
# Predictor-shared parameters (Section 2.2: same horizon/window for both)
# ---------------------------------------------------------------------------
PREDICTION_HORIZON: int = 1       # predict the very next flow signature
INPUT_WINDOW: int = 8             # number of past flow signatures fed to each predictor


@dataclass(frozen=True)
class RunKey:
    """Identifies one point in the experimental matrix (strategy x tau x scenario x repetition)."""
    strategy: str
    tau: float
    scenario: str
    seed: int

    def __post_init__(self):
        assert self.strategy in STRATEGIES, f"unknown strategy {self.strategy}"
        assert self.scenario in SCENARIOS, f"unknown scenario {self.scenario}"
        if self.strategy == "reactive":
            assert self.tau == 1.0 or self.tau is None, (
                "reactive-only installs nothing; tau is not meaningful for it. "
                "Use tau=1.0 as a convention so RunKey stays hashable/sortable."
            )

    def run_id(self) -> str:
        return f"{self.strategy}__tau{self.tau:.1f}__{self.scenario}__seed{self.seed}"
