"""
Shared data schema (Research Plan Section 2.3 Step 1: "the flow-signature
schema, the per-flow event-log schema ... are fixed first").

Every generator, predictor, the table simulator, and the live Ryu controller
import FlowSignature and FlowEvent from here. Nothing downstream should
invent its own ad-hoc tuple/dict shape for a flow.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Literal
import json


@dataclass(frozen=True)
class FlowSignature:
    """
    A flow signature: the (src_host, dst_host, dst_port) tuple that both the
    predictors and the flow table key on. src_port is deliberately excluded
    from the *prediction target* (ephemeral client ports are not usefully
    predictable) but is retained on the wire in `full_key` for realism /
    table-matching purposes.
    """
    src_host: str
    dst_host: str
    dst_port: int
    src_port: int = 0  # not part of the predicted signature; informational only

    def key(self) -> str:
        """The string the predictors and table actually predict/match on."""
        return f"{self.src_host}->{self.dst_host}:{self.dst_port}"

    def full_key(self) -> str:
        return f"{self.src_host}:{self.src_port}->{self.dst_host}:{self.dst_port}"


EvictionCause = Optional[Literal["capacity_lru", "capacity_random", "ttl", None]]
Decision = Literal["proactive_install", "no_rule", "reactive_install"]
MatchType = Literal["hit", "miss", None]


@dataclass
class FlowEvent:
    """
    One row of the per-flow event log (Research Plan Section 2.3 Step 1 /
    Section 2.4: "timestamp, signature, confidence, installed/not,
    matched/not, evicted/not, eviction cause"). All Section 2.4 metrics
    (coverage, hit rate, occupancy, eviction counts, harmful-eviction
    counts, Packet-In/flow-mod counts, controller processing time) are
    derived by aggregating these rows -- never computed inline elsewhere,
    so there is exactly one source of truth.
    """
    # -- identity of the run this event belongs to --
    run_id: str
    strategy: str
    tau: float
    scenario: str
    seed: int
    repetition_index: int

    # -- the flow itself --
    flow_id: int                 # monotonically increasing index within the run
    t_arrival: float             # simulated/wall-clock arrival time (seconds)
    signature: str               # FlowSignature.key()

    # -- prediction (None for reactive-only, which never predicts) --
    predicted_signature: Optional[str] = None
    confidence: Optional[float] = None

    # -- installation decision (Figure 1 logic) --
    decision: Decision = "no_rule"
    installed_proactively: bool = False

    # -- what happened when the flow actually arrived --
    match_type: MatchType = None          # "hit" if a rule was already installed, else "miss"
    controller_interaction: bool = False   # Packet-In -> controller round trip occurred
    flow_setup_latency_ms: Optional[float] = None

    # -- table dynamics caused by installing this flow's rule (if any) --
    caused_eviction: bool = False
    evicted_signature: Optional[str] = None
    eviction_cause: EvictionCause = None
    # The evicted entry's own natural-expiry deadline at the moment it was
    # evicted (its last_used + idle_timeout). Required, exactly, by the
    # harmful-eviction definition's condition (ii); see table_sim/flow_table.py.
    evicted_natural_expiry: Optional[float] = None
    table_occupancy_after: Optional[int] = None
    # One flow can trigger TWO installs (its own reactive install and a proactive
    # install for the predicted next flow), and each can evict a rule. The
    # evicted_* fields above describe the proactive eviction when there is one,
    # otherwise the reactive one; `evicted_by` says which. `n_evictions` (0-2) is
    # the number to SUM for eviction counts -- never count caused_eviction rows.
    n_evictions: int = 0
    evicted_by: Optional[str] = None                 # "proactive" | "reactive" | None
    reactive_evicted_signature: Optional[str] = None  # set only when BOTH installs evicted

    # -- filled in retrospectively by table_sim.harmful_eviction analysis --
    harmful_eviction: Optional[bool] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> "FlowEvent":
        d = json.loads(line)
        known = FlowEvent.__dataclass_fields__
        return FlowEvent(**{k: v for k, v in d.items() if k in known})  # tolerates older/newer logs
