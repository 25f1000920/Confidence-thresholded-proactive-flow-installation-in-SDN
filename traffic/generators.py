"""
Traffic generators (Research Plan Section 2.2).

All five scenarios share one host/port vocabulary (config.N_HOSTS hosts,
a small fixed set of "service" ports) and produce List[FlowSignature] with
arrival timestamps at config.FLOW_RATE_FLOWS_PER_SEC. Each generator takes
an explicit `seed` so repetitions vary the traffic realization
(Section 2.1: "Repetitions vary the random seed governing traffic
generation") while remaining fully reproducible.

- baseline:      periodic, repeating host-pair traffic with genuine
                  learnable structure (a random walk over a small, fixed
                  transition graph over (host, port) states).
- cold_start:     baseline-pattern traffic interleaved with flow signatures
                  that never appear in the training vocabulary at all.
- drift:          baseline pattern A for the first half of the run, then an
                  abrupt switch to an unrelated pattern B for the second half.
- unstructured:   uniform-random host/port selection; no learnable structure
                  by construction. Doubles as the calibration test.
- table_pressure: reuses baseline traffic verbatim -- pressure comes from
                  running it against config.TABLE_PRESSURE_CAPACITY at low
                  tau, not from a different traffic shape (see config.py).
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List

from shared.schema import FlowSignature
from shared import config as cfg

SERVICE_PORTS = [80, 443, 22, 3306, 8080]


def _hosts(n_hosts: int) -> List[str]:
    return [f"h{i}" for i in range(1, n_hosts + 1)]


def _build_transition_graph(rng: random.Random, hosts: List[str], n_states: int = 120, fanout: int = 3):
    """
    A small, fixed set of (src, dst, port) "states" with a sparse, skewed
    transition graph between them -- this is what makes baseline traffic
    genuinely learnable (a Markov predictor with the true graph would get
    high, not just nonzero, accuracy).

    IMPORTANT: `rng` here must be seeded from a fixed WORLD seed, not from
    the per-repetition traffic seed. The graph is "the world" -- shared,
    fixed structure that both training traffic and every evaluation
    repetition are drawn FROM. The per-repetition seed instead varies only
    the random WALK through this fixed graph (see `_walk` / the `seed`
    argument of each generate_* function below). Building a fresh graph
    per traffic seed (an earlier bug in this file) makes train and eval
    traffic structurally disjoint -- the predictor then has nothing to
    learn that transfers, which silently zeroes out every downstream
    proactive-install result. See tests/test_generators.py.
    """
    states = []
    for _ in range(n_states):
        src, dst = rng.sample(hosts, 2)
        port = rng.choice(SERVICE_PORTS)
        states.append(FlowSignature(src_host=src, dst_host=dst, dst_port=port).key())

    graph: dict[str, list[tuple[str, float]]] = {}
    for s in states:
        targets = rng.sample([t for t in states if t != s], k=min(fanout, len(states) - 1))
        weights = sorted([rng.random() for _ in targets], reverse=True)
        total = sum(weights)
        graph[s] = list(zip(targets, [w / total for w in weights]))
    return states, graph


# Fixed world seeds: define the transition graph(s) ONCE, independent of
# the per-repetition traffic seed. Locked alongside the tau grid and table
# capacity (Section 2.3 Step 1) -- do not change after the main matrix starts.
WORLD_SEED_A = 777          # baseline / cold_start / table_pressure / drift's first half
WORLD_SEED_B = 778          # drift's second half (structurally unrelated pattern)

_world_cache: dict[int, tuple] = {}


def _get_world(world_seed: int, n_hosts: int, n_states: int = 120, fanout: int = 3):
    cache_key = (world_seed, n_hosts, n_states, fanout)
    if cache_key not in _world_cache:
        rng = random.Random(world_seed)
        hosts = _hosts(n_hosts)
        _world_cache[cache_key] = _build_transition_graph(rng, hosts, n_states, fanout)
    return _world_cache[cache_key]


def _walk(rng: random.Random, states, graph, n_flows: int, start: str | None = None) -> List[str]:
    current = start or rng.choice(states)
    out = [current]
    for _ in range(n_flows - 1):
        targets, weights = zip(*graph[current])
        current = rng.choices(targets, weights=weights, k=1)[0]
        out.append(current)
    return out


def _sig_from_key(key: str, src_port: int = 0) -> FlowSignature:
    # key format: "src->dst:port"
    left, port = key.rsplit(":", 1)
    src, dst = left.split("->")
    return FlowSignature(src_host=src, dst_host=dst, dst_port=int(port), src_port=src_port)


def _timestamps(n_flows: int, rate: float, rng: random.Random) -> List[float]:
    """Poisson-ish arrivals at the target aggregate rate."""
    t = 0.0
    out = []
    for _ in range(n_flows):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def generate_baseline(seed: int, n_flows: int, n_hosts: int = cfg.N_HOSTS,
                       rate: float = cfg.FLOW_RATE_FLOWS_PER_SEC) -> List[FlowSignature]:
    """
    `seed` varies ONLY the random walk (which path through the fixed world
    graph this realization takes) and the arrival timestamps. The graph
    itself (WORLD_SEED_A) is shared across every call, train or eval,
    which is what makes it possible for a predictor trained on one seed's
    walk to generalize to another seed's walk -- both are draws from the
    same underlying structure, per Section 2.2 ("trained on traffic drawn
    from the same generators but kept disjoint from the traffic used for
    evaluation").
    """
    rng = random.Random(seed)
    states, graph = _get_world(WORLD_SEED_A, n_hosts)
    keys = _walk(rng, states, graph, n_flows)
    ts = _timestamps(n_flows, rate, rng)
    sigs = [_sig_from_key(k, src_port=40000 + i % 1000) for i, k in enumerate(keys)]
    return sigs, ts


def generate_cold_start(seed: int, n_flows: int, n_hosts: int = cfg.N_HOSTS,
                         rate: float = cfg.FLOW_RATE_FLOWS_PER_SEC,
                         novel_fraction: float = 0.35) -> List[FlowSignature]:
    """
    Baseline-pattern traffic (same transition graph, so the *familiar*
    portion is exactly as learnable as baseline) with `novel_fraction` of
    flows replaced by signatures using host/port combinations the graph
    never produces -- unseen at training time by construction.
    """
    rng = random.Random(seed)
    hosts = _hosts(n_hosts)
    states, graph = _get_world(WORLD_SEED_A, n_hosts)  # SAME world as baseline/training
    keys = _walk(rng, states, graph, n_flows)
    known_pairs = {tuple(k.split("->")) for k in states}

    sigs, ts = [], _timestamps(n_flows, rate, rng)
    novel_port_pool = [p for p in range(9000, 9100)]  # ports never used by baseline's SERVICE_PORTS
    for i, k in enumerate(keys):
        if rng.random() < novel_fraction:
            src, dst = rng.sample(hosts, 2)
            sig = FlowSignature(src_host=src, dst_host=dst, dst_port=rng.choice(novel_port_pool),
                                 src_port=50000 + i % 1000)
        else:
            sig = _sig_from_key(k, src_port=40000 + i % 1000)
        sigs.append(sig)
    return sigs, ts


def generate_drift(seed: int, n_flows: int, n_hosts: int = cfg.N_HOSTS,
                    rate: float = cfg.FLOW_RATE_FLOWS_PER_SEC,
                    shift_at_fraction: float = 0.5) -> List[FlowSignature]:
    """
    Pattern A (baseline graph) for the first `shift_at_fraction` of the run,
    then an abrupt, unrelated pattern B for the rest -- tests whether a
    predictor trained on A recovers, and how quickly, once B takes over.
    """
    rng = random.Random(seed)
    hosts = _hosts(n_hosts)
    n_a = int(n_flows * shift_at_fraction)
    n_b = n_flows - n_a

    states_a, graph_a = _get_world(WORLD_SEED_A, n_hosts)               # same world as baseline/training
    keys_a = _walk(rng, states_a, graph_a, n_a)

    # pattern B: a second FIXED world, structurally unrelated to A, shared
    # across repetitions the same way world A is (only the walk varies with seed).
    rng_b = random.Random(seed + 10_000)
    states_b, graph_b = _get_world(WORLD_SEED_B, n_hosts, n_states=100, fanout=4)
    keys_b = _walk(rng_b, states_b, graph_b, n_b)

    keys = keys_a + keys_b
    ts = _timestamps(n_flows, rate, rng)
    sigs = [_sig_from_key(k, src_port=40000 + i % 1000) for i, k in enumerate(keys)]
    return sigs, ts


def generate_unstructured(seed: int, n_flows: int, n_hosts: int = cfg.N_HOSTS,
                           rate: float = cfg.FLOW_RATE_FLOWS_PER_SEC) -> List[FlowSignature]:
    """Uniform-random host/port selection: no learnable structure by construction."""
    rng = random.Random(seed)
    hosts = _hosts(n_hosts)
    ts = _timestamps(n_flows, rate, rng)
    sigs = []
    for i in range(n_flows):
        src, dst = rng.sample(hosts, 2)
        port = rng.choice(SERVICE_PORTS)
        sigs.append(FlowSignature(src_host=src, dst_host=dst, dst_port=port, src_port=50000 + i % 1000))
    return sigs, ts


def generate_table_pressure(seed: int, n_flows: int, n_hosts: int = cfg.N_HOSTS,
                             rate: float = cfg.FLOW_RATE_FLOWS_PER_SEC) -> List[FlowSignature]:
    """Identical shape to baseline; pressure comes from capacity + tau at run time, not traffic shape."""
    return generate_baseline(seed, n_flows, n_hosts, rate)


GENERATORS = {
    "baseline": generate_baseline,
    "cold_start": generate_cold_start,
    "drift": generate_drift,
    "unstructured": generate_unstructured,
    "table_pressure": generate_table_pressure,
}
