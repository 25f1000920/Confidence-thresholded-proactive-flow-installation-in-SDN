"""
Offline replay harness.

Implements Section 2.3 Step 8's split: for baseline / cold_start / drift /
unstructured, prediction+confidence can be logged once per (strategy,
scenario, seed) and tau swept afterward in pure Python, because
installation decisions at one tau don't change future flows, predictions,
or table state relative to another tau *for these four scenarios only*
(no table-capacity interaction is being tested there). table_pressure is
deliberately NOT run through this file -- see controller/ryu_app.py.

This file must produce byte-for-byte the same event shape (schema.FlowEvent)
as a live run, since Section 2.4 analysis code must not care which one
produced a given log.
"""
from __future__ import annotations
from typing import List, Sequence
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.schema import FlowSignature, FlowEvent
from shared.event_log import EventLogWriter
from shared import config as cfg
from table_sim.flow_table import FlowTable, analyze_harmful_evictions, eviction_fields
from policy.threshold_policy import decide
from predictors.base import Prediction


# A fixed, small "controller round trip" cost model for reactive installs /
# misses, distinct from the near-zero cost of a table hit. This stands in
# for what the live Ryu app measures directly from real Packet-In/flow-mod
# timestamps (see controller/ryu_app.py) -- for offline replay we need an
# explicit model since there is no real controller round trip to time.
REACTIVE_LATENCY_MS = 8.0
HIT_LATENCY_MS = 0.5


def replay_run(
    strategy: str,
    tau: float,
    scenario: str,
    seed: int,
    repetition_index: int,
    signatures: List[FlowSignature],
    timestamps: List[float],
    predictor,  # None for reactive; a fitted Predictor otherwise
    table_capacity: int = cfg.TABLE_CAPACITY,
    idle_timeout: float = cfg.IDLE_TIMEOUT_SEC,
    out_path: str | None = None,
) -> List[FlowEvent]:
    assert scenario in cfg.OFFLINE_REPLAY_SCENARIOS, (
        f"{scenario} must be run live (controller/ryu_app.py), not replayed offline"
    )
    run_key_id = f"{strategy}__tau{tau:.1f}__{scenario}__seed{seed}"
    table = FlowTable(capacity=table_capacity, idle_timeout=idle_timeout)
    history: List[str] = []
    events: List[FlowEvent] = []

    writer = EventLogWriter(out_path) if out_path else None

    for flow_id, (sig, t) in enumerate(zip(signatures, timestamps)):
        key = sig.key()

        # 1) Proactive prediction+decision happens BEFORE the flow arrives,
        #    based on history seen so far (Figure 1).
        predicted_sig, confidence, decision, installed_proactively = None, None, "no_rule", False
        eviction = None
        if strategy != "reactive" and predictor is not None:
            window = history[-cfg.INPUT_WINDOW:]
            pred: Prediction = predictor.predict(window)
            predicted_sig, confidence = pred.signature, pred.confidence
            pd = decide(pred, tau)
            if pd.install_proactively:
                decision, installed_proactively = "proactive_install", True
                eviction = table.install(pd.signature, t)

        # 2) The flow itself arrives; check the table.
        match = table.match(key, t)
        controller_interaction = not match.hit
        latency = HIT_LATENCY_MS if match.hit else REACTIVE_LATENCY_MS
        reactive_eviction = None
        if not match.hit:
            # reactive install on miss, regardless of strategy (even a
            # proactive arm falls back reactively if its own proactive
            # install didn't happen to cover this exact flow)
            reactive_eviction = table.install(key, t)
            if decision == "no_rule":
                decision = "reactive_install"

        # A flow can trigger both a proactive and a reactive install; both evictions are counted.

        ev = FlowEvent(
            run_id=run_key_id, strategy=strategy, tau=tau, scenario=scenario,
            seed=seed, repetition_index=repetition_index,
            flow_id=flow_id, t_arrival=t, signature=key,
            predicted_signature=predicted_sig, confidence=confidence,
            decision=decision, installed_proactively=installed_proactively,
            match_type="hit" if match.hit else "miss",
            controller_interaction=controller_interaction,
            flow_setup_latency_ms=latency,
            **eviction_fields(eviction, reactive_eviction),
            table_occupancy_after=match.occupancy_after,
        )
        events.append(ev)
        history.append(key)

    analyze_harmful_evictions(events)

    if writer:
        for ev in events:
            writer.write(ev)
        writer.close()

    return events
