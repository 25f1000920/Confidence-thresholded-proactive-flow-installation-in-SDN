"""
Live Ryu controller app (Research Plan Section 2.3 Steps 3, 8, 9: run live,
once per tau, for the table_pressure scenario -- and can also be used to
validate the offline-replay engine against a live run at one tau, per
Step 8's last sentence).

Run on YOUR Linux machine (needs Mininet + Ryu; not runnable in this
sandbox). Use scripts/ryu_manager.py instead of the bare `ryu-manager`
command (see that file for why), and pass settings as ENVIRONMENT
VARIABLES, not CLI flags -- see _parse_args() below for why:
    STRATEGY=markov TAU=0.3 SCENARIO=table_pressure SEED=1000 OUT_DIR=logs/live \
        python3 scripts/ryu_manager.py --ofp-tcp-listen-port 6653 controller/ryu_app.py
(`python3` here must be the interpreter that has ryu installed, e.g. ~/ryu-venv/bin/python.)

WHY CAPACITY IS ENFORCED IN THE CONTROLLER, NOT THE SWITCH:
Commodity OpenFlow TCAM capacity is on the order of thousands of entries
(Research Plan Section 2.2), far above the 100-250-entry range this study
needs to create genuine pressure within a short emulated run. Rather than
fight the switch's real (much larger) hardware limit, this app tracks its
own logical `table_sim.FlowTable` of size cfg.TABLE_CAPACITY /
cfg.TABLE_PRESSURE_CAPACITY, and explicitly sends an OFPFC_DELETE_STRICT
flow-mod to evict a rule from the real switch whenever its own logical
table decides to evict -- so the switch's real table occupancy always
mirrors the logical one being measured. This keeps table_sim.FlowTable the
single source of truth for eviction, identical to offline replay.

WHAT "PROACTIVE" MEANS HERE:
On every Packet-In (a flow arriving reactively, i.e. the flow itself was
NOT preinstalled), the app (a) handles that flow reactively as normal, and
(b) asks the predictor what it thinks comes NEXT given the arrival history
so far, and -- if confidence >= tau -- proactively pushes a flow-mod for
that predicted future signature before it arrives. This is exactly Figure
1's per-flow loop, just triggered by real Packet-In events instead of a
pre-generated timestamp list.

This file intentionally does NOT generate traffic itself -- pair it with a
traffic-player script (see scripts/live_traffic_player.py, TODO) that
sends real packets between Mininet hosts according to one of
traffic/generators.py's sequences, so Packet-In arrival order/timing on
the wire matches the corresponding offline scenario.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Python 3.10+ compatibility shim, required before importing ryu ---
# Ryu hasn't had a release since ~2021 and its code (plus eventlet, its
# concurrency dependency) still expects collections.MutableMapping,
# collections.Mapping, etc. to exist directly under `collections`. Python
# moved these to collections.abc in 3.3 and removed the `collections`-level
# aliases entirely in 3.10, so importing ryu on a modern Ubuntu (3.10+)
# normally crashes with "AttributeError: module 'collections' has no
# attribute 'MutableMapping'". This restores the old aliases before ryu is
# imported. Harmless no-op on Python <3.10, where they already exist.
import collections
import collections.abc as _collections_abc
for _name in ("Mapping", "MutableMapping", "Sequence", "MutableSequence", "Set", "MutableSet"):
    if not hasattr(collections, _name) and hasattr(_collections_abc, _name):
        setattr(collections, _name, getattr(_collections_abc, _name))
# --- end compatibility shim ---

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, CONFIG_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp

from shared.schema import FlowSignature, FlowEvent
from shared.event_log import EventLogWriter
from shared import config as cfg
from table_sim.flow_table import FlowTable, eviction_fields
from policy.threshold_policy import decide
from predictors.markov import MarkovPredictor
from predictors.base import Predictor
from traffic.generators import generate_baseline
from shared.live_utils import ip_to_host, host_to_ip, host_to_port, parse_payload

# Cookie stamped on every rule WE install. A Packet-In whose cookie is not this
# value came from the table-miss entry, i.e. the switch had no rule: a true miss.
RULE_COOKIE = 0x1


class _Args:
    """Plain namespace mirroring what the old argparse.Namespace provided,
    so the rest of this file (self.args.strategy, self.args.tau, etc.)
    didn't need to change."""
    def __init__(self, strategy, tau, scenario, seed, repetition_index, capacity, out_dir):
        self.strategy = strategy
        self.tau = tau
        self.scenario = scenario
        self.seed = seed
        self.repetition_index = repetition_index
        self.capacity = capacity
        self.out_dir = out_dir


def _parse_args() -> _Args:
    """
    Reads run configuration from ENVIRONMENT VARIABLES, not from
    sys.argv/CLI flags.

    WHY: `ryu-manager` parses sys.argv itself (via oslo.config), and it
    only recognizes its own built-in options plus the app's path -- any
    extra flag like --strategy or --tau gets rejected with
    "ryu_manager: error: unrecognized arguments: ...", because oslo.config
    requires custom options to be registered through its own
    `cfg.CONF.register_opts(...)` mechanism before parsing, which is a lot
    of extra Ryu-specific machinery to get right just to pass a few
    values through. Environment variables sidestep this entirely: Ryu
    never inspects them, so there's nothing for its parser to reject.

    Set these before running `python3 scripts/ryu_manager.py
    controller/ryu_app.py` (see that file's updated usage example):

        STRATEGY=markov TAU=0.3 SCENARIO=table_pressure SEED=1000 \\
            python3 scripts/ryu_manager.py controller/ryu_app.py
    """
    strategy = os.environ.get("STRATEGY", "markov")
    tau = float(os.environ.get("TAU", "0.3"))
    scenario = os.environ.get("SCENARIO", "table_pressure")
    seed = int(os.environ.get("SEED", str(cfg.EVALUATION_SEEDS[0])))
    repetition_index = int(os.environ.get("REPETITION_INDEX", "0"))
    capacity = int(os.environ.get("CAPACITY", str(cfg.TABLE_PRESSURE_CAPACITY)))
    out_dir = os.environ.get("OUT_DIR", "logs")

    assert strategy in cfg.STRATEGIES, f"STRATEGY must be one of {cfg.STRATEGIES}, got {strategy!r}"
    assert scenario in cfg.LIVE_ONLY_SCENARIOS, (
        f"SCENARIO must be one of {cfg.LIVE_ONLY_SCENARIOS} (the rest are offline-replayed), got {scenario!r}"
    )

    return _Args(strategy, tau, scenario, seed, repetition_index, capacity, out_dir)


class ProactiveFlowInstallerApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.args = _parse_args()

        self.table = FlowTable(capacity=self.args.capacity, idle_timeout=cfg.IDLE_TIMEOUT_SEC)
        self.predictor: Predictor | None = self._load_predictor(self.args.strategy)
        self.history: list[str] = []
        self.flow_id = 0
        self.run_key_id = (
            f"{self.args.strategy}__tau{self.args.tau:.1f}__"
            f"{self.args.scenario}__seed{self.args.seed}"
        )
        self.t0 = time.time()

        out_path = os.path.join(self.args.out_dir, self.run_key_id + ".jsonl")
        self.writer = EventLogWriter(out_path)
        self.logger.info("Logging to %s", out_path)

        # datapath id -> port_no by mac, learned reactively (simple L2 learning
        # switch underneath the flow-installation policy, since something has
        # to actually forward packets for the experiment to be observable).
        self.mac_to_port: dict[int, dict[str, int]] = {}

    def _load_predictor(self, strategy: str) -> Predictor | None:
        if strategy == "reactive":
            return None
        if strategy == "markov":
            predictor = MarkovPredictor()
        elif strategy == "lstm":
            from predictors.lstm import LSTMPredictor
            predictor = LSTMPredictor()
        else:
            raise ValueError(strategy)

        # Train once, offline, on TRAIN_SEED traffic from the SAME generator
        # as the live scenario reuses (table_pressure == baseline shape).
        train_sigs, _ = generate_baseline(cfg.TRAIN_SEED, n_flows=4000)
        predictor.fit([s.key() for s in train_sigs])
        self.logger.info("Trained %s predictor on %d flows", strategy, len(train_sigs))
        return predictor

    # -- OpenFlow handlers ---------------------------------------------------

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        # table-miss entry (cookie 0): send unmatched packets to the controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, priority=0, match=match, actions=actions, idle_timeout=0, cookie=0)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        t_recv = time.time()
        msg = ev.msg
        datapath = msg.datapath
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        ip = pkt.get_protocol(ipv4.ipv4)
        u = pkt.get_protocol(udp.udp)
        if ip is None or u is None:
            return  # only the experiment's UDP flows matter here

        payload = pkt.protocols[-1] if pkt.protocols else b""
        parsed = parse_payload(payload) if isinstance(payload, (bytes, bytearray)) else None
        flow_id = parsed[0] if parsed else self.flow_id

        sig = FlowSignature(src_host=ip_to_host(ip.src), dst_host=ip_to_host(ip.dst),
                            dst_port=u.dst_port, src_port=u.src_port)
        physical_miss = (msg.cookie != RULE_COOKIE)
        self._handle_flow(datapath, in_port, msg.data, physical_miss, sig, flow_id, t_recv)

    # -- per-flow logic (mirrors scripts/replay.py) --------------------------
    #
    # Every rule we install ALSO copies matching packets to the controller, so
    # the controller observes EVERY arrival -- hits as well as misses. Without
    # that, a hit would be invisible (the switch forwards it silently) and the
    # log could not compute hit rate, coverage, or LRU recency. The packet still
    # takes the switch's fast path; the copy is extra. Latency is NOT measured
    # here: receiving hosts timestamp arrivals (scripts/host_agent.py) and
    # shared/live_merge.py joins those onto this log by flow_id afterwards.

    def _handle_flow(self, datapath, in_port, data, physical_miss: bool,
                     sig: FlowSignature, flow_id: int, t_recv: float):
        key = sig.key()
        t = t_recv - self.t0

        lm = self.table.match(key, t)  # refreshes logical recency on a logical hit
        reactive_eviction = None
        if physical_miss or not lm.hit:
            reactive_eviction = self.table.install(key, t)
            self._install_match(datapath, sig.src_host, sig.dst_host, sig.dst_port, reactive_eviction)
        if physical_miss:
            # The switch had no rule, so this packet is stuck at the controller: forward it.
            self._packet_out(datapath, in_port, sig.dst_host, data)

        # proactive prediction for what comes NEXT (Figure 1), using history incl. this arrival.
        # The arrival MUST be in the history before predicting: offline replay predicts flow k
        # from flows < k and installs BEFORE flow k arrives; live, the only moment we can act
        # is flow k's Packet-In, so the equivalent is to predict flow k+1 from flows <= k.
        # (Appending afterwards made every prediction target the flow that had just arrived.)
        self.history.append(key)
        predicted_sig, confidence, decision, installed_proactively = None, None, "no_rule", False
        proactive_eviction = None
        if self.predictor is not None:
            window = self.history[-cfg.INPUT_WINDOW:]
            pred = self.predictor.predict(window)
            predicted_sig, confidence = pred.signature, pred.confidence
            pd = decide(pred, self.args.tau)
            if pd.install_proactively:
                decision, installed_proactively = "proactive_install", True
                proactive_eviction = self.table.install(pd.signature, t)
                left, port = pd.signature.rsplit(":", 1)
                p_src, p_dst = left.split("->")
                self._install_match(datapath, p_src, p_dst, int(port), proactive_eviction)
        if physical_miss and decision == "no_rule":
            decision = "reactive_install"

        ev = FlowEvent(
            run_id=self.run_key_id, strategy=self.args.strategy, tau=self.args.tau,
            scenario=self.args.scenario, seed=self.args.seed,
            repetition_index=self.args.repetition_index,
            flow_id=flow_id, t_arrival=t, signature=key,
            predicted_signature=predicted_sig, confidence=confidence,
            decision=decision, installed_proactively=installed_proactively,
            match_type="miss" if physical_miss else "hit",
            controller_interaction=physical_miss,
            flow_setup_latency_ms=None,  # filled in later by shared/live_merge.py
            **eviction_fields(proactive_eviction, reactive_eviction),
            table_occupancy_after=lm.occupancy_after,
        )
        self.writer.write(ev)
        self.flow_id += 1

    # -- OpenFlow plumbing -----------------------------------------------

    def _add_flow(self, datapath, priority, match, actions, idle_timeout, cookie):
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match,
                                instructions=inst, idle_timeout=idle_timeout, cookie=cookie)
        datapath.send_msg(mod)

    @staticmethod
    def _rule_match(parser, src_host: str, dst_host: str, dst_port: int):
        # Traffic is UDP (ip_proto 17), so the match must use udp_dst, not tcp_dst.
        return parser.OFPMatch(eth_type=0x0800, ip_proto=17,
                               ipv4_src=host_to_ip(src_host), ipv4_dst=host_to_ip(dst_host),
                               udp_dst=dst_port)

    def _install_match(self, datapath, src_host, dst_host, dst_port, eviction):
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        if eviction is not None and eviction.caused_eviction:
            self._evict_rule(datapath, eviction.evicted_signature)
        match = self._rule_match(parser, src_host, dst_host, dst_port)
        actions = [parser.OFPActionOutput(host_to_port(dst_host)),
                   parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, priority=10, match=match, actions=actions,
                       idle_timeout=int(cfg.IDLE_TIMEOUT_SEC), cookie=RULE_COOKIE)

    def _evict_rule(self, datapath, key: str):
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        left, port = key.rsplit(":", 1)
        src, dst = left.split("->")
        match = self._rule_match(parser, src, dst, int(port))
        mod = parser.OFPFlowMod(datapath=datapath, command=ofproto.OFPFC_DELETE_STRICT,
                                priority=10, out_port=ofproto.OFPP_ANY,
                                out_group=ofproto.OFPG_ANY, match=match)
        datapath.send_msg(mod)

    def _packet_out(self, datapath, in_port, dst_host, data):
        ofproto, parser = datapath.ofproto, datapath.ofproto_parser
        actions = [parser.OFPActionOutput(host_to_port(dst_host))]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
