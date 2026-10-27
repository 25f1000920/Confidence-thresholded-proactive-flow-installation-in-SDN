"""
Drives controller/ryu_app.py's real logic with a FAKE Ryu and a FAKE switch,
because real Ryu/OVS can't run in CI or this sandbox. It checks WHAT the
controller would send (rule matches, evictions, packet-outs) and what it logs.
It does NOT prove the Ryu API calls are accepted by a real switch -- only a real
Mininet run does that.
"""
import os, sys, types, logging, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- fake ryu ----------------------------------------------------------------
class Rec:
    """Records constructor kwargs so tests can inspect what was 'sent'."""
    def __init__(self, *args, **kw):
        self.args, self.kw = args, kw


class FakeProto:
    OFP_VERSION = 4
    OFPP_CONTROLLER, OFPCML_NO_BUFFER, OFPIT_APPLY_ACTIONS = 0xfffffffd, 0xffff, 4
    OFPFC_DELETE_STRICT, OFPP_ANY, OFPG_ANY, OFP_NO_BUFFER = 4, 0xffffffff, 0xffffffff, 0xffffffff


class FakeParser:
    class OFPMatch(Rec): pass
    class OFPActionOutput(Rec): pass
    class OFPInstructionActions(Rec): pass
    class OFPFlowMod(Rec): pass
    class OFPPacketOut(Rec): pass


class FakeDatapath:
    ofproto, ofproto_parser = FakeProto, FakeParser
    def __init__(self): self.sent = []
    def send_msg(self, m): self.sent.append(m)


def _install_fake_ryu():
    def mod(name, **attrs):
        m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
    class RyuApp:
        def __init__(self, *a, **k): self.logger = logging.getLogger("fake")
    mod("ryu"); mod("ryu.base"); mod("ryu.base.app_manager", RyuApp=RyuApp)
    mod("ryu.controller"); mod("ryu.controller.ofp_event", EventOFPSwitchFeatures=1, EventOFPPacketIn=2)
    mod("ryu.controller.handler", MAIN_DISPATCHER=1, CONFIG_DISPATCHER=2, set_ev_cls=lambda *a, **k: (lambda f: f))
    mod("ryu.ofproto"); mod("ryu.ofproto.ofproto_v1_3", OFP_VERSION=4)
    mod("ryu.lib"); mod("ryu.lib.packet")
    mod("ryu.lib.packet.packet"); mod("ryu.lib.packet.ethernet"); mod("ryu.lib.packet.ipv4")
    mod("ryu.lib.packet.tcp"); mod("ryu.lib.packet.udp")
    sys.modules["ryu.lib.packet"].__dict__.update(
        packet=sys.modules["ryu.lib.packet.packet"], ethernet=sys.modules["ryu.lib.packet.ethernet"],
        ipv4=sys.modules["ryu.lib.packet.ipv4"], tcp=sys.modules["ryu.lib.packet.tcp"],
        udp=sys.modules["ryu.lib.packet.udp"])


def _make_app(strategy, tau, capacity):
    _install_fake_ryu()
    os.environ.update(STRATEGY=strategy, TAU=str(tau), SCENARIO="table_pressure", SEED="7",
                      CAPACITY=str(capacity), OUT_DIR=tempfile.mkdtemp())
    import importlib.util
    spec = importlib.util.spec_from_file_location("ryu_app_under_test", "controller/ryu_app.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, m.ProactiveFlowInstallerApp()


def _flows(dp, kind):
    return [x for x in dp.sent if isinstance(x, kind)]


def test_first_miss_installs_udp_rule_and_forwards():
    m, app = _make_app("reactive", 1.0, 5)
    dp = FakeDatapath()
    sig = m.FlowSignature("h3", "h5", 80, 40000)
    app._handle_flow(dp, in_port=3, data=b"PKT", physical_miss=True, sig=sig, flow_id=0, t_recv=app.t0 + 1)

    mods, outs = _flows(dp, FakeParser.OFPFlowMod), _flows(dp, FakeParser.OFPPacketOut)
    assert len(mods) == 1 and len(outs) == 1
    match = mods[0].kw["match"].kw
    assert match == dict(eth_type=0x0800, ip_proto=17, ipv4_src="10.0.0.3", ipv4_dst="10.0.0.5", udp_dst=80), match
    assert mods[0].kw["cookie"] == m.RULE_COOKIE
    inst_actions = mods[0].kw["instructions"][0].args[1]
    ports = [a.args[0] for a in inst_actions]
    assert ports == [5, FakeProto.OFPP_CONTROLLER], ports  # forward + copy to controller
    assert outs[0].kw["data"] == b"PKT" and outs[0].kw["actions"][0].args[0] == 5
    print("test_first_miss_installs_udp_rule_and_forwards: PASS")


def test_hit_is_logged_but_not_forwarded_again():
    m, app = _make_app("reactive", 1.0, 5)
    dp = FakeDatapath()
    sig = m.FlowSignature("h3", "h5", 80, 40000)
    app._handle_flow(dp, 3, b"PKT", True, sig, 0, app.t0 + 1)
    dp.sent.clear()
    app._handle_flow(dp, 3, b"PKT", False, sig, 1, app.t0 + 2)   # copy of a packet the switch already forwarded
    assert dp.sent == [], "a hit must not trigger another flow-mod or packet-out"
    app.writer.close()
    from shared.event_log import read_events_list
    evs = read_events_list(os.path.join(os.environ["OUT_DIR"], app.run_key_id + ".jsonl"))
    assert [e.match_type for e in evs] == ["miss", "hit"]
    assert [e.controller_interaction for e in evs] == [True, False]
    assert all(e.signature == "h3->h5:80" for e in evs)          # hN naming, not IPs
    print("test_hit_is_logged_but_not_forwarded_again: PASS")


def test_capacity_eviction_sends_delete_strict():
    m, app = _make_app("reactive", 1.0, 2)
    dp = FakeDatapath()
    for i, dst in enumerate(["h5", "h6", "h7"]):      # 3 distinct flows, capacity 2
        app._handle_flow(dp, 3, b"P", True, m.FlowSignature("h3", dst, 80, 1), i, app.t0 + i)
    dels = [x for x in _flows(dp, FakeParser.OFPFlowMod) if x.kw.get("command") == FakeProto.OFPFC_DELETE_STRICT]
    assert len(dels) == 1
    assert dels[0].kw["match"].kw["ipv4_dst"] == "10.0.0.5"      # LRU victim = first flow
    print("test_capacity_eviction_sends_delete_strict: PASS")


def test_markov_proactive_install_uses_host_names_and_valid_ips():
    m, app = _make_app("markov", 0.0, 50)
    dp = FakeDatapath()
    # Walk the REAL baseline world so the predictor has something to predict from.
    from traffic.generators import generate_baseline
    sigs, _ = generate_baseline(1234, 60)
    n_proactive = 0
    for i, s in enumerate(sigs):
        sig = m.FlowSignature(s.src_host, s.dst_host, s.dst_port, s.src_port)
        app._handle_flow(dp, 1, b"P", True, sig, i, app.t0 + i * 0.01)
    for fm in _flows(dp, FakeParser.OFPFlowMod):
        mt = fm.kw["match"].kw
        if "ipv4_src" in mt:
            assert mt["ipv4_src"].startswith("10.0.0.") and mt["ipv4_dst"].startswith("10.0.0."), mt
    app.writer.close()
    from shared.event_log import read_events_list
    evs = read_events_list(os.path.join(os.environ["OUT_DIR"], app.run_key_id + ".jsonl"))
    n_proactive = sum(e.installed_proactively for e in evs)
    assert n_proactive > 0, "markov at tau=0.0 on baseline traffic must install proactively"
    print(f"test_markov_proactive_install_uses_host_names_and_valid_ips: PASS ({n_proactive}/{len(evs)} proactive)")


def test_proactive_prediction_targets_the_next_flow_not_the_current_one():
    """Regression: the arrival must be appended to history BEFORE predicting, otherwise
    every proactive install is for the flow that has just arrived (useless)."""
    m, app = _make_app("markov", 0.0, 500)
    dp = FakeDatapath()
    from traffic.generators import generate_baseline
    sigs, _ = generate_baseline(1234, 400)
    for i, s in enumerate(sigs):
        sig = m.FlowSignature(s.src_host, s.dst_host, s.dst_port, s.src_port)
        app._handle_flow(dp, 1, b"P", True, sig, i, app.t0 + i * 0.01)
    app.writer.close()
    from shared.event_log import read_events_list
    evs = read_events_list(os.path.join(os.environ["OUT_DIR"], app.run_key_id + ".jsonl"))
    pro = [(i, e) for i, e in enumerate(evs[:-1]) if e.installed_proactively]
    next_ok = sum(e.predicted_signature == evs[i + 1].signature for i, e in pro)
    current = sum(e.predicted_signature == e.signature for i, e in pro)
    assert next_ok > 0.3 * len(pro), (next_ok, len(pro))
    assert current < 0.1 * len(pro), (current, len(pro))
    print(f"test_proactive_prediction_targets_the_next_flow_not_the_current_one: PASS "
          f"({next_ok}/{len(pro)} predicted the next flow)")


def test_both_evictions_on_one_flow_are_counted():
    """A miss that triggers a reactive install AND a proactive install, with a full
    table, evicts two rules -- both must be logged (n_evictions == 2) and both
    must be deleted from the switch."""
    m, app = _make_app("markov", 0.0, 2)
    dp = FakeDatapath()
    from predictors.base import Prediction
    app.predictor.predict = lambda window: Prediction("h9->h10:80", 1.0)
    for i, dst in enumerate(["h5", "h6"]):                       # fill the table (capacity 2)
        app.table.install(f"h3->{dst}:80", app.t0 + i - app.t0)
    dp.sent.clear()
    app._handle_flow(dp, 3, b"P", True, m.FlowSignature("h3", "h7", 80, 1), 0, app.t0 + 3)
    dels = [x for x in _flows(dp, FakeParser.OFPFlowMod) if x.kw.get("command") == FakeProto.OFPFC_DELETE_STRICT]
    assert len(dels) == 2, len(dels)
    app.writer.close()
    from shared.event_log import read_events_list
    ev = read_events_list(os.path.join(os.environ["OUT_DIR"], app.run_key_id + ".jsonl"))[0]
    assert ev.n_evictions == 2 and ev.caused_eviction, ev
    assert ev.evicted_by == "proactive" and ev.reactive_evicted_signature is not None, ev
    print("test_both_evictions_on_one_flow_are_counted: PASS")


if __name__ == "__main__":
    test_first_miss_installs_udp_rule_and_forwards()
    test_hit_is_logged_but_not_forwarded_again()
    test_capacity_eviction_sends_delete_strict()
    test_markov_proactive_install_uses_host_names_and_valid_ips()
    test_proactive_prediction_targets_the_next_flow_not_the_current_one()
    test_both_evictions_on_one_flow_are_counted()
    print("\nAll controller-logic tests passed.")
