"""
Mininet topology: one Open vSwitch, config.N_HOSTS hosts, one remote Ryu
controller (Research Plan Section 2.2: "a single Open vSwitch instance
under one Ryu controller, connecting 16 emulated hosts").

Run on YOUR Linux machine (needs root + the openvswitch kernel module +
network namespaces -- none of which are available in this sandboxed
container). Typical install:

    sudo apt-get install mininet openvswitch-switch
    pip install ryu --break-system-packages    # or mininet's own venv

Usage:
    sudo python3 topo/mininet_topo.py
  (starts the topology and drops you into the Mininet CLI; run the Ryu app
   in a separate terminal first -- see controller/ryu_app.py's header)

Or import `build_net()` from a driver script that also launches traffic.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

from shared import config as cfg


def build_net(controller_ip: str = "127.0.0.1", controller_port: int = 6653) -> Mininet:
    net = Mininet(switch=OVSSwitch, controller=None, autoSetMacs=True)

    c0 = net.addController(
        "c0", controller=RemoteController, ip=controller_ip, port=controller_port,
    )
    s1 = net.addSwitch("s1", protocols="OpenFlow13")

    hosts = []
    for i in range(1, cfg.N_HOSTS + 1):
        h = net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
        net.addLink(h, s1)
        hosts.append(h)

    net.build()
    # Static ARP: hosts know each other's MACs up front, so no ARP broadcasts
    # (which the controller deliberately ignores) are needed before traffic flows.
    net.staticArp()
    c0.start()
    s1.start([c0])
    return net


if __name__ == "__main__":
    setLogLevel("info")
    net = build_net()
    print(f"Topology up: 1 switch, {cfg.N_HOSTS} hosts, controller at 127.0.0.1:6653")
    print("Start the Ryu app in another terminal BEFORE relying on flow installation:")
    print("  ~/ryu-venv/bin/python scripts/ryu_manager.py --ofp-tcp-listen-port 6653 controller/ryu_app.py")
    CLI(net)
    net.stop()
