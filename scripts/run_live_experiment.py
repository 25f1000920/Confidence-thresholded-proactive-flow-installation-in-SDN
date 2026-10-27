"""
End-to-end live-run driver. Run on YOUR Mininet/Ryu machine (root required),
AFTER starting the Ryu app in a separate terminal:

    # terminal 1  (the python that has ryu, e.g. ~/ryu-venv/bin/python; NO sudo needed)
    STRATEGY=markov TAU=0.3 SCENARIO=table_pressure SEED=1000 OUT_DIR=logs/live \\
        ~/ryu-venv/bin/python scripts/ryu_manager.py --ofp-tcp-listen-port 6653 controller/ryu_app.py

    # terminal 2  (system python, as root)
    sudo python3 scripts/run_live_experiment.py \\
        --scenario table_pressure --seed 1000 --n-flows 1500

Settings go to the controller as ENVIRONMENT VARIABLES: ryu-manager rejects
unknown flags such as --strategy/--tau with "unrecognized arguments".
The two processes must be started with MATCHING scenario/seed so the
resulting event log corresponds to a single, well-defined point in the
experimental matrix -- the run_id embedded in the controller's log file
name is the source of truth for what a given log actually contains.

If a previous run crashed, clean up first:  sudo mn -c

Not runnable in this sandbox (needs Mininet's root/netns access).
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mininet.log import setLogLevel

from shared import config as cfg
from topo.mininet_topo import build_net
from scripts.live_traffic_player import play_traffic, check_timing_accuracy, LiveSender

AGENT_SCRIPT = os.path.join(os.path.dirname(__file__), "host_agent.py")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=cfg.LIVE_ONLY_SCENARIOS, default="table_pressure",
                    help="Live-only scenarios only -- the other 4 are offline-replayed, not played live.")
    p.add_argument("--seed", type=int, default=cfg.EVALUATION_SEEDS[0])
    p.add_argument("--n-flows", type=int, default=1500)
    p.add_argument("--rate", type=float, default=cfg.FLOW_RATE_FLOWS_PER_SEC)
    p.add_argument("--speed-factor", type=float, default=1.0,
                    help="> 1.0 compresses elapsed time for a fast pilot; keep at 1.0 for real runs.")
    p.add_argument("--agent-startup-wait", type=float, default=1.0,
                    help="Seconds to let host_agent.py processes bind their sockets before sending traffic.")
    p.add_argument("--sock-dir", default="/tmp")
    p.add_argument("--recv-out", default="logs/live/recv_latest.csv",
                    help="Where to write the combined receive-side timestamps (for latency).")
    p.add_argument("--drain-wait", type=float, default=1.5,
                    help="Seconds to wait after the last send for in-flight packets to arrive.")
    return p.parse_args()


def start_agents(net, sock_dir: str):
    """Launches scripts/host_agent.py inside every host's network namespace."""
    procs = []
    for i in range(1, cfg.N_HOSTS + 1):
        h = net.get(f"h{i}")
        sock_path = os.path.join(sock_dir, f"agent_h{i}.sock")
        recv_path = os.path.join(sock_dir, f"recv_h{i}.csv")
        # stderr goes to a file: with the default PIPE nobody reads it, so an agent
        # that died on startup (port in use, import error) would fail silently.
        err = open(os.path.join(sock_dir, f"agent_h{i}.err"), "w")
        proc = h.popen([sys.executable, AGENT_SCRIPT, sock_path, recv_path], stderr=err)
        procs.append(proc)
    return procs


def check_agents(procs, sock_dir: str) -> None:
    dead = [i for i, p in enumerate(procs, 1) if p.poll() is not None]
    if dead:
        first = os.path.join(sock_dir, f"agent_h{dead[0]}.err")
        msg = open(first).read().strip() if os.path.exists(first) else ""
        raise SystemExit(f"ERROR: host agents died on startup: {['h%d' % i for i in dead]}\n"
                         f"--- {first} ---\n{msg}")


def stop_agents(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=2)
        except Exception:
            p.kill()


def collect_recv(sock_dir: str, out_path: str) -> int:
    """Concatenate every host's receive log into one CSV. Returns row count."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w") as out:
        for i in range(1, cfg.N_HOSTS + 1):
            p = os.path.join(sock_dir, f"recv_h{i}.csv")
            if os.path.exists(p):
                for line in open(p):
                    out.write(line)
                    n += 1
    return n


def main():
    args = _parse_args()
    setLogLevel("info")

    net = build_net()
    print(f"Mininet up: 1 switch, {cfg.N_HOSTS} hosts. Waiting for the switch to connect to the controller...")
    try:
        connected = net.waitConnected(timeout=20)
    except AttributeError:  # very old Mininet: no waitConnected
        time.sleep(5)
        connected = True
    if not connected:
        print("ERROR: the switch never connected to a controller on 127.0.0.1:6653. "
              "Is the Ryu controller running in another terminal?", file=sys.stderr)
        net.stop()
        sys.exit(1)
    time.sleep(1.0)  # let the controller install the table-miss rule
    print("Switch connected. Starting host agents...")
    agent_procs = start_agents(net, args.sock_dir)
    time.sleep(args.agent_startup_wait)
    try:
        check_agents(agent_procs, args.sock_dir)
    except SystemExit:
        stop_agents(agent_procs)
        net.stop()
        raise

    sender = LiveSender(sock_dir=args.sock_dir)
    print(f"Playing '{args.scenario}' traffic: seed={args.seed}, n_flows={args.n_flows}, "
          f"rate={args.rate}/s, speed_factor={args.speed_factor}")
    print("(Make sure the controller is already running with matching "
          f"SCENARIO={args.scenario} SEED={args.seed}, or its own log will not line up.)")

    try:
        log = play_traffic(
            args.scenario, args.seed, args.n_flows, sender.send,
            rate=args.rate, speed_factor=args.speed_factor,
        )
        check_timing_accuracy(log)
        print(f"Done: {len(log)} flows dispatched, timing drift within tolerance.")
    except RuntimeError as e:
        print(f"WARNING: {e}", file=sys.stderr)
        print("Data from this run should be treated as suspect -- see the error above.", file=sys.stderr)
    finally:
        time.sleep(args.drain_wait)
        n_recv = collect_recv(args.sock_dir, args.recv_out)
        print(f"Collected {n_recv} receive records -> {args.recv_out}")
        stop_agents(agent_procs)
        net.stop()


if __name__ == "__main__":
    main()
