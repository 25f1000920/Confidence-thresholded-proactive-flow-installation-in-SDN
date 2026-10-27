"""
Runs the FULL live portion of the experiment matrix (table_pressure only):

    strategies {reactive, markov, lstm} x TAU_GRID x EVALUATION_SEEDS

for each combo, spawns `ryu-manager controller/ryu_app.py` with the right
args, waits for it to start listening, plays the matching traffic via
run_live_experiment's machinery, then tears the controller down and moves
to the next combo. Reactive-only is tau-independent and is run once per
seed (tau=1.0 convention), same as the offline matrix.

MUST be run as root (Mininet requirement), on your own Mininet/Ryu
machine -- not runnable in this sandbox.

Usage (TWO interpreters are involved -- see README "Live runs: setup"):
    # this script + Mininet run under the SYSTEM python (sudo python3);
    # the controller runs under whichever python has ryu (RYU_PYTHON, after sudo):
    sudo RYU_PYTHON=$HOME/ryu-venv/bin/python python3 scripts/run_live_matrix.py --dry-run
    sudo RYU_PYTHON=$HOME/ryu-venv/bin/python python3 scripts/run_live_matrix.py --strategies markov --taus 0.0 0.3 0.6 0.9
    sudo RYU_PYTHON=$HOME/ryu-venv/bin/python python3 scripts/run_live_matrix.py

A preflight check runs first and stops with the exact reason if ryu, mininet,
OVS or (for lstm) torch cannot be found. Each run's controller output is kept
in logs/live/<run_id>.ryu.log.

Output: logs/live/<strategy>__tau<tau>__table_pressure__seed<seed>.jsonl
        (written by controller/ryu_app.py itself, via its --out-dir)

NOTE ON RUNTIME: an 11-value tau grid x 2 predictor strategies x
N_REPETITIONS seeds, plus one reactive run per seed, is
(11*2 + 1) * N_REPETITIONS live runs. At a few seconds of controller
startup overhead each, budget real wall-clock time for this -- it is the
single most expensive part of the whole experiment matrix. Consider
starting with a coarser tau subset (--taus) or fewer seeds to check
everything works before committing to the full grid.
"""
from __future__ import annotations
import argparse
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import config as cfg
from shared.live_merge import merge_latency

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RYU_APP = os.path.join(REPO_ROOT, "controller", "ryu_app.py")
LIVE_DRIVER = os.path.join(REPO_ROOT, "scripts", "run_live_experiment.py")
# Use our own launcher (scripts/ryu_manager.py) instead of the bare
# `ryu-manager` console script: on Python 3.10+, the real `ryu-manager`
# entry point can crash during its OWN startup (ryu -> eventlet -> an old
# pinned dnspython using collections.MutableMapping) before it ever loads
# RYU_APP, so the compatibility shim at the top of controller/ryu_app.py
# is too late to help. scripts/ryu_manager.py applies the same shim
# before any of that gets imported at all. See that file's docstring.
RYU_MANAGER_LAUNCHER = os.path.join(REPO_ROOT, "scripts", "ryu_manager.py")


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, timeout: float, proc: subprocess.Popen) -> bool:
    """True once `port` accepts connections; False on timeout OR if `proc` (the controller) dies first."""
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        if port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def tail(path: str, n: int = 25) -> str:
    try:
        with open(path, errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no controller output captured)"


def preflight(ryu_python: str, strategies) -> None:
    """Fail fast, with the real reason, instead of 23 x N 'did not start listening' lines."""
    problems = []
    if os.geteuid() != 0:
        problems.append("Not running as root. Mininet needs root: run this script with sudo.")
    try:
        import mininet  # noqa: F401
    except ImportError:
        problems.append(
            f"'{sys.executable}' cannot import mininet. Run this script with the SYSTEM python "
            "(sudo python3 ...), not a venv python, and install Mininet with: sudo apt install mininet")
    for tool in ("ovs-vsctl", "mnexec"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            problems.append(f"'{tool}' not found. Install with: sudo apt install mininet openvswitch-switch")

    check = ("import collections, collections.abc as a\n"
             "[setattr(collections, n, getattr(a, n)) for n in ('Mapping','MutableMapping','Sequence',"
             "'MutableSequence','Set','MutableSet') if not hasattr(collections, n)]\n"
             "import eventlet.wsgi as w\n"
             "if not hasattr(w, 'ALREADY_HANDLED'): w.ALREADY_HANDLED = object()\n"
             "import ryu.cmd.manager, ryu.ofproto.ofproto_v1_3\n")
    if "lstm" in strategies:
        check += "import torch\n"
    try:
        r = subprocess.run([ryu_python, "-c", check], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            last = (r.stderr.strip().splitlines() or ["(no output)"])[-1]
            problems.append(
                f"The controller interpreter '{ryu_python}' cannot start Ryu"
                f"{' + torch' if 'lstm' in strategies else ''}:\n      {last}\n"
                "    sudo does NOT see packages you pip-installed as your own user or inside a venv.\n"
                "    Point RYU_PYTHON at the interpreter that has ryu, and put it AFTER sudo:\n"
                "      sudo RYU_PYTHON=$HOME/ryu-venv/bin/python python3 scripts/run_live_matrix.py --dry-run")
    except (OSError, subprocess.TimeoutExpired) as e:
        problems.append(f"Could not run the controller interpreter '{ryu_python}': {e}")

    if port_open("127.0.0.1", 6653):
        problems.append(
            "Something is already listening on port 6653 (an old ryu-manager?). Stop it first:\n"
            "      sudo pkill -f ryu_manager.py ; sudo pkill -f ryu-manager")

    if problems:
        print("PREFLIGHT FAILED -- nothing was run:\n")
        for i, pr in enumerate(problems, 1):
            print(f" {i}. {pr}\n")
        sys.exit(2)
    print(f"Preflight OK. Driver python: {sys.executable} | controller python: {ryu_python}\n")


def cleanup_mininet() -> None:
    """Remove leftovers of a crashed run (stale s1 bridge / veth pairs make the NEXT run fail
    with 'RTNETLINK answers: File exists')."""
    subprocess.run(["mn", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_one_combo(strategy: str, tau: float, seed: int, n_flows: int, capacity: int,
                   rate: float, out_dir: str, settle_time: float = 2.0,
                   startup_timeout: float = 20.0) -> bool:
    run_id = f"{strategy}__tau{tau:.1f}__table_pressure__seed{seed}"
    print(f"\n=== {run_id} ===")

    # controller/ryu_app.py reads its settings from environment variables,
    # not CLI flags -- ryu-manager's own argument parser rejects any flag
    # it doesn't already know about, so passing --strategy/--tau/etc.
    # directly on the command line fails with "unrecognized arguments".
    # See controller/ryu_app.py::_parse_args for the full explanation.
    # The controller needs the venv python (has ryu); the Mininet driver needs the
    # system python (has mininet) -- i.e. whatever interpreter is running THIS script.
    ryu_python = os.environ.get("RYU_PYTHON", sys.executable)
    ryu_cmd = [ryu_python, RYU_MANAGER_LAUNCHER, "--ofp-tcp-listen-port", "6653", RYU_APP]
    run_env = os.environ.copy()
    run_env.update({
        "STRATEGY": strategy, "TAU": str(tau), "SCENARIO": "table_pressure",
        "SEED": str(seed), "CAPACITY": str(capacity), "OUT_DIR": out_dir,
    })
    events_path = os.path.join(REPO_ROOT, out_dir, run_id + ".jsonl")
    recv_out = os.path.join(REPO_ROOT, out_dir, run_id + ".recv.csv")
    if os.path.exists(events_path):
        os.remove(events_path)
    # Controller output goes to a per-run file, so a crash on startup is visible
    # afterwards instead of scrolling past (or being mistaken for a slow start).
    ryu_log_path = os.path.join(REPO_ROOT, out_dir, run_id + ".ryu.log")
    os.makedirs(os.path.dirname(ryu_log_path), exist_ok=True)
    ryu_log = open(ryu_log_path, "w")
    ryu_proc = subprocess.Popen(ryu_cmd, cwd=REPO_ROOT, env=run_env,
                                stdout=ryu_log, stderr=subprocess.STDOUT)

    # The controller trains its predictor BEFORE it starts listening, so the LSTM
    # arm needs far longer than Markov/reactive to come up.
    ok = wait_for_port("127.0.0.1", 6653, timeout=startup_timeout, proc=ryu_proc)
    if not ok:
        if ryu_proc.poll() is not None:
            print(f"  FAILED: the controller CRASHED on startup (exit code {ryu_proc.returncode}) "
                  f"-- skipping {run_id}")
        else:
            print(f"  FAILED: the controller did not start listening on 6653 within "
                  f"{startup_timeout:.0f}s -- skipping {run_id}")
            ryu_proc.terminate()
        ryu_log.close()
        print(f"  ---- last lines of {ryu_log_path} ----")
        print("  " + tail(ryu_log_path).replace("\n", "\n  "))
        return False
    time.sleep(settle_time)  # let the app finish its own setup (predictor training, etc.)

    player_cmd = [
        sys.executable, LIVE_DRIVER,
        "--scenario", "table_pressure", "--seed", str(seed),
        "--n-flows", str(n_flows), "--rate", str(rate),
        "--recv-out", recv_out,
    ]
    result = subprocess.run(player_cmd, cwd=REPO_ROOT)

    ryu_proc.terminate()
    try:
        ryu_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        ryu_proc.kill()
        ryu_proc.wait()
    ryu_log.close()

    if result.returncode != 0:
        print(f"  WARNING: traffic player exited with code {result.returncode} for {run_id}")
        cleanup_mininet()  # a crashed driver leaves s1/veths behind, which would break every later combo
        return False
    if not os.path.exists(events_path) or not os.path.exists(recv_out):
        print(f"  FAILED: missing event log or receive log for {run_id}")
        return False
    stats = merge_latency(events_path, recv_out)
    print(f"  merged: {stats}")
    if stats["n_events"] == 0 or stats["n_with_latency"] == 0:
        print(f"  FAILED: no usable data for {run_id} (controller saw no packets, or none arrived)")
        print(f"  ---- last lines of {ryu_log_path} ----")
        print("  " + tail(ryu_log_path).replace("\n", "\n  "))
        print(f"  Host-agent errors, if any, are in /tmp/agent_h*.err")
        return False
    print(f"  OK: {run_id}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+", default=cfg.STRATEGIES, choices=cfg.STRATEGIES)
    p.add_argument("--taus", nargs="+", type=float, default=cfg.TAU_GRID)
    p.add_argument("--seeds", nargs="+", type=int, default=cfg.EVALUATION_SEEDS)
    p.add_argument("--n-flows", type=int, default=1500)
    p.add_argument("--capacity", type=int, default=cfg.TABLE_PRESSURE_CAPACITY)
    p.add_argument("--rate", type=float, default=cfg.FLOW_RATE_FLOWS_PER_SEC)
    p.add_argument("--out-dir", default="logs/live")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--startup-timeout", type=float, default=None,
                   help="Seconds to wait for the controller to listen on 6653 "
                        "(default: 30 for reactive/markov, 180 for lstm, which trains on startup).")
    args = p.parse_args()

    strategies, taus, seeds, n_flows = args.strategies, args.taus, args.seeds, args.n_flows
    if args.dry_run:
        strategies = ["markov"]
        taus = [0.3]
        seeds = seeds[:1]
        n_flows = 200
        print("--dry-run: single combo only, to check the ryu-manager/traffic-player wiring.\n")

    os.makedirs(os.path.join(REPO_ROOT, args.out_dir), exist_ok=True)
    preflight(os.environ.get("RYU_PYTHON", sys.executable), strategies)
    cleanup_mininet()
    results = []

    def timeout_for(strategy: str) -> float:
        if args.startup_timeout is not None:
            return args.startup_timeout
        return 180.0 if strategy == "lstm" else 30.0

    for seed in seeds:
        if "reactive" in strategies:
            ok = run_one_combo("reactive", 1.0, seed, n_flows, args.capacity, args.rate, args.out_dir,
                               startup_timeout=timeout_for("reactive"))
            results.append(("reactive", 1.0, seed, ok))

        for strategy in strategies:
            if strategy == "reactive":
                continue
            for tau in taus:
                ok = run_one_combo(strategy, tau, seed, n_flows, args.capacity, args.rate, args.out_dir,
                                   startup_timeout=timeout_for(strategy))
                results.append((strategy, tau, seed, ok))

    n_ok = sum(1 for *_ , ok in results if ok)
    print(f"\n{n_ok}/{len(results)} live combos completed successfully. Logs under {args.out_dir}/")
    failed = [(s, t, sd) for s, t, sd, ok in results if not ok]
    if failed:
        print("Failed combos (re-run these individually before trusting the matrix as complete):")
        for s, t, sd in failed:
            print(f"  strategy={s} tau={t} seed={sd}")


if __name__ == "__main__":
    main()
