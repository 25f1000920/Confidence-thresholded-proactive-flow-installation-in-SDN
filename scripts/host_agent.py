"""
Per-host traffic agent: SENDS flows on command and RECEIVES them, timestamping
both ends so end-to-end flow-setup latency can be measured.

Runs INSIDE one Mininet host's network namespace (started by
scripts/run_live_experiment.py -- never run directly):
    python3 scripts/host_agent.py <control_socket_path> <recv_log_path>

Sending: listens on a UNIX datagram control socket for
"<dst_ip> <dst_port> <flow_id>" and fires one UDP packet whose payload is
"<flow_id>,<t_send>" (shared/live_utils.py).

Receiving: a background thread binds every service port and appends
"<flow_id>,<t_send>,<t_recv>" to <recv_log_path> for each packet that
arrives. All processes share one machine clock, so t_recv - t_send is a true
one-way latency: small when a rule is already installed in the switch (hit),
larger when the packet had to go switch -> controller -> switch (miss).
"""
import os
import select
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.live_utils import build_payload, parse_payload
from traffic.generators import SERVICE_PORTS


def receiver_loop(recv_log_path: str) -> None:
    socks = []
    for port in SERVICE_PORTS:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        socks.append(s)
    with open(recv_log_path, "w", buffering=1) as out:
        while True:
            ready, _, _ = select.select(socks, [], [])
            for s in ready:
                data, _addr = s.recvfrom(2048)
                t_recv = time.time()
                parsed = parse_payload(data)
                if parsed is not None:
                    out.write(f"{parsed[0]},{parsed[1]:.6f},{t_recv:.6f}\n")


def main(sock_path: str, recv_log_path: str) -> None:
    if os.path.exists(sock_path):
        os.remove(sock_path)
    control = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    control.bind(sock_path)
    os.chmod(sock_path, 0o666)

    threading.Thread(target=receiver_loop, args=(recv_log_path,), daemon=True).start()

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        try:
            data, _ = control.recvfrom(256)
            dst_ip, dst_port_s, flow_id_s = data.decode().strip().split()
            sender.sendto(build_payload(int(flow_id_s), time.time()), (dst_ip, int(dst_port_s)))
        except (ValueError, UnicodeDecodeError):
            continue  # malformed command; drop it and keep serving
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: host_agent.py <control_socket_path> <recv_log_path>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
