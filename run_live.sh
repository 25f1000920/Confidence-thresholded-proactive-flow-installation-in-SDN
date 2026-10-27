#!/usr/bin/env bash
# Runs the live sweep with the right Pythons, so the command stays short.
#
#   sudo bash run_live.sh --dry-run     # one quick combination
#   sudo bash run_live.sh               # the full 138-run sweep
#
# Any extra arguments are passed to scripts/run_live_matrix.py
# (e.g. --strategies markov --taus 0.3 --seeds 1000).

cd "$(dirname "$0")" || exit 1

if [ "$(id -u)" -ne 0 ]; then
    echo "Please run with sudo:  sudo bash run_live.sh $*"
    exit 1
fi

USER_NAME="${SUDO_USER:-root}"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
export RYU_PYTHON="${RYU_PYTHON:-$USER_HOME/ryu-venv/bin/python}"

if [ ! -x "$RYU_PYTHON" ]; then
    echo "Ryu environment not found at $RYU_PYTHON"
    echo "Set it up first (without sudo):  bash scripts/setup_ryu_venv.sh"
    exit 1
fi

mn -c > /dev/null 2>&1
/usr/bin/python3 scripts/run_live_matrix.py "$@"
status=$?

# Files written under sudo belong to root; hand them back to you.
if [ -d logs ] && [ "$USER_NAME" != "root" ]; then
    chown -R "$USER_NAME":"$USER_NAME" logs
fi
exit $status
