#!/usr/bin/env bash
# One-time setup of everything Python-side, without long commands to paste.
# Safe to run again: it skips what is already installed.
#
#   bash scripts/setup_ryu_venv.sh        (NO sudo)
#
# Creates ~/ryu-venv with Ryu 4.34 + eventlet 0.33.3 + PyTorch (CPU) for the
# controller, and adds PyTorch + NumPy to your normal python3 for offline runs.
# System packages (mininet, openvswitch-switch) are installed separately with apt.

set -e
if [ "$(id -u)" -eq 0 ]; then
    echo "Run this WITHOUT sudo:  bash scripts/setup_ryu_venv.sh"
    exit 1
fi

VENV="$HOME/ryu-venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
TORCH_URL="https://download.pytorch.org/whl/cpu"

echo "== 1/4 Ryu environment at $VENV"
if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV"
fi

echo "== 2/4 Ryu and eventlet"
if "$PY" -c "import ryu" 2>/dev/null; then
    echo "   already installed"
else
    "$PIP" install -q "setuptools==67.6.1" wheel     # Ryu 4.34 only installs with this
    "$PIP" install -q "eventlet==0.33.3" "ryu==4.34"
fi
"$PIP" install -q "setuptools>=77.0.3,<80"            # what PyTorch wants, once Ryu is in

echo "== 3/4 PyTorch and NumPy in the Ryu environment"
"$PY" -c "import torch" 2>/dev/null || "$PIP" install -q torch --index-url "$TORCH_URL"
"$PY" -c "import numpy" 2>/dev/null || "$PIP" install -q numpy

echo "== 4/4 PyTorch and NumPy in your normal python3 (offline runs)"
python3 -c "import torch" 2>/dev/null || pip3 install --user -q torch --index-url "$TORCH_URL"
python3 -c "import numpy" 2>/dev/null || pip3 install --user -q numpy

echo
echo "== Checking"
"$PY" - <<'EOF'
import collections, collections.abc as a
for n in ("Mapping", "MutableMapping", "Sequence", "MutableSequence", "Set", "MutableSet"):
    if not hasattr(collections, n):
        setattr(collections, n, getattr(a, n))
import eventlet.wsgi as w
if not hasattr(w, "ALREADY_HANDLED"):
    w.ALREADY_HANDLED = object()
import ryu.cmd.manager, torch, numpy
print("   Ryu environment OK   (torch", torch.__version__ + ")")
EOF
python3 -c "import torch, numpy; print('   python3 OK           (torch', torch.__version__ + ')')"
echo
echo "Setup complete."
