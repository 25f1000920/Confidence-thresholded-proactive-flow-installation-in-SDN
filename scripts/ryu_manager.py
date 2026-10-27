"""
Drop-in replacement for the `ryu-manager` console-script command.

WHY THIS FILE EXISTS:
Ryu depends on eventlet, which in turn depends on an old, pinned version
of dnspython (`dnspython<2.0.0`, per eventlet's own requirements). That
old dnspython release still writes `class NameDict(collections.MutableMapping)`
directly against `collections` instead of `collections.abc` -- Python 3.10
removed that alias, so this crashes with:

    AttributeError: module 'collections' has no attribute 'MutableMapping'

This happens INSIDE `ryu-manager`'s own startup sequence (ryu.cmd.manager
-> ryu.lib.hub -> eventlet -> dnspython), before your app file
(controller/ryu_app.py) is ever loaded -- so the compatibility shim
already added at the top of controller/ryu_app.py is too late to help
here; by the time Python reaches your file, the crash has already
happened inside the `ryu-manager` script itself.

This launcher applies the exact same shim, but as the VERY FIRST thing
that runs, before `ryu.cmd.manager` (and therefore eventlet and
dnspython) are imported at all -- so the broken collections lookup
never happens in the first place.

SECOND, SEPARATE BUG patched below: Ryu 4.34's own `ryu/app/wsgi.py` does
`from eventlet.wsgi import ALREADY_HANDLED` at class-definition time.
eventlet removed that exact name starting at version 0.30.3 (replaced by
`WSGI_LOCAL.already_handled`) -- a breaking change eventlet shipped in a
PATCH release. This is an open, unfixed bug in Ryu itself (see
https://github.com/faucetsdn/ryu/issues/138): there is no single eventlet
version where both this AND the collections/TimeoutError Python 3.10
fixes above are simultaneously satisfied, because any eventlet new enough
to have those fixes is also new enough to have removed ALREADY_HANDLED.

This project never uses Ryu's REST/WSGI app support (ryu.app.wsgi) --
our controller/ryu_app.py only uses the core OpenFlow event API -- so a
harmless placeholder value is enough to satisfy the import and let
ryu-manager finish starting up; it is never exercised for real.

USAGE: Identical to ryu-manager, just swap the command:
    OLD:  ryu-manager --ofp-tcp-listen-port 6653 controller/ryu_app.py --strategy markov --tau 0.3 ...
    NEW:  python3 scripts/ryu_manager.py --ofp-tcp-listen-port 6653 controller/ryu_app.py --strategy markov --tau 0.3 ...

Every other script in this repo that spawns `ryu-manager` as a
subprocess (scripts/run_live_matrix.py) should be pointed at this file
instead if you hit the same crash -- see the note in that file.
"""
import collections
import collections.abc as _collections_abc

for _name in ("Mapping", "MutableMapping", "Sequence", "MutableSequence", "Set", "MutableSet"):
    if not hasattr(collections, _name) and hasattr(_collections_abc, _name):
        setattr(collections, _name, getattr(_collections_abc, _name))

import eventlet.wsgi as _eventlet_wsgi  # noqa: E402  (must come before ryu.cmd.manager)

if not hasattr(_eventlet_wsgi, "ALREADY_HANDLED"):
    _eventlet_wsgi.ALREADY_HANDLED = object()  # unused placeholder; see docstring above

import sys  # noqa: E402

from ryu.cmd.manager import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
