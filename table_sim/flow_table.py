"""
Flow-table simulator.

This is the single piece of logic shared by:
  - offline replay (baseline / cold_start / drift / unstructured), where it
    runs in pure Python against pre-logged predictions, and
  - the live Ryu app (controller/ryu_app.py), where it mirrors -- in
    software -- the artificially small capacity we impose on the switch
    (see controller/ryu_app.py docstring for why capacity is enforced in
    the controller rather than relying on switch hardware limits).

Because both paths call the exact same FlowTable class, "the implementation
is trusted on generated traffic only once its event log matches [the hand
trace] exactly" (Research Plan Section 2.3 Step 2) is a meaningful,
sufficient check: validate this file once, and both replay and live runs
inherit that guarantee. See tests/test_flow_table.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _Entry:
    installed_at: float
    last_used: float
    natural_expiry: float  # last_used + idle_timeout, recomputed on every hit


@dataclass
class InstallResult:
    installed: bool
    caused_eviction: bool = False
    evicted_signature: Optional[str] = None
    eviction_cause: Optional[str] = None  # "capacity_lru" | None
    # The evicted entry's own natural-expiry deadline (its last_used +
    # idle_timeout) at the moment it was evicted. This is the exact
    # deadline the harmful-eviction definition needs for condition (ii);
    # it is NOT derivable from the evicting flow's own timestamp.
    evicted_natural_expiry: Optional[float] = None
    occupancy_after: int = 0


@dataclass
class MatchResult:
    hit: bool
    occupancy_after: int = 0


class FlowTable:
    """
    Capacity-limited flow table.

    - `capacity`: max number of resident entries.
    - `idle_timeout`: OpenFlow-style idle timeout; an entry not matched for
      this long is naturally expired (call `expire(t)` before each new
      event so TTL expiry is applied in-order).
    - Eviction policy on overflow: LRU (evict the entry with the oldest
      `last_used`), which is the standard, simplest, and most defensible
      default for an OpenFlow-style table and is what the harmful-eviction
      definition assumes ("that rule would have been requested again before
      natural TTL expiry" only makes sense against a deterministic policy).
    """

    def __init__(self, capacity: int, idle_timeout: float):
        self.capacity = capacity
        self.idle_timeout = idle_timeout
        self._entries: dict[str, _Entry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, signature: str) -> bool:
        return signature in self._entries

    def expire(self, t: float) -> list[str]:
        """Remove and return signatures whose idle timeout has elapsed as of time t."""
        expired = [sig for sig, e in self._entries.items() if e.natural_expiry <= t]
        for sig in expired:
            del self._entries[sig]
        return expired

    def match(self, signature: str, t: float) -> MatchResult:
        """A flow arrives. Returns hit/miss and refreshes the entry's idle timer on hit."""
        self.expire(t)
        entry = self._entries.get(signature)
        if entry is not None:
            entry.last_used = t
            entry.natural_expiry = t + self.idle_timeout
            return MatchResult(hit=True, occupancy_after=len(self._entries))
        return MatchResult(hit=False, occupancy_after=len(self._entries))

    def install(self, signature: str, t: float) -> InstallResult:
        """
        Install a rule for `signature` at time t (proactive or reactive).
        If the table is at capacity and this is a new entry, evicts the
        least-recently-used entry first.
        """
        self.expire(t)
        if signature in self._entries:
            # already installed; just refresh (idempotent re-install)
            e = self._entries[signature]
            e.last_used = t
            e.natural_expiry = t + self.idle_timeout
            return InstallResult(installed=True, occupancy_after=len(self._entries))

        evicted_sig, evicted_cause, evicted_expiry = None, None, None
        if len(self._entries) >= self.capacity:
            lru_sig, lru_entry = min(self._entries.items(), key=lambda kv: kv[1].last_used)
            evicted_sig, evicted_cause = lru_sig, "capacity_lru"
            evicted_expiry = lru_entry.natural_expiry
            del self._entries[lru_sig]

        self._entries[signature] = _Entry(installed_at=t, last_used=t, natural_expiry=t + self.idle_timeout)
        return InstallResult(
            installed=True,
            caused_eviction=evicted_sig is not None,
            evicted_signature=evicted_sig,
            eviction_cause=evicted_cause,
            evicted_natural_expiry=evicted_expiry,
            occupancy_after=len(self._entries),
        )


def eviction_fields(proactive: "InstallResult | None", reactive: "InstallResult | None") -> dict:
    """
    Combine the (up to two) installs one flow arrival can trigger into the
    FlowEvent eviction fields. Used identically by scripts/replay.py and
    controller/ryu_app.py so offline and live logs count evictions the same way.

    (Earlier versions picked `proactive or reactive` on the InstallResult
    objects themselves -- always truthy -- so whenever a proactive install
    happened, a reactive eviction on the same flow was silently dropped.
    At low tau that undercounted evictions by up to ~70%.)
    """
    p = proactive if (proactive is not None and proactive.caused_eviction) else None
    r = reactive if (reactive is not None and reactive.caused_eviction) else None
    main = p or r
    return dict(
        caused_eviction=main is not None,
        n_evictions=(p is not None) + (r is not None),
        evicted_signature=main.evicted_signature if main else None,
        eviction_cause=main.eviction_cause if main else None,
        evicted_natural_expiry=main.evicted_natural_expiry if main else None,
        evicted_by="proactive" if p else ("reactive" if r else None),
        reactive_evicted_signature=r.evicted_signature if (p and r) else None,
    )


def analyze_harmful_evictions(events: list) -> None:
    """
    Retrospective, whole-run analysis implementing the pre-registered
    harmful-eviction definition (Research Plan Section 2.4):

        (i)   a proactive install evicts an existing rule;
        (ii)  that rule would have been requested again before its natural
              TTL expiry, i.e. before `ev.evicted_natural_expiry` (the
              evicted entry's own last_used + idle_timeout at the moment
              it was evicted -- set by FlowTable.install, NOT derived from
              the evicting flow's own timestamp);
        (iii) the resulting later arrival of the evicted signature is a
              miss that causes a controller interaction that would not
              otherwise have occurred.

    Mutates each FlowEvent's `.harmful_eviction` field in place:
        True  -> confirmed harmful by the definition above
        False -> a capacity_lru eviction occurred but condition (ii) or
                 (iii) was not met (e.g. the evicted flow never returns,
                 or only returns after it would have naturally expired anyway)
        None  -> not an eviction event (left untouched)

    NOTE: this requires the full ordered event list for ONE run (single
    strategy/tau/scenario/seed), not a mix of runs, since it looks forward
    in time within that run's own flow arrival sequence.
    """
    # index: for each signature, the ordered list of arrival timestamps
    # (t_arrival) at which it appears later in the run, with whether that
    # arrival was a miss/controller interaction.
    events_sorted = sorted(events, key=lambda e: e.t_arrival)

    for i, ev in enumerate(events_sorted):
        if not (ev.caused_eviction and ev.eviction_cause == "capacity_lru"):
            continue
        by = getattr(ev, "evicted_by", None)
        caused_by_proactive = (by == "proactive") if by is not None else (ev.decision == "proactive_install")
        if not caused_by_proactive:
            # condition (i) requires the eviction to be caused by a *proactive* install
            ev.harmful_eviction = False
            continue

        evicted_sig = ev.evicted_signature
        deadline = ev.evicted_natural_expiry
        if deadline is None:
            # Should not happen if FlowTable.install populated it correctly;
            # fail closed (not harmful) rather than guess.
            ev.harmful_eviction = False
            continue

        found_harm = False
        for later in events_sorted[i + 1:]:
            if later.signature != evicted_sig:
                continue
            # Only the *next* reappearance of this signature is relevant to
            # this specific eviction (later reappearances, if any, are
            # attributable to whatever happened at their own point in time).
            if later.t_arrival < deadline:
                # condition (ii): it came back before it would have expired naturally
                # condition (iii): and that arrival was a miss + controller interaction
                found_harm = later.match_type == "miss" and later.controller_interaction
            break

        ev.harmful_eviction = found_harm
