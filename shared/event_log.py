"""
Append-only JSONL event logger. One file per run (RunKey.run_id()).

Used identically by:
  - the offline replay harness (table_sim + policy, no real network), and
  - the live Ryu controller (controller/ryu_app.py) during live runs.

This is what makes Section 2.4's "all metrics derived from per-flow event
logs" true in practice: analysis code never touches predictors, the table,
or Mininet/Ryu directly, only these log files.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Iterable
from .schema import FlowEvent


class EventLogWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", buffering=1)  # truncate: re-runs must not duplicate lines

    def write(self, event: FlowEvent) -> None:
        self._fh.write(event.to_json() + "\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_events(path: str | Path) -> Iterator[FlowEvent]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield FlowEvent.from_json(line)


def read_events_list(path: str | Path) -> list[FlowEvent]:
    return list(read_events(path))
