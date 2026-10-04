#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
TTI-Scope session library — hosts several multi-gigabyte captures at once.

The problem this solves is not disk space, it is RAM and startup time. A
single sweep iteration is ~4 GB of raw artifacts (3.2 GB nsys sqlite + 435 MB
nvlog + 225 MB printf log) and a sweep produces one of those per cell count.
Comparing 2C against 19C against 20C means three of them open together.

Two properties make that affordable:

  * Building is one-time. A capture becomes a ~250 MB indexed store; the
    4 GB of source files are never touched again. Rebuilds happen only when
    the store format changes.
  * Opening is O(1). SessionStore holds a read-only SQLite handle, the kernel
    table (59 rows) and the slot start array. Everything else is queried per
    view. Ten open sessions cost a few MB, not a few GB.

An LRU cap still applies, because each open store keeps an mmap window and a
file descriptor, and because a user who opens twenty sessions almost certainly
wants the two they are looking at to stay fast.

    lib = SessionLibrary()
    lib.scan("~/captures")
    for e in lib.entries: print(e.label, e.state)
    store = lib.open(entry)          # builds on first use, cached after
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ttiscope_cpu import CpuCapture, discover_cpu_captures
from ttiscope_ingest import Capture, SessionStore, default_store_path

# Sweep output directories: all_kernels_printf_<N>C_<pattern>_<stamp>
RUN_DIR_RE = re.compile(r"^(?:all_kernels_printf|pusch_kernel_printf|stress)_"
                        r"(\d+)C_(\w+?)_(\d{8})_(\d{6})$")

MAX_OPEN = 6


@dataclass
class SessionEntry:
    """A capture the library knows about. Cheap — no file is opened to make
    one, so a scan of a directory of 4 GB captures costs a stat() each."""
    path: Path
    label: str
    cells: Optional[int] = None
    pattern: str = ""
    when: str = ""
    store_path: Optional[Path] = None
    built: bool = False
    src_bytes: int = 0
    sources: str = ""
    note: str = ""

    @property
    def state(self) -> str:
        return "ready" if self.built else ("no sources" if not self.sources
                                           else "needs build")

    def sort_key(self):
        return (self.cells if self.cells is not None else 1 << 30, self.when)


class SessionLibrary:
    """Registry of captures plus an LRU cache of open stores."""

    def __init__(self, max_open: int = MAX_OPEN):
        self.entries: list[SessionEntry] = []
        self._open: dict[Path, SessionStore] = {}
        self._used: dict[Path, float] = {}
        self.max_open = max_open
        # CPU perf captures live in a different tree from the nsys captures and
        # come from different runs, so they are indexed separately and matched
        # to a session by (cells, pattern) — never assumed to be the same run.
        self.cpu_captures: dict = {}
        self.cpu_roots: list[str] = []

    # -- discovery ----------------------------------------------------------
    def scan(self, root, recurse: bool = True) -> list[SessionEntry]:
        """Add every capture directory under `root`. Returns the new entries.

        A directory is a capture if it directly contains — or wraps, via the
        tmp/<run>/ nesting the sweep archive preserves — at least one of the
        three artifact kinds.
        """
        root = Path(root).expanduser()
        found = []
        if not root.exists():
            return found
        cands = [root]
        if recurse:
            cands += [p for p in sorted(root.iterdir()) if p.is_dir()]
        known = {e.path for e in self.entries}
        for d in cands:
            if d in known:
                continue
            try:
                cap = Capture.discover(d)
            except (OSError, StopIteration):
                continue
            if not (cap.sqlite or cap.ktrace_log or cap.nvlog):
                continue
            e = self._entry_for(d, cap)
            self.entries.append(e)
            known.add(d)
            found.append(e)
        self.entries.sort(key=SessionEntry.sort_key)
        return found

    def remove(self, entries) -> int:
        """Drop sessions from the library. Returns how many were removed.

        Only the index entry goes; the capture directory on disk is untouched,
        so removing is always safe and re-addable with Add folder. Any open
        store for a removed path is closed so its sqlite handle is released.
        """
        if not isinstance(entries, (list, tuple, set)):
            entries = [entries]
        paths = {str(getattr(e, "path", e)) for e in entries}
        before = len(self.entries)
        for p in paths:
            st = self._open.pop(p, None) if hasattr(self, "_open") else None
            if st is not None:
                try:
                    st.close()
                except Exception:
                    pass
        self.entries = [e for e in self.entries if str(e.path) not in paths]
        return before - len(self.entries)

    def remove_cpu_root(self, root) -> int:
        """Forget a CPU capture root. Returns how many roots were removed."""
        root = str(root)
        before = len(self.cpu_roots)
        self.cpu_roots = [r for r in self.cpu_roots if str(r) != root]
        return before - len(self.cpu_roots)

    def scan_cpu(self, root) -> int:
        """Index perf_percore_* captures under `root`. Returns how many."""
        found = discover_cpu_captures(root)
        self.cpu_captures.update(found)
        r = str(Path(root).expanduser())
        if found and r not in self.cpu_roots:
            self.cpu_roots.append(r)
        return len(found)

    def cpu_for(self, entry: SessionEntry) -> Optional[CpuCapture]:
        """The perf capture matching this session's cell count and pattern."""
        if entry.cells is None:
            return None
        c = self.cpu_captures.get((entry.cells, entry.pattern))
        if c is not None:
            return c
        # Fall back on cell count alone — the pattern is usually identical
        # across a sweep and a mismatch there is not worth losing the match.
        for (cells, _pat), cap in self.cpu_captures.items():
            if cells == entry.cells:
                return cap
        return None

    def add(self, path) -> Optional[SessionEntry]:
        """Add one capture directory (or a directory containing one)."""
        got = self.scan(path, recurse=False)
        if got:
            return got[0]
        got = self.scan(path, recurse=True)
        return got[0] if got else None

    @staticmethod
    def _entry_for(d: Path, cap: Capture) -> SessionEntry:
        m = RUN_DIR_RE.match(d.name)
        cells = int(m.group(1)) if m else None
        pattern = m.group(2) if m else ""
        when = f"{m.group(3)}-{m.group(4)}" if m else ""
        label = (f"{cells}C {pattern}" if m else d.name)
        sp = default_store_path(d)
        total = 0
        for p in (cap.sqlite, cap.ktrace_log, cap.nvlog):
            if p:
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        note = ""
        if cap.nsys_rep and not cap.sqlite:
            note = ("only a .nsys-rep is present — export it first:\n"
                    f"    nsys export --type sqlite -o "
                    f"{cap.nsys_rep.with_suffix('.sqlite').name} "
                    f"{cap.nsys_rep.name}\n"
                    "  without it the GPU timeline has no host-launched "
                    "kernels and no wall-clock anchor")
        return SessionEntry(path=d, label=label, cells=cells, pattern=pattern,
                            when=when, store_path=sp, built=sp.exists(),
                            src_bytes=total, sources=cap.describe(), note=note)

    # -- opening ------------------------------------------------------------
    def open(self, entry: SessionEntry, log: Callable = print,
             force: bool = False) -> SessionStore:
        """Open (building on first use). LRU-evicts to stay under max_open."""
        key = entry.path
        st = self._open.get(key)
        if st is not None and not force:
            self._used[key] = time.time()
            return st
        if st is not None:
            st.close()
            del self._open[key]
        st = SessionStore.open_or_build(entry.path, entry.store_path,
                                        force=force, log=log)
        entry.built = True
        entry.store_path = st.path
        self._open[key] = st
        self._used[key] = time.time()
        self._evict()
        return st

    def _evict(self):
        while len(self._open) > self.max_open:
            oldest = min(self._used, key=self._used.get)
            self._open.pop(oldest).close()
            self._used.pop(oldest, None)

    def close(self, entry: SessionEntry):
        st = self._open.pop(entry.path, None)
        self._used.pop(entry.path, None)
        if st:
            st.close()

    def close_all(self):
        for st in self._open.values():
            st.close()
        self._open.clear()
        self._used.clear()

    # -- persistence --------------------------------------------------------
    # This is a hacky code. We shall revisit later
    
    def state_file(self) -> Path:
        d = Path(os.environ.get("XDG_CACHE_HOME",
                                Path.home() / ".cache")) / "ttiscope"
        d.mkdir(parents=True, exist_ok=True)
        return d / "library.json"

    def save(self):
        try:
            self.state_file().write_text(json.dumps(
                {"captures": [str(e.path) for e in self.entries],
                 "cpu_roots": self.cpu_roots}, indent=1))
        except OSError:
            pass

    def load(self):
        f = self.state_file()
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            return

        caps = data.get("captures", []) if isinstance(data, dict) else data
        for p in caps:
            if Path(p).exists():
                self.scan(Path(p), recurse=False)
        if isinstance(data, dict):
            for r in data.get("cpu_roots", []):
                if Path(r).exists():
                    self.scan_cpu(r)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Scan a directory of captures and build their stores")
    ap.add_argument("root", nargs="+")
    ap.add_argument("-b", "--build", action="store_true",
                    help="build stores for every capture found")
    ap.add_argument("-f", "--force", action="store_true")
    a = ap.parse_args()

    lib = SessionLibrary()
    for r in a.root:
        lib.scan(r)
    print(f"{len(lib.entries)} capture(s)\n")
    for e in lib.entries:
        print(f"  {e.label:<14} {e.state:<12} {e.src_bytes/1e9:5.1f} GB  "
              f"[{e.sources}]")
        if e.note:
            print(f"    note: {e.note}")
    if not a.build:
        return
    for e in lib.entries:
        if not e.sources:
            continue
        print(f"\n=== {e.label} ===")
        try:
            st = lib.open(e, force=a.force)
            print(f"  {st.n_events:,} events, {st.n_slots:,} slots, "
                  f"{len(st.kernels)} kernels")
        except SystemExit as ex:
            print(f"  SKIPPED: {ex}")
    lib.save()


if __name__ == "__main__":
    main()
