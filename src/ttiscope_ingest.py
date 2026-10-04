#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
TTI-Scope ingest — builds one compact, indexed *session store* from everything
a sweep produces, so the UI can query TTI windows on demand instead of holding
a multi-gigabyte capture in RAM.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from ttiscope_occupancy import (P_IDLE, P_TDP, dynamic_watts,
                                load_static_resources, slot_energy_uj,
                                theoretical_occupancy)
from ttiscope_taxonomy import DEVICE_FUNCTIONS, KTRACE_ONLY, classify_kernel

SCHEMA_VERSION = 7

# ─────────────────────────────────────────────────────────────────────────────
# Log grammars
# ─────────────────────────────────────────────────────────────────────────────

# KERNEL_TRACE_ENTRY(), one line per launch. NOT anchored with ^: device printf
# is drained in fixed-size chunks and a flush boundary can land mid-line, which
# in the 20-cell log produced ~250 records glued to the tail of the previous
# one (",1KTRACE t0=..."). Scanning rather than matching recovers them.
KTRACE_RE = re.compile(
    r"KTRACE t0=(\d+) grid=(\d+),(\d+),(\d+) block=(\d+),(\d+),(\d+) fn=(.*)")
# KERNEL_TIME_END_PRINT()
KDUR_RE = re.compile(r"KERNEL_DURATION\s+([A-Za-z_]\w*)\s+duration_ns=(\d+)")
IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

# nvlog: "... [L2A.TICK_TIMES] SFN 262.10 current_time=<ns>, tick=<ns>"
# The 500us slot grid, and the only source of true SFN.slot identity.
TICK_RE = re.compile(
    r"\[L2A\.TICK_TIMES\]\s+SFN\s+(\d+)\.(\d+)\s+current_time=(\d+),\s*tick=(\d+)")

# nvlog, written by the TTI-miss instrumentation in cuphydriver_api.cpp
# (cleanup_err) and scf_5g_fapi_phy.cpp (L1 recovery). Both carry ANSI colour,
# so every field is matched loosely rather than anchored to the line start:
#   "\x1b[1;31m[TTI-MISS #12] SFN 164.11 DROPPED  dt_from_tick=130045 ns
#    (130.0 us = 0.260 slot periods)  from cuphydriver_api.cpp:1136 ..."
#   "\x1b[1;31m[TTI-MISS-RECOVERY #210] SFN 176.9 DROPPED  L1 in recovery ..."
MISS_RE = re.compile(
    r"\[TTI-MISS(?P<rec>-RECOVERY)?\s+#(?P<n>\d+)\]\s+"
    r"SFN\s+(?P<sfn>\d+)\.(?P<slot>\d+)\s+DROPPED")
MISS_DT_RE = re.compile(r"dt_from_tick=(-?\d+)\s*ns")
MISS_SITE_RE = re.compile(r"from\s+([A-Za-z0-9_]+\.cpp:\d+)")
# Negative lookbehind: without it this matches inside "dt_from_tick=",
# which silently records the latency as the tick timestamp.
MISS_TICK_RE = re.compile(r"(?<![_A-Za-z])tick=(\d+)\s*ns")
# nvlog: "{TI} <DL Task PDSCH,262,9,99,8> <0> Start Task:<ns>,Cuda Setup:<ns>,..."
#              ^name          ^sfn ^slot ^buf ^worker  ^cell
TI_RE = re.compile(
    r"\{TI\}\s+<([^,>]+),(\d+),(\d+),(\d+),(\d+)>\s+<(\d+)>\s+(.*)")
TI_STAGE_RE = re.compile(r"([A-Za-z][A-Za-z0-9 _/-]*?):(\d{15,})")
# nvlog startup banner: "[CTL.YAML] cell_id 7 nic_index :0" — the only place
# the configured cell count appears. The {TI} <0> field is the cell GROUP.
CELL_RE = re.compile(r"cell_id\s+(\d+)\s+nic_index")
# cuphycontroller startup, one line per MPS context it creates, in creation
# order: "[DRV.CTX] PUSCH MPS context with max. SM count of 60."
# The SM count is the execution-affinity cap that context's kernels run
# under. It appears in both ctrl_stdout.log and the nvlog.
MPS_CTX_RE = re.compile(
    r"\[DRV\.CTX\]\s+(.+?)\s+MPS context with max\. SM count of\s+(\d+)")

# Which channels each cuphycontroller MPS context is expected to run, used
# to cross-check the creation-order mapping of log lines onto contextIds.
MPS_CTX_EXPECT = {
    "PUSCH": {"PUSCH", "UCI"}, "PUCCH": {"PUCCH"}, "PRACH": {"PRACH"},
    "PDSCH": {"PDSCH", "FH-DL", "CSI-RS"}, "DL Ctrl": {"PDCCH", "CSI-RS", "SSB"},
    "UL Order": {"FH-UL"}, "GPU comm": {"FH-DL"}, "SRS": {"SRS"},
}

SRC_NSYS, SRC_KTRACE = 0, 1

# A launch whose duration came from the per-kernel median rather than a
# measured KERNEL_DURATION sample. Kept explicit so the UI can hatch those bars
# instead of presenting an estimate as a measurement.
DUR_MEASURED, DUR_ESTIMATED = 0, 1

# Calibration keeps per-kernel timestamp samples in RAM. Kernels launched more
# often than this are poor calibration candidates anyway (see _clock_offset),
# so capping costs nothing and bounds memory at ~60 x 50k x 8B.
CALIB_CAP = 50_000


def short_kernel_name(sig: str) -> str:
    """'void ch_est::fooKernel<A,B>(ns::T*, ns::U*) [with ...]' -> 'fooKernel'

    Parameter types are themselves namespaced, so splitting the whole string on
    '::' lands inside the parameter list. Cut at the first '<' or '(' to isolate
    the qualified name first.
    """
    s = (sig or "").strip()
    if s.startswith("void "):
        s = s[5:]
    cut = len(s)
    for ch in ("<", "("):
        i = s.find(ch)
        if i != -1:
            cut = min(cut, i)
    return s[:cut].split("::")[-1].strip()


def phase_base(name: str) -> str:
    """'DL PBCH Run507' -> 'DL PBCH Run'. cuPHY appends a per-instance counter
    to some NVTX labels, which turns ~20 real phases into 4,500 menu rows."""
    s = name.rstrip("0123456789").rstrip()
    return s if len(s) >= 2 else name


def warps_of(blocks: int, tpb: int) -> int:
    """Warps resident if every block were co-resident. The occupancy
    denominator is per-SM capacity, so this is the numerator of 'how much of
    the machine did this launch ask for'."""
    if not blocks or not tpb:
        return 0
    return int(blocks) * ((int(tpb) + 31) // 32)


# ─────────────────────────────────────────────────────────────────────────────
# Store schema
# ─────────────────────────────────────────────────────────────────────────────

STORE_DDL = """
PRAGMA journal_mode=OFF;
PRAGMA synchronous=OFF;

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE kernels (
    id           INTEGER PRIMARY KEY,
    name         TEXT UNIQUE,
    channel      TEXT,
    role         TEXT,
    nsys_visible INTEGER,       -- 0 = no nsys event for it in THIS capture
    src_mask     INTEGER,       -- bit0 = seen by nsys, bit1 = seen by KTRACE
    n_launch     INTEGER,
    dur_n        INTEGER,       -- measured duration samples
    dur_min      INTEGER, dur_med INTEGER, dur_p99 INTEGER, dur_max INTEGER,
    -- Theoretical occupancy of this kernel's most common launch shape, and
    -- the resource that caps it. res_src says where regs/smem came from:
    --   'cupti'     actual per-launch values recorded by nsys
    --   'cuobjdump' static maximum across template instantiations, i.e. a
    --               LOWER bound, for kernels nsys cannot see
    --   'unknown'   neither available; occupancy is left NULL, not guessed
    regs         INTEGER, smem INTEGER, res_src TEXT,
    occ_pct      REAL,    occ_limiter TEXT,
    -- 'kernel'    a real __global__ entry point
    -- 'device_fn' a __device__ function whose body carries a KTRACE macro, so
    --             it emits launch-shaped records that are NOT launches
    -- 'unknown'   no cuobjdump table available to decide
    kind         TEXT,
    parent       TEXT           -- for device_fn: the kernel it runs inside
);

-- One row per kernel launch, from EITHER source, on the nsys timeline.
CREATE TABLE events (
    t0        INTEGER NOT NULL,
    t1        INTEGER,
    kernel_id INTEGER NOT NULL,
    src       INTEGER NOT NULL,   -- 0=nsys 1=ktrace
    dur_kind  INTEGER NOT NULL,   -- 0=measured 1=estimated from median
    stream    INTEGER,
    blocks    INTEGER,
    tpb       INTEGER,
    warps     INTEGER,
    regs      INTEGER,
    smem      INTEGER,
    occ       REAL,               -- theoretical occupancy % of THIS launch
    slot_idx  INTEGER,            -- resolved at build time
    -- CUDA context the launch ran in. cuPHY creates one MPS context per
    -- channel group, each capped to a number of SMs (see `contexts`), so
    -- this is what bounds how many SMs a launch can occupy at once.
    ctx       INTEGER,
    gctx      INTEGER,            -- green context id, 0/NULL when none
    -- CUPTI sharedMemoryExecuted: the L1/shared carveout the driver
    -- configured for this launch, bytes per SM. NULL for KTRACE rows.
    smem_cfg  INTEGER
);

-- SM cap per CUDA context. src: 'green_context' (TARGET_INFO numMultiprocessors),
-- 'mps_log' ([DRV.CTX] lines, mapped by creation order), or 'none'.
-- verified = the context's dominant channel matches its log label.
CREATE TABLE contexts (
    ctx INTEGER PRIMARY KEY, process INTEGER, label TEXT, sm_cap INTEGER,
    src TEXT, verified INTEGER, n_launch INTEGER, channel TEXT
);

-- The 500us TTI grid. idx is capture order; sfn/slot are the DU's own naming.
CREATE TABLE slots (
    idx      INTEGER PRIMARY KEY,
    t0       INTEGER NOT NULL,   -- nsys ns
    t1       INTEGER NOT NULL,
    sfn      INTEGER,
    slot     INTEGER,
    kind     TEXT NOT NULL       -- 'tick' | 'nvtx' | 'inferred'
);

-- Precomputed per-slot rollup: the global views never touch `events`.
CREATE TABLE slot_stats (
    idx        INTEGER PRIMARY KEY,
    n_kernels  INTEGER, n_nsys INTEGER, n_ktrace INTEGER,
    busy_ns    INTEGER,          -- sum of durations (overlap not removed)
    span_ns    INTEGER,          -- union of busy intervals: real GPU busy time
    warp_ns    INTEGER,          -- sum(warps * duration) -> warp demand
    n_streams  INTEGER,
    idle_ns    INTEGER,          -- slot length - span_ns
    energy_uj  REAL,             -- ESTIMATED, see ttiscope_occupancy
    occ_w      REAL,             -- busy-time-weighted theoretical occupancy %
    occ_cov    REAL,             -- fraction of busy time that occ_w covers
    n_devfn    INTEGER           -- device-function trace records, NOT launches
);

-- TTI misses parsed from the nvlog TTI-MISS markers. One row per dropped
-- slot. `kind` is 'cleanup' (slot command abandoned in l1_enqueue_phy_work)
-- or 'recovery' (L1 in recovery, already deduplicated per SFN.slot by the
-- instrumentation itself). dt_ns is how long after the slot's own tick the
-- drop decision was taken; NULL for recovery rows, which have no such figure.
CREATE TABLE tti_miss (
    seq      INTEGER,            -- the #n from the marker
    kind     TEXT NOT NULL,
    sfn      INTEGER, slot INTEGER,
    tick_ns  INTEGER,            -- nvlog ns, mapped to nsys ns at read time
    dt_ns    INTEGER,
    site     TEXT,
    -- SFN wraps every 1024 frames (10.24 s), so (sfn,slot) recurs roughly
    -- every 20,480 slots and is NOT unique in a capture of any length. A
    -- 40 s capture repeats each pair ~4x. Joining misses to slots on
    -- (sfn,slot) therefore marks 4 slots per miss, 3 of them at times the
    -- DU never dropped. This column pins each miss to the one slot it
    -- actually happened in, resolved at ingest.
    slot_idx INTEGER
);
CREATE INDEX tti_miss_sfn ON tti_miss(sfn, slot);
CREATE INDEX tti_miss_slot ON tti_miss(slot_idx);

CREATE TABLE slot_channel (
    idx     INTEGER, channel TEXT,
    n       INTEGER, busy_ns INTEGER, warp_ns INTEGER
);

-- NVTX ranges: the cuPHY pipeline phases (SLOT_DL, cuphyRunPuschRx-ALL, ...)
--   base : name with its per-instance counter stripped, so 'DL PBCH Run507'
--          groups with 'DL PBCH Run27' instead of adding 4,500 menu entries.
CREATE TABLE phases (
    t0       INTEGER NOT NULL,
    t1       INTEGER NOT NULL,
    name     TEXT NOT NULL,
    base     TEXT NOT NULL,
    slot_idx INTEGER
);

-- Host-side per-TTI task pipeline from nvlog {TI} records.
CREATE TABLE tasks (
    t0       INTEGER NOT NULL,   -- nsys ns
    t1       INTEGER NOT NULL,
    name     TEXT NOT NULL,
    sfn      INTEGER, slot INTEGER, cell INTEGER, worker INTEGER,
    slot_idx INTEGER,
    stages   TEXT                -- 'Stage:rel_ns;Stage:rel_ns;...'
);

CREATE TABLE streams (
    id INTEGER PRIMARY KEY, label TEXT, role TEXT, n INTEGER
);
"""

STORE_INDICES = """
CREATE INDEX idx_events_t0     ON events(t0);
CREATE INDEX idx_events_slot   ON events(slot_idx);
CREATE INDEX idx_events_kernel ON events(kernel_id);
CREATE INDEX idx_phases_t0     ON phases(t0);
CREATE INDEX idx_phases_slot   ON phases(slot_idx);
CREATE INDEX idx_phases_base   ON phases(base);
CREATE INDEX idx_tasks_slot    ON tasks(slot_idx);
CREATE INDEX idx_slotchan      ON slot_channel(idx);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Capture discovery
# ─────────────────────────────────────────────────────────────────────────────

def _is_nvlog(name: str) -> bool:
    """True for any spelling of the cuPHY nvlog, plain or gzipped."""
    n = name.lower()
    if n.startswith("kernel_printf_"):
        return False                      # that is the KTRACE log, not the nvlog
    if not (n.endswith(".log") or n.endswith(".log.gz")):
        return False
    stem = n[:-3] if n.endswith(".gz") else n
    stem = stem[:-4]                       # drop ".log"
    return stem in ("cuphy", "phy") or stem.startswith("cuphy_") or stem.startswith("phy_")


@dataclass
class Capture:
    """The files one sweep iteration produces. A sweep writes them into a
    directory named all_kernels_printf_<N>C_<pat>_<stamp>, sometimes nested
    under tmp/<same name>/ because the archive preserves the container path."""
    root: Path
    sqlite: Optional[Path] = None
    ktrace_log: Optional[Path] = None
    nvlog: Optional[Path] = None
    nsys_rep: Optional[Path] = None
    ctrl_log: Optional[Path] = None     # cuphycontroller console output
    label: str = ""

    @classmethod
    def discover(cls, d: Path) -> "Capture":
        d = Path(d)
        # Descend through single-child wrapper dirs (tmp/<run>/) to the level
        # that actually holds the artifacts.
        probe = d
        for _ in range(4):
            if any(p.suffix in (".sqlite", ".nsys-rep") or
                   p.name.startswith("kernel_printf_") for p in probe.iterdir()
                   if p.is_file()):
                break
            subs = [p for p in probe.iterdir() if p.is_dir()]
            if len(subs) != 1:
                break
            probe = subs[0]

        c = cls(root=probe, label=d.name)
        for p in sorted(probe.iterdir()):
            if not p.is_file():
                continue
            n = p.name
            if n.endswith(".sqlite"):
                c.sqlite = p
            elif n.endswith(".nsys-rep"):
                c.nsys_rep = p
            elif n.startswith("kernel_printf_") and n.endswith(".log"):
                c.ktrace_log = p
            elif n.startswith("ctrl") and n.endswith(".log"):
                c.ctrl_log = p
            elif _is_nvlog(n):
                # The nvlog is the ONLY source of true SFN.slot. Capture
                # directories name it several ways depending on who wrote them:
                #   cuphy_20C_59c.log   the original convention
                #   cuphy.log           what aerial_join.sh writes
                #   cuphy.log.gz        what it writes when the log is kept
                #   phy.log             the raw nvlog name
                # Matching only "cuphy_*.log" silently produced a capture with
                # no SFN at all - the boundaries fell back to tiling from zero
                # and every slot label was blank.
                if c.nvlog is None or p.stat().st_size > c.nvlog.stat().st_size:
                    c.nvlog = p
        return c

    def describe(self) -> str:
        bits = []
        for tag, p in (("nsys", self.sqlite), ("ktrace", self.ktrace_log),
                       ("nvlog", self.nvlog)):
            if p:
                bits.append(f"{tag} {p.stat().st_size/1e6:.0f}MB")
        if self.nsys_rep and not self.sqlite:
            bits.append("nsys-rep NOT EXPORTED")
        return ", ".join(bits) or "empty"


# ─────────────────────────────────────────────────────────────────────────────
# Read side
# ─────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """Read-side API. Every query is windowed — nothing is held in RAM beyond
    the kernel table and the slot index."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True,
                                  check_same_thread=False)
        self.db.execute("PRAGMA mmap_size=1073741824")   # 1GB mmap window
        self._meta = dict(self.db.execute("SELECT key,value FROM meta"))
        self.kernels = {i: (n, ch, ro, bool(vis), kind) for
                        i, n, ch, ro, vis, kind in
                        self.db.execute("SELECT id,name,channel,role,"
                                        "nsys_visible,kind FROM kernels")}
        self.kernel_names = {i: v[0] for i, v in self.kernels.items()}
        self._load_provenance()
        self._reclassify()
        self._slot_t0 = [r[0] for r in
                         self.db.execute("SELECT t0 FROM slots ORDER BY idx")]
        # Small result cache. The expensive windowed queries are pure
        # functions of (lo, hi), and the common navigation pattern is to move
        # back and forth over the same few windows - without this, every
        # revisit repays the full cost (intra_slot_profile is ~420 ms over
        # 1,000 slots). Bounded so a long session cannot grow it without limit.
        self._cache: dict = {}
        self._cache_order: list = []

    _CACHE_MAX = 24

    def _cached(self, key, compute):
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        val = compute()
        self._cache[key] = val
        self._cache_order.append(key)
        while len(self._cache_order) > self._CACHE_MAX:
            self._cache.pop(self._cache_order.pop(0), None)
        return val

    # -- TTI misses ---------------------------------------------------------
    def tti_miss_all(self):
        """Every parsed TTI miss: (seq, kind, sfn, slot, tick_ns, dt_ns, site).

        Empty when the capture predates the instrumentation, so every caller
        must treat "no rows" as "not measured", never as "none occurred".
        """
        try:
            return list(self.db.execute(
                "SELECT seq,kind,sfn,slot,tick_ns,dt_ns,site FROM tti_miss "
                "ORDER BY seq"))
        except sqlite3.OperationalError:
            return []

    def has_tti_miss(self) -> bool:
        try:
            self.db.execute("SELECT 1 FROM tti_miss LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False

    def tti_miss_slot_idx(self) -> dict:
        """slot_idx -> (kind, sfn, slot, dt_ns, site) for misses the capture
        actually covers. Keyed by slot index so the views can mark a lane
        without re-joining per frame."""
        try:
            rows = self.db.execute(
                "SELECT s.idx,m.kind,m.sfn,m.slot,m.dt_ns,m.site "
                "FROM tti_miss m JOIN slots s ON s.idx=m.slot_idx")
            return {r[0]: (r[1], r[2], r[3], r[4], r[5]) for r in rows}
        except sqlite3.OperationalError:
            return {}

    def tti_miss_in(self, lo, hi) -> list:
        """Misses whose slot overlaps [lo,hi] in nsys ns."""
        try:
            return list(self.db.execute(
                "SELECT s.t0,s.t1,m.kind,m.sfn,m.slot,m.dt_ns,m.site "
                "FROM tti_miss m JOIN slots s ON s.idx=m.slot_idx "
                "WHERE s.t1>=? AND s.t0<=? ORDER BY s.t0", (lo, hi)))
        except sqlite3.OperationalError:
            return []

    # -- metadata -----------------------------------------------------------
    def _m(self, key, cast=str, default=None):
        v = self._meta.get(key)
        if v in (None, "", "None"):
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default

    @property
    def label(self):            return self._m("label", str, self.path.stem)
    @property
    def clock_offset_ns(self):  return self._m("clock_offset_ns", int)
    @property
    def clock_iqr_ns(self):     return self._m("clock_iqr_ns", int)
    @property
    def utc_epoch_ns(self):     return self._m("utc_epoch_ns", int)
    @property
    def gpu_name(self):         return self._m("gpu_name", str, "Unknown GPU")
    @property
    def sm_count(self):         return self._m("sm_count", int, 132)
    @property
    def max_warps(self):        return self._m("max_warps", int, 132 * 64)
    @property
    def n_slots(self):          return self._m("n_slots", int, 0)
    @property
    def n_events(self):         return self._m("n_events", int, 0)
    @property
    def n_cells(self):          return self._m("n_cells", int, 0)
    @property
    def slot_dur_ns(self):      return self._m("slot_dur_ns", int, 500_000)
    @property
    def max_dur_ns(self):       return self._m("max_dur_ns", int, 10_000_000)
    @property
    def max_cluster(self):      return self._m("max_cluster", int, 1)

    def context_caps(self) -> dict:
        """ctx -> (sm_cap or None, label, src, verified). Empty for stores
        built before schema 7, which the spare model reports as uncapped."""
        try:
            return {r[0]: (r[1], r[2], r[3], bool(r[4])) for r in self.db.execute(
                "SELECT ctx,sm_cap,COALESCE(label,channel),src,verified "
                "FROM contexts")}
        except sqlite3.OperationalError:
            return {}

    def wall_ns(self, nsys_ns: int) -> Optional[int]:
        e = self.utc_epoch_ns
        return None if e is None else e + int(nsys_ns)

    # -- slots --------------------------------------------------------------
    def slot(self, idx: int):
        return self.db.execute(
            "SELECT idx,t0,t1,sfn,slot,kind FROM slots WHERE idx=?",
            (idx,)).fetchone()

    def slot_of_ns(self, t_ns: int) -> int:
        """Slot containing t_ns, clamped. Bisect over the cached t0 array."""
        if not self._slot_t0:
            return 0
        i = bisect.bisect_right(self._slot_t0, int(t_ns)) - 1
        return max(0, min(i, len(self._slot_t0) - 1))

    def slots_by_sfn(self, sfn: int, slot=None) -> list:
        """Every slot index carrying this SFN[.slot], in time order.

        Returns a list, never a single index, because SFN wraps at 1024 frames
        (10.24 s): any longer capture answers to the same SFN.slot several
        times over. The caller picks which occurrence it meant - collapsing
        them here would silently jump to the first one and call it "the" slot.
        """
        if slot is None:
            rows = self.db.execute(
                "SELECT idx FROM slots WHERE sfn=? ORDER BY idx", (int(sfn),))
        else:
            rows = self.db.execute(
                "SELECT idx FROM slots WHERE sfn=? AND slot=? ORDER BY idx",
                (int(sfn), int(slot)))
        return [r[0] for r in rows]

    def slot_label(self, idx: int) -> str:
        r = self.slot(idx)
        if not r:
            return f"#{idx}"
        return f"SFN {r[3]}.{r[4]}" if r[3] is not None else f"#{idx}"

    def slot_stats(self, lo: int, hi: int):
        """Per-slot rollup for [lo,hi] — reads slot_stats, never events.

        Columns: idx, t0, t1, sfn, slot, n_kernels, busy_ns, span_ns, warp_ns,
                 n_ktrace, idle_ns, energy_uj, occ_w, occ_cov
        """
        return self.db.execute(
            "SELECT s.idx,s.t0,s.t1,s.sfn,s.slot,"
            "       COALESCE(st.n_kernels,0),COALESCE(st.busy_ns,0),"
            "       COALESCE(st.span_ns,0),COALESCE(st.warp_ns,0),"
            "       COALESCE(st.n_ktrace,0),"
            "       COALESCE(st.idle_ns, s.t1-s.t0),st.energy_uj,st.occ_w,"
            "       COALESCE(st.occ_cov,0) "
            "FROM slots s LEFT JOIN slot_stats st ON st.idx=s.idx "
            "WHERE s.idx BETWEEN ? AND ? ORDER BY s.idx", (lo, hi)).fetchall()

    def intra_slot_profile(self, lo: int, hi: int, n_bins: int = 5):
        return self._cached(("intra", lo, hi, n_bins),
                            lambda: self._intra_slot_profile(lo, hi, n_bins))

    def kernel_window_stats_cached(self, lo: int, hi: int):
        return self._cached(("kws", lo, hi),
                            lambda: self.kernel_window_stats(lo, hi))

    def _intra_slot_profile(self, lo: int, hi: int, n_bins: int = 5):
        """Where inside a TTI the work actually lands.

        Every slot in [lo,hi] is divided into `n_bins` equal sub-windows (5 x
        100 us for a 500 us TTI) and the same three quantities are computed for
        each, then aggregated ACROSS slots. This answers "which 100 us of the
        slot is busiest", which a per-slot scalar cannot.

        Returns a dict of per-bin lists, each of length n_bins:
          edges_us   bin boundaries, length n_bins+1
          occ        mean busy-time-weighted theoretical occupancy %
          occ_lo/hi  P5 / P95 of the per-slot values
          conc       mean number of concurrently resident kernels
          e_min/e_p95/e_max   energy in uJ across the slots
          n_slots    how many slots contributed

        Energy uses the same swept, TDP-clamped instantaneous power as the
        per-slot rollup, so summing the five bins reproduces the slot figure.
        A kernel that starts in one bin and ends in another is split between
        them; one that spans a slot boundary is split across slots.
        """
        rows = self.db.execute(
            "SELECT idx,t0,t1 FROM slots WHERE idx BETWEEN ? AND ? ORDER BY idx",
            (lo, hi)).fetchall()
        if not rows:
            return None
        bounds = {r[0]: (r[1], r[2]) for r in rows}
        dyn_cap = max(0.0, P_TDP - P_IDLE)

        # Reach back two slots: an event is filed under the slot it STARTS in,
        # so a long kernel from an earlier slot still occupies this one's first
        # bin and would otherwise leave it looking idle.
        cur = self.db.execute(
            "SELECT e.slot_idx,e.t0,e.t1,e.occ,e.warps,k.name FROM events e "
            "JOIN kernels k ON k.id=e.kernel_id "
            "WHERE e.slot_idx BETWEEN ? AND ? AND e.t1 IS NOT NULL "
            "AND e.t1 > e.t0 AND k.kind IS NOT 'device_fn' ORDER BY e.t0",
            (lo - 2, hi))

        per_slot: dict = {}
        dw_cache: dict = {}
        for _s, t0, t1, occ, warps, name in cur:
            w = min(int(warps or 0), self.max_warps)
            key = (name, w)
            dw = dw_cache.get(key)
            if dw is None:
                dw = dynamic_watts(name, w, self.max_warps)
                dw_cache[key] = dw
            s = self.slot_of_ns(t0)
            while s <= hi:
                b = bounds.get(s)
                if b is None:
                    s += 1
                    continue
                if b[0] >= t1:
                    break
                a, z = max(t0, b[0]), min(t1, b[1])
                if z > a:
                    # t0 is carried so a launch can be counted in the bin the
                    # kernel STARTS in and only there - a kernel spanning three
                    # bins is one launch, not three.
                    per_slot.setdefault(s, []).append((a, z, dw, occ, w, t0))
                s += 1

        occ_v = [[] for _ in range(n_bins)]
        con_v = [[] for _ in range(n_bins)]
        e_v = [[] for _ in range(n_bins)]
        # Per-bin counterparts of the other rows in the Statistics table, so
        # every metric listed there can be folded into the slot the same way
        # occupancy is - not just the three this profile started with.
        u_v = [[] for _ in range(n_bins)]      # GPU utilisation %
        i_v = [[] for _ in range(n_bins)]      # idle us
        d_v = [[] for _ in range(n_bins)]      # warp demand %
        l_v = [[] for _ in range(n_bins)]      # launches
        n_used = 0
        for s, segs in per_slot.items():
            st0, st1 = bounds[s]
            dur = st1 - st0
            if dur <= 0:
                continue
            bin_ns = dur / n_bins
            occ_n = [0.0] * n_bins
            occ_d = [0.0] * n_bins
            busy = [0.0] * n_bins
            watt = [0.0] * n_bins
            # UNION of kernel intervals, which is what "GPU busy" means. busy[]
            # above sums overlapping kernels and so measures concurrency, not
            # utilisation; using it as a percentage would exceed 100%.
            uni = [0.0] * n_bins
            warp = [0.0] * n_bins
            lau = [0] * n_bins

            # Sweep instantaneous power once, clamping the concurrent sum to
            # the board limit, and split each constant-power stretch across
            # whichever bins it covers.
            ev = []
            for a, z, dw, _o, _w, _k0 in segs:
                ev.append((a, dw, 1))
                ev.append((z, -dw, -1))
            ev.sort()
            cur_w, depth, prev_t = 0.0, 0, None
            for t, d, k in ev:
                if prev_t is not None and t > prev_t and depth > 0:
                    p = min(cur_w, dyn_cap)
                    x = prev_t
                    while x < t:
                        bi = min(n_bins - 1, int((x - st0) // bin_ns))
                        edge = st0 + (bi + 1) * bin_ns
                        y = min(t, edge)
                        watt[bi] += p * (y - x)
                        # depth > 0 means at least one kernel is resident, so
                        # this stretch counts once however many overlap.
                        uni[bi] += (y - x)
                        x = y
                cur_w += d
                depth += k
                prev_t = t

            for a, z, _dw, o, wq, k0 in segs:
                if st0 <= k0 < st1:
                    lau[min(n_bins - 1, int((k0 - st0) // bin_ns))] += 1
                x = a
                while x < z:
                    bi = min(n_bins - 1, int((x - st0) // bin_ns))
                    edge = st0 + (bi + 1) * bin_ns
                    y = min(z, edge)
                    dt = y - x
                    busy[bi] += dt
                    warp[bi] += wq * dt
                    if o is not None:
                        occ_n[bi] += o * dt
                        occ_d[bi] += dt
                    x = y

            n_used += 1
            for i in range(n_bins):
                if occ_d[i]:
                    occ_v[i].append(occ_n[i] / occ_d[i])
                con_v[i].append(busy[i] / bin_ns)
                e_v[i].append((P_IDLE * bin_ns + watt[i]) * 1e-3)
                u_v[i].append(100.0 * min(uni[i], bin_ns) / bin_ns)
                i_v[i].append(max(0.0, bin_ns - uni[i]) / 1000.0)
                d_v[i].append(100.0 * warp[i] / (self.max_warps * bin_ns))
                l_v[i].append(float(lau[i]))

        def _agg(vals, q):
            if not vals:
                return 0.0
            v = sorted(vals)
            return v[max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))]

        dur = rows[0][2] - rows[0][1]
        return dict(
            n_bins=n_bins, n_slots=n_used, slot_us=dur / 1000.0,
            edges_us=[dur / 1000.0 * i / n_bins for i in range(n_bins + 1)],
            occ=[(sum(v) / len(v)) if v else 0.0 for v in occ_v],
            occ_lo=[_agg(v, .05) for v in occ_v],
            occ_hi=[_agg(v, .95) for v in occ_v],
            conc=[(sum(v) / len(v)) if v else 0.0 for v in con_v],
            conc_min=[_agg(v, 0.0) for v in con_v],
            conc_med=[_agg(v, .50) for v in con_v],
            conc_p95=[_agg(v, .95) for v in con_v],
            conc_max=[_agg(v, 1.0) for v in con_v],
            conc_hi=[_agg(v, .95) for v in con_v],
            e_min=[_agg(v, 0.0) for v in e_v],
            # The median matters here even though P95 was what was asked for:
            # under load the modelled power pins to the board limit, so P95 and
            # max are the same flat line and only the median carries shape.
            e_med=[_agg(v, .50) for v in e_v],
            e_p95=[_agg(v, .95) for v in e_v],
            e_max=[_agg(v, 1.0) for v in e_v],
            # Same shape as occ/conc/e above: a mean plus a spread, per bin.
            **{f"{k}_{q}": [f(v) for v in src]
               for k, src in (("util", u_v), ("idle", i_v),
                              ("demand", d_v), ("launch", l_v))
               for q, f in (("mean", lambda v: (sum(v)/len(v)) if v else 0.0),
                            ("min", lambda v: _agg(v, 0.0)),
                            ("med", lambda v: _agg(v, .50)),
                            ("p95", lambda v: _agg(v, .95)),
                            ("max", lambda v: _agg(v, 1.0)))})

    def kernel_window_stats(self, lo: int, hi: int):
        """Per-kernel statistics computed over ONLY the slots in [lo,hi].

        The inventory used to report whole-capture figures while every other
        tab reported the selected window, so the same kernel showed two
        different launch counts on two tabs. Everything is windowed now.

        Returns per kernel: name, channel, role, nsys_visible, kind, occ_pct,
        occ_limiter, res_src, regs, smem, n_launch, n_slots_seen, dur_n,
        dur_min, dur_med, dur_p95, dur_max — durations in ns, from the
        individual launches inside the window.
        """
        n_slots = max(1, hi - lo + 1)
        meta = {r[0]: r for r in self.db.execute(
            "SELECT id,name,channel,role,nsys_visible,kind,occ_pct,"
            "occ_limiter,res_src,regs,smem FROM kernels")}
        durs: dict = {}
        counts: dict = {}
        slots: dict = {}
        for kid, d, slot in self.db.execute(
                "SELECT kernel_id,t1-t0,slot_idx FROM events "
                "WHERE slot_idx BETWEEN ? AND ?", (lo, hi)):
            counts[kid] = counts.get(kid, 0) + 1
            slots.setdefault(kid, set()).add(slot)
            if d is not None:
                durs.setdefault(kid, []).append(d)

        if self.taxonomy_drift:
            meta = {k: (r[0], r[1], self._chan_of.get(r[1], r[2]),
                        self._role_of.get(r[1], r[3])) + tuple(r[4:])
                    for k, r in meta.items()}

        def q(v, p):
            return v[max(0, min(len(v) - 1, int(round(p * (len(v) - 1)))))]

        out = []
        for kid, n in counts.items():
            m = meta.get(kid)
            if not m:
                continue
            v = sorted(durs.get(kid, []))
            out.append((m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
                        m[9], m[10], n, len(slots.get(kid, ())),
                        len(v),
                        q(v, 0.0) if v else None, q(v, .50) if v else None,
                        q(v, .95) if v else None, q(v, 1.0) if v else None,
                        n / n_slots))
        out.sort(key=lambda r: -r[10])
        return out

    def kernel_slot_durations(self, lo: int, hi: int):
        """Per-(kernel, slot) busy time over a slot range.

        The unit the statistics tab needs for "per-slot duration": a kernel
        launched three times in one slot contributes one row of their sum, so
        the distribution is over SLOTS, not over launches.
        """
        rows = self.db.execute(
            "SELECT k.name,k.channel,k.occ_pct,k.occ_limiter,k.res_src,"
            "       k.regs,k.smem,e.slot_idx,"
            "       SUM(COALESCE(e.t1-e.t0,0)),COUNT(*) "
            "FROM events e JOIN kernels k ON k.id=e.kernel_id "
            "WHERE e.slot_idx BETWEEN ? AND ? "
            "GROUP BY k.id,e.slot_idx", (lo, hi)).fetchall()
        if self.taxonomy_drift:
            rows = [(r[0], self._chan_of.get(r[0], r[1])) + tuple(r[2:])
                    for r in rows]
        return rows

    def occupancy_series(self, lo: int, hi: int):
        """(idx, warp_demand_pct, gpu_busy_pct) per slot.

        GPU busy    = union of kernel intervals / slot length. The fraction of
                      the slot in which the GPU had ANY work resident. Bounded
                      by 100% and, at 20 cells, pinned there.

        Warp demand = warp-nanoseconds / (max resident warps x slot length).
                      DELIBERATELY NOT CAPPED. Above 100% means the concurrent
                      kernel mix asked for more warps than the machine can hold
                      at once, so blocks queued rather than ran — which is the
                      quantity that actually separates a comfortable slot from
                      an overloaded one once GPU-busy has saturated. Measured
                      on the 20-cell capture: 56%-195% across neighbouring
                      slots while GPU busy sat at 100% throughout.

        Both use requested warps (blocks x ceil(tpb/32)), not warps limited by
        register or shared-memory pressure, so this is demand for the machine
        rather than achieved occupancy.
        """
        out = []
        for r in self.slot_stats(lo, hi):
            dur = max(1, r[2] - r[1])
            out.append((r[0],
                        100.0 * r[8] / (self.max_warps * dur),
                        min(100.0, 100.0 * r[7] / dur)))
        return out

    # -- the hot path -------------------------------------------------------
    def events_in_slot(self, idx: int):
        return self.db.execute(
            "SELECT t0,t1,kernel_id,src,dur_kind,stream,blocks,tpb,warps,"
            "regs,smem FROM events WHERE slot_idx=? ORDER BY t0",
            (idx,)).fetchall()

    # -- taxonomy drift -----------------------------------------------------
    def _reclassify(self):
        """Re-apply the CURRENT taxonomy to the stored kernel names.

        channel and role are resolved once at ingest and baked into the store,
        so a capture built before a taxonomy correction keeps the old answer
        forever. The PUSCH LDPC decoder cubins were the case that surfaced
        this: a bare "ldpc" substring rule filed them under PDSCH, and every
        capture built while that rule stood still says PDSCH.

        Rather than invalidate the schema and force a multi-minute rebuild of
        every existing capture, re-derive the classification at load. Where it
        moved, the pre-aggregated slot_channel table is stale too, so record
        that and let the aggregate queries fall back to the events.
        """
        self.reclassified = {}
        for kid, (name, ch, role, vis, kind) in list(self.kernels.items()):
            nch, nrole = classify_kernel(name)
            if (nch, nrole) != (ch, role):
                self.kernels[kid] = (name, nch, nrole, vis, kind)
                self.reclassified[name] = (ch, nch)
        self.taxonomy_drift = any(a != b for a, b in self.reclassified.values())
        self._chan_of = {v[0]: v[1] for v in self.kernels.values()}
        self._role_of = {v[0]: v[2] for v in self.kernels.values()}

    # -- capture provenance -------------------------------------------------
    def _load_provenance(self):
        """Work out which tracer actually contributed each kernel's events.

        Read from the src_mask column when the capture was built by an ingest
        that records it, and derived from the event rows otherwise, so captures
        built before src_mask existed report correctly without a rebuild.

        Nothing here consults a hardcoded kernel list. Whether a kernel is
        visible to nsys depends on the run config (device-graph launch hides
        the PUSCH pipeline), so it is a property of the capture and has to be
        measured from it.
        """
        self.n_nsys = int(self._meta.get("n_nsys") or 0)
        self.n_ktrace = int(self._meta.get("n_ktrace") or 0)
        self.kernel_src = {}
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(kernels)")}
        try:
            if "src_mask" in cols:
                rows = self.db.execute(
                    "SELECT id, COALESCE(src_mask,0) FROM kernels")
                self.kernel_src = {i: int(m) for i, m in rows}
            if not any(self.kernel_src.values()):
                rows = self.db.execute(
                    "SELECT kernel_id, SUM(DISTINCT 1 << src) FROM events "
                    "GROUP BY kernel_id")
                self.kernel_src = {i: int(m or 0) for i, m in rows}
        except sqlite3.Error:
            self.kernel_src = {}

    SRC_NAMES = ((1, "nsys"), (2, "KTRACE"))

    def capture_sources(self) -> list:
        """Tracers that actually contributed events, in this capture."""
        out = []
        if self.n_nsys:
            out.append("nsys")
        if self.n_ktrace:
            out.append("KTRACE")
        if not out and self.kernel_src:
            mask = 0
            for m in self.kernel_src.values():
                mask |= m
            out = [n for b, n in self.SRC_NAMES if mask & b]
        return out

    def source_label(self, kid) -> str:
        """"nsys", "KTRACE", "nsys + KTRACE", or "" when the kernel has no
        events at all in this capture."""
        m = self.kernel_src.get(kid, 0)
        return " + ".join(n for b, n in self.SRC_NAMES if m & b)

    def source_counts(self) -> dict:
        """How many distinct kernels each tracer saw, plus the overlap."""
        only_n = only_k = both = 0
        for m in self.kernel_src.values():
            if m & 1 and m & 2:
                both += 1
            elif m & 1:
                only_n += 1
            elif m & 2:
                only_k += 1
        return {"nsys_only": only_n, "ktrace_only": only_k, "both": both}

    def events_in_window(self, t0: int, t1: int, lookback_ns: int = 1_000_000):
        """Every launch overlapping [t0,t1).

        `lookback_ns` exists because the index is on t0: a long-running kernel
        (the GPU-resident pingpong kernel runs ~887us, spanning several slot
        boundaries) can START before the window and still be RUNNING inside it.
        Scanning from t0-lookback catches those without a second index.
        """
        return self.db.execute(
            "SELECT t0,t1,kernel_id,src,dur_kind,stream,blocks,tpb,warps,"
            "regs,smem FROM events "
            "WHERE t0 >= ? AND t0 < ? AND (t1 IS NULL OR t1 > ?) ORDER BY t0",
            (t0 - lookback_ns, t1, t0)).fetchall()

    def phases_in_window(self, t0: int, t1: int):
        return self.db.execute(
            "SELECT t0,t1,name,base FROM phases "
            "WHERE t0 >= ? AND t0 < ? ORDER BY t0",
            (t0 - 5_000_000, t1)).fetchall()

    def phases_in_slot(self, idx: int):
        return self.db.execute(
            "SELECT t0,t1,name,base FROM phases WHERE slot_idx=? ORDER BY t0",
            (idx,)).fetchall()

    def phase_menu(self, limit: int = 60):
        """(base, count, median duration ns) for the phase picker, commonest
        first. Reads the base index, never the range rows."""
        return self.db.execute(
            "SELECT base,COUNT(*) c,AVG(t1-t0) FROM phases "
            "GROUP BY base ORDER BY c DESC LIMIT ?", (limit,)).fetchall()

    def phase_occurrences(self, base: str, limit: int = 5000):
        return self.db.execute(
            "SELECT t0,t1,slot_idx FROM phases WHERE base=? ORDER BY t0 "
            "LIMIT ?", (base, limit)).fetchall()

    def tasks_in_slot(self, idx: int):
        return self.db.execute(
            "SELECT t0,t1,name,sfn,slot,cell,worker,stages FROM tasks "
            "WHERE slot_idx=? ORDER BY t0", (idx,)).fetchall()

    def kernel_histogram(self, lo_slot: int, hi_slot: int):
        """Per-kernel count + busy time over a slot range, aggregated in SQL so
        the UI never materialises the rows."""
        rows = self.db.execute(
            "SELECT k.name,k.channel,COUNT(*),SUM(COALESCE(e.t1-e.t0,0)),"
            "       SUM(e.warps*COALESCE(e.t1-e.t0,0)),e.src "
            "FROM events e JOIN kernels k ON k.id=e.kernel_id "
            "WHERE e.slot_idx BETWEEN ? AND ? "
            "GROUP BY k.name,e.src ORDER BY 4 DESC", (lo_slot, hi_slot)).fetchall()
        if self.taxonomy_drift:
            rows = [(r[0], self._chan_of.get(r[0], r[1])) + tuple(r[2:])
                    for r in rows]
        return rows

    def channel_histogram(self, lo_slot: int, hi_slot: int):
        if not self.taxonomy_drift:
            return self.db.execute(
                "SELECT channel,SUM(n),SUM(busy_ns),SUM(warp_ns) FROM "
                "slot_channel WHERE idx BETWEEN ? AND ? GROUP BY channel "
                "ORDER BY 3 DESC", (lo_slot, hi_slot)).fetchall()
        # slot_channel was aggregated under the taxonomy in force when the
        # store was built, so a reclassified kernel's busy time is still filed
        # under its old channel there. Re-aggregate from the events, which
        # carry kernel_id and are therefore immune to it.
        agg = {}
        for kid, cnt, busy, warp in self.db.execute(
                "SELECT kernel_id,COUNT(*),SUM(COALESCE(t1-t0,0)),"
                "SUM(warps*COALESCE(t1-t0,0)) FROM events "
                "WHERE slot_idx BETWEEN ? AND ? GROUP BY kernel_id",
                (lo_slot, hi_slot)):
            ch = self.kernels.get(kid, (None, "Unknown"))[1]
            a = agg.setdefault(ch, [0, 0, 0])
            a[0] += cnt or 0
            a[1] += busy or 0
            a[2] += warp or 0
        return sorted(((ch, v[0], v[1], v[2]) for ch, v in agg.items()),
                      key=lambda r: -r[2])

    def kernel_table(self):
        rows = self.db.execute(
            "SELECT name,channel,role,nsys_visible,n_launch,dur_n,dur_min,"
            "dur_med,dur_p99,dur_max,occ_pct,occ_limiter,res_src,regs,smem,"
            "kind,parent FROM kernels ORDER BY n_launch DESC").fetchall()
        if self.taxonomy_drift:
            rows = [(r[0], self._chan_of.get(r[0], r[1]),
                     self._role_of.get(r[0], r[2])) + tuple(r[3:])
                    for r in rows]
        return rows

    def stream_map(self):
        return {i: (lbl, role, n) for i, lbl, role, n in
                self.db.execute("SELECT id,label,role,n FROM streams")}

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass

    # -- build --------------------------------------------------------------
    @classmethod
    def open_or_build(cls, capture_dir, out=None, force=False, log=print
                      ) -> "SessionStore":
        d = Path(capture_dir)
        out = Path(out) if out else default_store_path(d)
        if out.exists() and not force:
            try:
                st = cls(out)
                if st._meta.get("schema") == str(SCHEMA_VERSION):
                    log(f"reusing {out.name} ({st.n_events:,} events, "
                        f"{st.n_slots:,} slots)")
                    return st
                st.close()
                log(f"{out.name} is schema {st._meta.get('schema')}, "
                    f"rebuilding for schema {SCHEMA_VERSION}")
            except sqlite3.DatabaseError:
                pass
        build_store(d, out, log=log)
        return cls(out)


def default_store_path(capture_dir: Path) -> Path:
    """Store next to the capture when writable, else in a user cache dir.
    Captures often live on read-only media or a shared mount."""
    d = Path(capture_dir)
    cand = d / f"{d.name}.tti"
    try:
        probe = d / ".tti_write_probe"
        probe.touch()
        probe.unlink()
        return cand
    except OSError:
        cache = Path.home() / ".cache" / "ttiscope"
        cache.mkdir(parents=True, exist_ok=True)
        return cache / f"{d.name}.tti"


# ─────────────────────────────────────────────────────────────────────────────
# Clock calibration
# ─────────────────────────────────────────────────────────────────────────────

def _align_series(a: list, b: list, iters: int = 5, cap: int = 4000) -> tuple:
    """Offset mapping timestamp series `a` onto `b`, plus its residual IQR.

    Pairwise differencing of two SORTED series (a[i] vs b[i]) is wrong here. The two sources do not start together:
    the printf log begins at process start and includes warmup, while nsys
    starts capturing later. A prefix of unmatched launches shifts every
    subsequent pairing, and the resulting "offset" is off by however much
    traffic ran in the gap — which is how encodeRateMatchMultipleSSBsKernel
    scored an 81 ms IQR over 4,280 launches on the 20-cell capture.

    Nearest-neighbour refinement (ICP) does not care about counts or offsets in
    coverage: seed with the difference of medians, then repeatedly snap each
    `a` point to its nearest `b` point and shift by the median residual. Two
    series of the same physical events converge to a residual spread of tens of
    nanoseconds; unrelated series stay milliseconds apart, which is exactly the
    signal needed to rank candidates.
    """
    if len(a) > cap:                       # uniform thinning keeps the span
        step = len(a) / cap
        a = [a[int(i * step)] for i in range(cap)]
    off = b[len(b) // 2] - a[len(a) // 2]
    iqr = None
    for _ in range(iters):
        res = []
        for x in a:
            y = x + off
            i = bisect.bisect_left(b, y)
            best = None
            for j in (i - 1, i):
                if 0 <= j < len(b):
                    d = b[j] - y
                    if best is None or abs(d) < abs(best):
                        best = d
            if best is not None:
                res.append(best)
        if not res:
            return None, None
        res.sort()
        n = len(res)
        off += res[n // 2]
        iqr = res[int(n * .75)] - res[int(n * .25)]
    return int(off), int(iqr)


def _clock_offset(nsys_by: dict, ktrace_by: dict, counts: dict) -> tuple:
    """Map %globaltimer -> nsys timeline by cross-referencing the same run.

    %globaltimer and nsys use unrelated epochs, so one clock has to be
    expressed in the other. Every kernel visible to BOTH sources is a candidate
    calibration pair; each is aligned independently and ranked by the TIGHTEST
    residual spread, not the largest sample. A kernel launched a handful of
    times but recorded identically by both sources is worth more than one
    launched thousands of times whose two records disagree.
    """
    best, considered = None, []
    for kname, pt in ktrace_by.items():
        nt = nsys_by.get(kname)
        # 4 rather than 8: a kernel launched a handful of times at init
        # (warmup_kernel) is unambiguous and uncontended, and on the 2-cell
        # capture its 17 launches aligned to a 5 ns residual — three orders of
        # magnitude better than any steady-state kernel. Small samples are
        # accepted only if they also fit tightly, see the guard below.
        if not nt or len(pt) < 4 or len(nt) < 4:
            continue
        # A ktrace series truncated at CALIB_CAP no longer spans the capture,
        # so its median seed is biased toward the start.
        if counts.get(kname, len(pt)) > CALIB_CAP:
            continue
        off, iqr = _align_series(sorted(pt), sorted(nt))
        if off is None:
            continue
        n = min(len(pt), len(nt))
        considered.append((iqr, off, kname, n))
        # A short series can align tightly by luck. Trust it only if it aligns
        # to well under a microsecond; otherwise it needs the sample count.
        if n < 8 and iqr > 1_000:
            continue
        if best is None or iqr < best[0]:
            best = considered[-1]
    if not best:
        return None, None, None, 0, considered
    iqr, off, kname, n = best
    return int(off), int(iqr), kname, n, sorted(considered)


# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Build:
    st: sqlite3.Connection
    kid: dict = field(default_factory=dict)
    launches: dict = field(default_factory=dict)

    def kernel_id(self, name: str) -> int:
        i = self.kid.get(name)
        if i is None:
            i = len(self.kid) + 1
            ch, role = classify_kernel(name)
            self.st.execute(
                "INSERT INTO kernels(id,name,channel,role,nsys_visible,"
                "n_launch,dur_n) VALUES(?,?,?,?,?,0,0)",
                (i, name, ch, role, 0 if name in KTRACE_ONLY else 1))
            self.kid[name] = i
        self.launches[i] = self.launches.get(i, 0) + 1
        return i


def _read_ktrace(path: Path, b: _Build, log) -> tuple:
    """Stream the KTRACE log into a raw staging table.

    One pass. Rows land with RAW %globaltimer stamps; the offset is not known
    until the nsys side has been read, so it is applied later with a single
    SQL UPDATE rather than by holding a million tuples in RAM.
    """
    b.st.execute("CREATE TABLE ktrace_raw (t0 INTEGER, kernel_id INTEGER, "
                 "blocks INTEGER, tpb INTEGER, warps INTEGER)")
    samples: dict[str, list] = {}
    counts: dict[str, int] = {}
    durations: dict[str, list] = {}
    batch, n = [], 0
    log(f"  reading {path.name} ({path.stat().st_size/1e6:.0f} MB{' gzipped' if path.suffix == '.gz' else ''})")
    _open = gzip.open if path.suffix == ".gz" else open
    with _open(path, "rt", errors="replace") as fh:
        for line in fh:
            if "KTRACE t0=" in line:
                for m in KTRACE_RE.finditer(line):
                    name = short_kernel_name(m.group(8))
                    if not IDENT_RE.match(name):
                        continue
                    t0 = int(m.group(1))
                    blocks = int(m.group(2)) * int(m.group(3)) * int(m.group(4))
                    tpb = int(m.group(5)) * int(m.group(6)) * int(m.group(7))
                    batch.append((t0, b.kernel_id(name), blocks, tpb,
                                  warps_of(blocks, tpb)))
                    counts[name] = counts.get(name, 0) + 1
                    s = samples.setdefault(name, [])
                    if len(s) < CALIB_CAP:
                        s.append(t0)
                    n += 1
                if len(batch) >= 100_000:
                    b.st.executemany("INSERT INTO ktrace_raw VALUES(?,?,?,?,?)",
                                     batch)
                    batch.clear()
            elif "KERNEL_DURATION" in line:
                for m in KDUR_RE.finditer(line):
                    durations.setdefault(m.group(1), []).append(int(m.group(2)))
    if batch:
        b.st.executemany("INSERT INTO ktrace_raw VALUES(?,?,?,?,?)", batch)
    log(f"    {n:,} KTRACE launches, {len(counts)} kernels, "
        f"{sum(len(v) for v in durations.values()):,} duration samples "
        f"({len(durations)} kernels)")
    return n, samples, counts, durations


def _read_nvlog(path: Path, log) -> tuple:
    """Stream cuPHY's nvlog for the host-side TTI picture.

    Returns (ticks, tasks). Ticks are the 500us slot grid with real SFN.slot;
    tasks are the {TI} pipeline records, each already carrying absolute ns
    stage timestamps.
    """
    ticks, tasks, misses, mps = [], [], [], []
    cells, cell_ids = set(), set()
    # Most recent slot boundary seen in the stream. TTI-MISS-RECOVERY lines
    # carry no tick= of their own (81 of 104 in a 20C/59c capture), so this is
    # the only thing that says which SFN wrap cycle they belong to.
    last_tick = None
    log(f"  reading {path.name} ({path.stat().st_size/1e6:.0f} MB{' gzipped' if path.suffix == '.gz' else ''})")
    _open = gzip.open if path.suffix == ".gz" else open
    with _open(path, "rt", errors="replace") as fh:
        for line in fh:
            if "cell_id" in line and "nic_index" in line:
                m = CELL_RE.search(line)
                if m:
                    cell_ids.add(int(m.group(1)))
            elif "MPS context with" in line:
                m = MPS_CTX_RE.search(line)
                if m:
                    mps.append((m.group(1).strip(), int(m.group(2))))
            elif "TICK_TIMES" in line:
                m = TICK_RE.search(line)
                if m:
                    ticks.append((int(m.group(4)), int(m.group(1)),
                                  int(m.group(2))))
                    last_tick = ticks[-1][0]
            elif "TTI-MISS" in line:
                m = MISS_RE.search(line)
                if m:
                    dt = MISS_DT_RE.search(line)
                    tk = MISS_TICK_RE.search(line)
                    st = MISS_SITE_RE.search(line)
                    misses.append((int(m.group("n")),
                                   "recovery" if m.group("rec") else "cleanup",
                                   int(m.group("sfn")), int(m.group("slot")),
                                   int(tk.group(1)) if tk else None,
                                   int(dt.group(1)) if dt else None,
                                   st.group(1) if st else "",
                                   last_tick))
            elif "{TI}" in line:
                m = TI_RE.search(line)
                if not m:
                    continue
                stages = TI_STAGE_RE.findall(m.group(7))
                if not stages:
                    continue
                ts = [int(v) for _, v in stages]
                cells.add(int(m.group(6)))
                tasks.append((min(ts), max(ts), m.group(1).strip(),
                              int(m.group(2)), int(m.group(3)),
                              int(m.group(6)), int(m.group(5)),
                              ";".join(f"{k.strip()}:{int(v)-ts[0]}"
                                       for k, v in stages)))
    log(f"    {len(ticks):,} slot ticks, {len(tasks):,} host task records, "
        f"{len(cell_ids)} cells configured, {len(cells)} cell group(s)")
    if misses:
        nc = sum(1 for m in misses if m[1] == "cleanup")
        log(f"    {len(misses):,} TTI MISSES  ({nc} cleanup_err, "
            f"{len(misses)-nc} L1-recovery)")
    return ticks, tasks, len(cell_ids), misses, mps


def read_mps_contexts(path: Path) -> list:
    """[(label, sm_cap), ...] in creation order from a cuphycontroller log.

    The ctrl_stdout.log is small and the preferred source; the nvlog carries
    the same lines and is the fallback. A DU that re-initialised writes the
    block twice, so a label seen again replaces its earlier cap in place.
    """
    out: list = []
    _open = gzip.open if path.suffix == ".gz" else open
    try:
        with _open(path, "rt", errors="replace") as fh:
            for line in fh:
                if "MPS context with" in line:
                    m = MPS_CTX_RE.search(line)
                    if m:
                        out.append((m.group(1).strip(), int(m.group(2))))
    except OSError:
        return []
    return _dedupe_mps(out)


def _dedupe_mps(rows: list) -> list:
    order, cap = [], {}
    for label, n in rows:
        if label not in cap:
            order.append(label)
        cap[label] = n
    return [(label, cap[label]) for label in order]


def _read_nsys(sq: sqlite3.Connection, b: _Build, log) -> tuple:
    """Stream nsys kernels + NVTX + device info. Never fetchall()s KERNEL.

    Deliberately does NOT touch CUPTI_ACTIVITY_KIND_RUNTIME: on the 20-cell
    capture that table has 32.7M rows, and joining it makes loading
    impossible.
    """
    info = {}
    try:
        cols = [c[1] for c in sq.execute("PRAGMA table_info(TARGET_INFO_GPU)")]
        row = sq.execute("SELECT * FROM TARGET_INFO_GPU LIMIT 1").fetchone()
        gi = dict(zip(cols, row)) if row else {}
        info["gpu_name"] = gi.get("name", "Unknown GPU")
        info["sm_count"] = gi.get("smCount") or 132
        wps = gi.get("maxWarpsPerSm") or 64
        info["max_warps"] = int(info["sm_count"]) * int(wps)
        # The whole per-SM capacity table, straight from the capture. Without
        # these the occupancy calculator falls back to a hardcoded sm_90 table,
        # which is right for GH200 and wrong for anything else - a Blackwell
        # capture would be scored against Hopper's limits and every spare-GPU
        # figure derived from it would be wrong. nsys has always recorded them;
        # they were simply never read.
        for key, col in (("max_warps_per_sm", "maxWarpsPerSm"),
                         ("max_blocks_per_sm", "maxBlocksPerSm"),
                         ("regs_per_sm", "maxRegistersPerSm"),
                         ("smem_per_sm", "maxShmemPerSm"),
                         ("smem_per_block_optin", "maxShmemPerBlockOptin"),
                         ("warp_size", "threadsPerWarp"),
                         ("max_threads_per_block", "maxThreadsPerBlock"),
                         ("compute_major", "computeMajor"),
                         ("compute_minor", "computeMinor"),
                         ("total_memory", "totalMemory"),
                         ("chip_name", "chipName")):
            v = gi.get(col)
            if v not in (None, ""):
                info[key] = v
    except sqlite3.OperationalError:
        info.update(gpu_name="Unknown GPU", sm_count=132, max_warps=132 * 64)
    try:
        r = sq.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME "
                       "LIMIT 1").fetchone()
        info["utc_epoch_ns"] = int(r[0]) if r and r[0] else None
    except sqlite3.OperationalError:
        info["utc_epoch_ns"] = None

    nsys_by: dict[str, list] = {}
    stream_n: dict[int, int] = {}
    stream_kernels: dict[int, set] = {}
    # Per-context launch counts by channel, for mapping MPS log lines onto
    # contextIds and verifying the mapping (see _write_contexts).
    ctx_chan: dict[int, dict] = {}
    batch, n = [], 0
    kcols = {c[1] for c in sq.execute(
        "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}

    def col(name, dflt="NULL"):
        return f"k.{name}" if name in kcols else dflt
    cur = sq.execute(
        "SELECT k.start,k.end,s.value,k.streamId,"
        "       k.gridX*k.gridY*k.gridZ,k.blockX*k.blockY*k.blockZ,"
        "       k.registersPerThread,"
        "       k.staticSharedMemory+k.dynamicSharedMemory,"
        f"      {col('contextId')},{col('greenContextId')},"
        f"      {col('sharedMemoryExecuted')} "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName=s.id ORDER BY k.start")
    ins = "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)"
    for (start, end, raw, stream, blocks, tpb, regs, smem,
         ctx, gctx, smem_cfg) in cur:
        name = short_kernel_name(raw)
        kid = b.kernel_id(name)
        batch.append((start, end, kid, SRC_NSYS, DUR_MEASURED, stream,
                      blocks, tpb, warps_of(blocks, tpb), regs, smem, None,
                      ctx, gctx or None, smem_cfg))
        nsys_by.setdefault(name, []).append(start)
        stream_n[stream] = stream_n.get(stream, 0) + 1
        stream_kernels.setdefault(stream, set()).add(name)
        key = gctx or ctx
        if key is not None:
            ch = ctx_chan.setdefault(key, {})
            c0 = classify_kernel(name)[0]
            ch[c0] = ch.get(c0, 0) + 1
        n += 1
        if len(batch) >= 100_000:
            b.st.executemany(ins, batch)
            batch.clear()
    if batch:
        b.st.executemany(ins, batch)
    info["ctx_chan"] = ctx_chan
    # Thread-block clusters (sm_90+) are co-scheduled across a GPC, which the
    # per-SM mean field in ttiscope_spare does not model. Recorded so the
    # spare analysis can refuse to pretend otherwise if they ever appear.
    if "clusterX" in kcols:
        r = sq.execute("SELECT MAX(MAX(clusterX,1)*MAX(clusterY,1)*"
                       "MAX(clusterZ,1)) FROM CUPTI_ACTIVITY_KIND_KERNEL"
                       ).fetchone()
        info["max_cluster"] = int(r[0] or 1)
    try:
        ccols = [c[1] for c in sq.execute(
            "PRAGMA table_info(TARGET_INFO_CUDA_CONTEXT_INFO)")]
        info["ctx_info"] = [dict(zip(ccols, r)) for r in sq.execute(
            "SELECT * FROM TARGET_INFO_CUDA_CONTEXT_INFO")]
    except sqlite3.OperationalError:
        info["ctx_info"] = []
    log(f"    {n:,} nsys kernel launches, {len(nsys_by)} kernels, "
        f"{len(stream_n)} streams")

    for sid, names in sorted(stream_kernels.items()):
        chans = {classify_kernel(x)[0] for x in names}
        for pref in ("FH-DL", "FH-UL", "PDSCH", "PDCCH", "PUSCH", "PRACH",
                     "PUCCH", "CSI-RS", "SSB"):
            if pref in chans:
                role = pref
                break
        else:
            role = sorted(chans)[0] if chans else "other"
        b.st.execute("INSERT INTO streams VALUES(?,?,?,?)",
                     (sid, f"Stream {sid}\n({role})", role, stream_n[sid]))

    n_ph = 0
    try:
        rows = sq.execute(
            "SELECT COALESCE(e.text,s.value),e.start,e.end FROM NVTX_EVENTS e "
            "LEFT JOIN StringIds s ON e.textId=s.id "
            "WHERE e.end IS NOT NULL AND e.end > e.start ORDER BY e.start")
        batch = []
        for txt, s, e in rows:
            # 15,170 ranges in the 20-cell capture carry a binary payload
            # rather than a label. They are real ranges, but they have no name
            # to show and would otherwise appear as thousands of bogus
            # "distinct phases" in the phase menu.
            if (txt is None or not txt.strip() or not txt.isascii()
                    or not txt.isprintable()):
                continue
            batch.append((s, e, str(txt), phase_base(txt), None))
            n_ph += 1
            if len(batch) >= 100_000:
                b.st.executemany("INSERT INTO phases VALUES(?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            b.st.executemany("INSERT INTO phases VALUES(?,?,?,?,?)", batch)
    except sqlite3.OperationalError as ex:
        log(f"    WARNING: NVTX read stopped after {n_ph:,} rows: {ex}")
    names = b.st.execute("SELECT COUNT(DISTINCT base) FROM phases").fetchone()[0]
    log(f"    {n_ph:,} NVTX phase ranges, {names} distinct phases")
    return n, nsys_by, info


def _traced_span(b: _Build):
    """The [t0,t1] the event trace actually covers, in nsys ns, or None.

    cuPHY's nvlog runs for the entire DU session while nsys profiles a window
    inside it - 112 s of log against 15 s of trace is typical for the A2
    captures. Everything derived from the nvlog has to be clipped to this, or
    the session is mostly empty grid: slots with no kernels, host tasks with
    no launches to sit beside, and a timeline whose scale is set by a span
    nothing was recorded in.
    """
    row = b.st.execute(
        "SELECT MIN(t0),MAX(COALESCE(t1,t0)) FROM events").fetchone()
    if not row or row[0] is None:
        return None                        # nvlog-only capture: keep it all
    # One slot period of margin so the boundary slots stay whole.
    return row[0] - 500_000, row[1] + 500_000


def _build_slots(b: _Build, ticks, epoch, log) -> int:
    """The TTI grid.

    Priority:
      1. L2A.TICK_TIMES from nvlog — the DU's own 500us slot boundary, with
         real SFN.slot. Requires the nsys UTC epoch to place on the timeline.
      2. NVTX SLOT_DL ranges — GPU-side ground truth, no SFN.
      3. Fixed 500us tiling over the event span.
    Slot boundaries are never guessed from memset_kernel gaps: both real
    sources outrank that heuristic, which yields a different slot count than
    either.
    """
    if ticks and epoch:
        ticks = sorted(set(ticks))
        span = _traced_span(b)
        rows, dropped = [], 0
        for i, (wall, sfn, slot) in enumerate(ticks):
            t0 = wall - epoch
            t1 = (ticks[i + 1][0] - epoch) if i + 1 < len(ticks) else t0 + 500_000
            if span and (t1 < span[0] or t0 > span[1]):
                dropped += 1
                continue
            # Index by position in the kept sequence, not in the tick list, so
            # slot indices stay contiguous from 0.
            rows.append((len(rows), t0, t1, sfn, slot, "tick"))
        b.st.executemany("INSERT INTO slots VALUES(?,?,?,?,?,?)", rows)
        b.st.execute("CREATE INDEX idx_slots_t0 ON slots(t0)")
        log(f"    {len(rows):,} slots from L2A.TICK_TIMES (SFN.slot known)")
        if dropped:
            log(f"      {dropped:,} ticks lie outside the {(span[1]-span[0])/1e9:.1f} s "
                f"the trace covers - the nvlog outlives the nsys window")
        return len(rows)

    dl = b.st.execute("SELECT t0,t1 FROM phases WHERE name='SLOT_DL' "
                      "ORDER BY t0").fetchall()
    if len(dl) >= 2:
        rows = [(i, s, dl[i + 1][0] if i + 1 < len(dl) else e, None, None, "nvtx")
                for i, (s, e) in enumerate(dl)]
        b.st.executemany("INSERT INTO slots VALUES(?,?,?,?,?,?)", rows)
        b.st.execute("CREATE INDEX idx_slots_t0 ON slots(t0)")
        log(f"    {len(rows):,} slots from NVTX SLOT_DL (no SFN available)")
        return len(rows)

    span = b.st.execute("SELECT MIN(t0),MAX(COALESCE(t1,t0)) FROM events").fetchone()
    if not span or span[0] is None:
        return 0
    t0, t1 = span
    w = 500_000
    n = max(1, (t1 - t0 + w - 1) // w)
    b.st.executemany("INSERT INTO slots VALUES(?,?,?,?,?,?)",
                     [(i, t0 + i * w, t0 + (i + 1) * w, None, None, "inferred")
                      for i in range(n)])
    b.st.execute("CREATE INDEX idx_slots_t0 ON slots(t0)")
    log(f"    {n:,} slots by fixed {w/1000:.0f}us tiling (no markers found)")
    return n


def _assign_slots(b: _Build, log) -> None:
    """Attach every event / phase / task to the slot containing its start.

    Done in SQL with a correlated subquery over the indexed slots table rather
    than in Python: at 1.7M events, a per-row bisect in Python costs minutes
    and a temporary list the size of the capture.
    """
    for tbl in ("events", "phases", "tasks"):
        b.st.execute(
            f"UPDATE {tbl} SET slot_idx = ("
            "  SELECT s.idx FROM slots s WHERE s.t0 <= "
            f"{tbl}.t0 ORDER BY s.t0 DESC LIMIT 1)")
    log("    slot attribution done")


def _compute_occupancy(b: _Build, log) -> None:
    """Theoretical occupancy per launch, plus a per-kernel summary.

    Occupancy depends only on (threads/block, registers/thread, shared bytes),
    so it is computed once per DISTINCT launch shape rather than per launch —
    on the 20-cell capture that is 91 shapes covering 937,789 events.

    nsys rows carry CUPTI's actual register and shared-memory figures. KTRACE
    rows carry neither (device printf cannot report them) and fall back to the
    cuobjdump table, whose values are the maximum across template
    instantiations — so those occupancies are a LOWER bound, and res_src
    records that so the two are never conflated in a report.
    """
    static = load_static_resources()
    if not static:
        log("    WARNING: no all_kernel_resources_sm90.csv found (set "
            "TTISCOPE_KERNEL_RESOURCES or place it in ~/.config/tti-scope/); "
            "kernels only KTRACE can see will have no occupancy figure")

    names = {i: n for n, i in b.kid.items()}

    # Classify every traced name as a real kernel or a device function BEFORE
    # computing occupancy: a device function has no launch configuration of
    # its own — the grid and block it reports are its parent's — so giving it
    # an occupancy figure would be inventing one.
    has_nsys = {r[0] for r in b.st.execute(
        f"SELECT DISTINCT kernel_id FROM events WHERE src={SRC_NSYS}")}
    n_devfn = 0
    for kid, name in names.items():
        if kid in has_nsys or name in static:
            kind, parent = "kernel", None      # CUPTI or cuobjdump says so
        elif name in DEVICE_FUNCTIONS:
            kind, parent = "device_fn", DEVICE_FUNCTIONS[name]
        elif static:
            # The cuobjdump table lists every __global__ entry point in the
            # built libraries. Absent from it, and never seen by CUPTI, means
            # this is not a launch.
            kind, parent = "device_fn", None
        else:
            kind, parent = "unknown", None
        if kind == "device_fn":
            n_devfn += 1
        b.st.execute("UPDATE kernels SET kind=?,parent=? WHERE id=?",
                     (kind, parent, kid))
    if n_devfn:
        log(f"    {n_devfn} traced name(s) are __device__ functions, not "
            f"kernels — excluded from launch counts")

    shapes = b.st.execute(
        "SELECT DISTINCT e.kernel_id,e.src,e.tpb,e.regs,e.smem FROM events e "
        "JOIN kernels k ON k.id=e.kernel_id "
        "WHERE k.kind != 'device_fn'").fetchall()
    n_upd = 0
    for kid, src, tpb, regs, smem in shapes:
        if src == SRC_NSYS:
            r, s, why = regs, smem, "cupti"
        else:
            r, s = static.get(names.get(kid, ""), (None, None))
            why = "cuobjdump" if r is not None else "unknown"
        if r is None or not tpb:
            continue
        occ, _blk, _w, lim = theoretical_occupancy(tpb, r, s)
        b.st.execute(
            "UPDATE events SET occ=? WHERE kernel_id=? AND src=? AND tpb=? "
            "AND regs IS ? AND smem IS ?",
            (occ, kid, src, tpb, regs, smem))
        n_upd += 1

    # Per-kernel headline figure: the shape it spends the most TIME in, not the
    # one it launches most often — a rare long launch dominates a slot budget.
    for kid, name in names.items():
        if name in DEVICE_FUNCTIONS or (static and name not in static
                                        and kid not in has_nsys):
            continue
        row = b.st.execute(
            "SELECT tpb,regs,smem,src,SUM(COALESCE(t1-t0,0)) tot,COUNT(*) n "
            "FROM events WHERE kernel_id=? GROUP BY tpb,regs,smem,src "
            "ORDER BY tot DESC, n DESC LIMIT 1", (kid,)).fetchone()
        if not row:
            continue
        tpb, regs, smem, src, _tot, _n = row
        if src == SRC_NSYS:
            r, s, why = regs, smem, "cupti"
        else:
            r, s = static.get(name, (None, None))
            why = "cuobjdump" if r is not None else "unknown"
        if r is None or not tpb:
            b.st.execute("UPDATE kernels SET res_src='unknown' WHERE id=?",
                         (kid,))
            continue
        occ, _blk, _w, lim = theoretical_occupancy(tpb, r, s)
        b.st.execute("UPDATE kernels SET regs=?,smem=?,res_src=?,occ_pct=?,"
                     "occ_limiter=? WHERE id=?", (r, s, why, occ, lim, kid))
    log(f"    theoretical occupancy over {n_upd} distinct launch shapes")


def _rollup(b: _Build, max_warps: int, log) -> None:
    """Precompute per-slot and per-slot-per-channel aggregates.

    Two things this has to get right, both of which a naive GROUP BY gets
    wrong:

    * A launch is attributed to the slot it STARTS in, which is correct for
      counting launches but wrong for accounting time. The GPU-resident
      pingpong kernel runs ~887us and covers two whole 500us slots; charging
      all of it to its start slot leaves the slots it actually occupied
      looking idle. Every launch is therefore SPLIT across the slots it spans.

    * `busy_ns` sums concurrent kernels, so with 24 streams it routinely
      exceeds the slot length. `span_ns` is the UNION of the busy intervals —
      the fraction of the slot in which the GPU had ANY work resident — and is
      the only one of the two that can be read as a percentage.
    """
    slot_t0 = [r[0] for r in b.st.execute("SELECT t0 FROM slots ORDER BY idx")]
    slot_t1 = [r[0] for r in b.st.execute("SELECT t1 FROM slots ORDER BY idx")]
    n_slot = len(slot_t0)
    if not n_slot:
        return
    busy = [0] * n_slot
    warpns = [0] * n_slot
    occnum = [0.0] * n_slot     # SUM occ * busy  -> time-weighted occupancy
    occden = [0] * n_slot       # busy time that HAD an occupancy figure

    names = {i: n for n, i in b.kid.items()}
    dynw_cache: dict = {}

    b.st.execute("CREATE TABLE seg (slot INTEGER, a INTEGER, z INTEGER, "
                 "dw REAL)")
    cur = b.st.execute(
        "SELECT slot_idx,t0,t1,warps,kernel_id,occ FROM events "
        "WHERE slot_idx IS NOT NULL AND t1 IS NOT NULL AND t1 > t0 ORDER BY t0")
    batch = []
    for s0, t0, t1, warps, kid, occ in cur:
        # Requested warps are capped at the machine: one launch cannot occupy
        # more of the GPU than exists, however many blocks it asks for.
        w = min(int(warps or 0), max_warps)
        key = (kid, w)
        dw = dynw_cache.get(key)
        if dw is None:
            dw = dynamic_watts(names.get(kid, ""), w, max_warps)
            dynw_cache[key] = dw
        s = s0
        while s < n_slot and slot_t0[s] < t1:
            a = max(t0, slot_t0[s])
            z = min(t1, slot_t1[s])
            if z > a:
                d = z - a
                busy[s] += d
                warpns[s] += w * d
                if occ is not None:
                    occnum[s] += occ * d
                    occden[s] += d
                batch.append((s, a, z, dw))
            s += 1
        if len(batch) >= 200_000:
            b.st.executemany("INSERT INTO seg VALUES(?,?,?,?)", batch)
            batch.clear()
    if batch:
        b.st.executemany("INSERT INTO seg VALUES(?,?,?,?)", batch)

    # n_kernels counts LAUNCHES, so device-function trace records are held
    # separately rather than inflating it.
    b.st.execute(
        "INSERT INTO slot_stats(idx,n_kernels,n_nsys,n_ktrace,busy_ns,"
        "span_ns,warp_ns,n_streams,idle_ns,energy_uj,occ_w,occ_cov,n_devfn) "
        "SELECT e.slot_idx,"
        "       SUM(k.kind IS NOT 'device_fn'),"
        "       SUM(e.src=0),"
        "       SUM(e.src=1 AND k.kind IS NOT 'device_fn'),0,0,0,"
        "       COUNT(DISTINCT e.stream),0,0,0,0,"
        "       SUM(k.kind IS 'device_fn') "
        "FROM events e JOIN kernels k ON k.id=e.kernel_id "
        "WHERE e.slot_idx IS NOT NULL GROUP BY e.slot_idx")
    # ── one sweep per slot: busy union AND capped energy ───────────────────
    #
    # Summing each kernel's power over its own duration is what a first cut
    # does, and it is wrong once kernels overlap: on the 20-cell capture it
    # produced a mean of 892 W on a 700 W part. A GPU does not draw the sum of
    # what its concurrent kernels would each draw alone — it draws what the
    # board allows, and the hardware power limit is the ceiling.
    #
    # So instantaneous dynamic power is swept properly: every segment
    # contributes +dw at its start and -dw at its end, the running sum is
    # clamped to the dynamic budget (TDP - idle), and that is integrated. The
    # same walk yields the union of busy intervals, since the sum is non-zero
    # exactly when at least one kernel is resident.
    dyn_cap = max(0.0, P_TDP - P_IDLE)
    spans, energy = [], []

    def _sweep(slot, segs):
        ev = []
        for a, z, dw in segs:
            ev.append((a, dw))
            ev.append((z, -dw))
        ev.sort()
        cur_w, prev_t, span, wattns = 0.0, None, 0, 0.0
        depth = 0
        for t, d in ev:
            if prev_t is not None and t > prev_t:
                dt = t - prev_t
                if depth > 0:
                    span += dt
                    wattns += min(cur_w, dyn_cap) * dt
            cur_w += d
            depth += 1 if d > 0 else -1
            prev_t = t
        spans.append((span, slot))
        energy.append((slot_energy_uj(slot_t1[slot] - slot_t0[slot], wattns),
                       slot))

    cur_slot, segs = None, []
    for slot, a, z, dw in b.st.execute(
            "SELECT slot,a,z,dw FROM seg ORDER BY slot,a"):
        if slot != cur_slot:
            if cur_slot is not None:
                _sweep(cur_slot, segs)
            cur_slot, segs = slot, []
        segs.append((a, z, dw))
    if cur_slot is not None:
        _sweep(cur_slot, segs)

    b.st.executemany("UPDATE slot_stats SET span_ns=? WHERE idx=?", spans)
    b.st.executemany("UPDATE slot_stats SET energy_uj=? WHERE idx=?", energy)
    b.st.execute("DROP TABLE seg")

    upd = [(busy[i], warpns[i],
            (occnum[i] / occden[i]) if occden[i] else None,
            (occden[i] / busy[i]) if busy[i] else 0.0, i)
           for i in range(n_slot) if busy[i] or warpns[i]]
    b.st.executemany("UPDATE slot_stats SET busy_ns=?,warp_ns=?,occ_w=?,"
                     "occ_cov=? WHERE idx=?", upd)
    # Idle is what is left of the slot once the union of busy intervals is
    # removed — the complement of GPU-busy, not of the summed durations.
    b.st.execute(
        "UPDATE slot_stats SET idle_ns = MAX(0, "
        "  (SELECT s.t1-s.t0 FROM slots s WHERE s.idx=slot_stats.idx) "
        "  - COALESCE(span_ns,0))")

    b.st.execute(
        "INSERT INTO slot_channel(idx,channel,n,busy_ns,warp_ns) "
        "SELECT e.slot_idx,k.channel,COUNT(*),"
        "       SUM(COALESCE(e.t1-e.t0,0)),"
        "       SUM(MIN(e.warps,?)*COALESCE(e.t1-e.t0,0)) "
        "FROM events e JOIN kernels k ON k.id=e.kernel_id "
        "WHERE e.slot_idx IS NOT NULL GROUP BY e.slot_idx,k.channel",
        (max_warps,))

    b.st.execute("UPDATE kernels SET n_launch=COALESCE((SELECT COUNT(*) "
                 "FROM events e WHERE e.kernel_id=kernels.id),0)")
    log(f"    rolled up {len(spans):,} slots")


def export_nsys_rep(rep: Path, log=print) -> Optional[Path]:
    """Run `nsys export` to produce the sqlite a .nsys-rep does not contain.

    A sweep archives the .nsys-rep, not the export, so most captures arrive
    without the one file the GPU timeline needs. Doing it here rather than
    telling the user to is worth it: the export is mechanical, takes a couple
    of minutes, and without it the capture silently loses every host-launched
    kernel AND the wall-clock anchor that gives slots their SFN.slot names.
    """
    import shutil
    import subprocess
    exe = shutil.which("nsys")
    if not exe:
        return None
    out = rep.with_suffix(".sqlite")
    log(f"  exporting {rep.name} -> {out.name} (nsys export, one-time)")
    t0 = time.time()
    try:
        r = subprocess.run([exe, "export", "--type", "sqlite", "--force-overwrite",
                            "true", "-o", str(out), str(rep)],
                           capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as ex:
        log(f"    export failed: {ex}")
        return None
    if r.returncode != 0 or not out.exists():
        log(f"    export failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout or '').strip()[:300]}")
        return None
    log(f"    exported {out.stat().st_size/1e9:.1f} GB in {time.time()-t0:.0f}s")
    return out



def _write_tti_miss(b, misses, epoch, log):
    """Persist the TTI-miss rows, each pinned to the one slot it happened in.

    SFN.slot is how the `slots` table is keyed, but it is not a unique key:
    SFN wraps at 1024 frames, so the pair recurs every 10.24 s and any capture
    longer than that contains several slots answering to the same SFN.slot.
    Resolution therefore needs a timestamp as well as the pair:

      * cleanup markers carry `tick=<ns>`, which is exactly the slot's own
        boundary - an exact match against slots.t0;
      * recovery markers carry no tick, so they are anchored to the last
        TICK_TIMES line seen before them in the stream and resolved to the
        nearest candidate.

    Rows the capture window does not cover keep slot_idx NULL: a miss the
    trace did not span is still a miss, and dropping it would understate the
    count the paper reports.
    """
    if not misses:
        return 0
    # (sfn,slot) -> [(t0, idx), ...] in time order, one entry per wrap cycle.
    cand: dict = {}
    for idx, t0, sfn, slot in b.st.execute(
            "SELECT idx,t0,sfn,slot FROM slots "
            "WHERE sfn IS NOT NULL ORDER BY t0"):
        cand.setdefault((sfn, slot), []).append((t0, idx))

    rows, resolved, ambiguous, uncovered = [], 0, 0, 0
    for seq, kind, sfn, slot, tick_ns, dt_ns, site, anchor in misses:
        c = cand.get((sfn, slot))
        idx = None
        if c and epoch:
            ref = tick_ns if tick_ns is not None else anchor
            if ref is not None:
                ref -= epoch
                idx = min(c, key=lambda r: abs(r[0] - ref))[1]
                resolved += 1
            else:
                # No timestamp at all. Guessing a wrap cycle would put a red
                # cross on a slot that was very likely fine, so decline.
                idx = c[0][1] if len(c) == 1 else None
                ambiguous += 1
        else:
            uncovered += 1
        rows.append((seq, kind, sfn, slot, tick_ns, dt_ns, site, idx))

    b.st.executemany("INSERT INTO tti_miss (seq,kind,sfn,slot,tick_ns,dt_ns,"
                     "site,slot_idx) VALUES (?,?,?,?,?,?,?,?)", rows)
    n = len(rows)
    placed = sum(1 for r in rows if r[7] is not None)
    log(f"    TTI misses stored: {n:,} ({placed:,} pinned to a slot)")
    if ambiguous:
        log(f"      {ambiguous:,} had no timestamp and repeat across SFN "
            f"wraps - left unplaced rather than marked at a guessed slot")
    if uncovered:
        log(f"      {uncovered:,} fall outside the captured window")
    return n

def _write_contexts(b, info: dict, mps: list, log) -> int:
    """Resolve an SM cap for every CUDA context the trace used.

    cuPHY runs each channel group in its own MPS context created with an
    SM-count execution affinity, and that cap is what bounds how many SMs a
    launch can occupy. Ignoring it spreads every grid over the whole GPU:
    measured on A2_20C-59c, that alone made modelled residency 2.2x what the
    caps allow and was most of the gap to DCGM sm_occupancy.

    Sources, best first:
      1. green contexts carry numMultiprocessors in TARGET_INFO_CUDA_CONTEXT_INFO;
      2. MPS contexts: the [DRV.CTX] log lines, in creation order, mapped onto
         the cuphycontroller process's contextIds in id order (nsys allocates
         ids in creation order; the primary context, created first by
         cudaSetDevice and launching nothing, is skipped). Each mapping is
         then checked against the channels its kernels belong to.
    A context with no resolvable cap is stored with sm_cap NULL, and the
    spare model treats it as uncapped - and says so.
    """
    ctx_chan = info.get("ctx_chan", {})
    ctx_info = info.get("ctx_info", [])
    rows = {}

    def dominant(cid):
        ch = {k: v for k, v in ctx_chan.get(cid, {}).items()
              if k not in ("Graph", "Utility", "Unknown")}
        return max(ch, key=ch.get) if ch else None

    for ci in ctx_info:
        cid = ci.get("contextId")
        if cid is None:
            continue
        cap = ci.get("numMultiprocessors") if ci.get("isGreenContext") else None
        rows[cid] = [cid, ci.get("processId"), None, cap,
                     "green_context" if cap else "none", 0,
                     sum(ctx_chan.get(cid, {}).values()), dominant(cid)]
    for cid in ctx_chan:                       # contexts absent from TARGET_INFO
        rows.setdefault(cid, [cid, None, None, None, "none", 0,
                              sum(ctx_chan[cid].values()), dominant(cid)])

    if mps:
        # The DU's process is the one whose contexts run known Aerial kernels.
        known: dict = {}
        for cid, r in rows.items():
            k = sum(v for c, v in ctx_chan.get(cid, {}).items() if c != "Unknown")
            known[r[1]] = known.get(r[1], 0) + k
        proc = max(known, key=known.get) if known else None
        cands = sorted(cid for cid, r in rows.items() if r[1] == proc)
        if len(cands) == len(mps) + 1 and not ctx_chan.get(cands[0]):
            cands = cands[1:]                  # primary context
        if len(cands) == len(mps):
            bad = []
            for cid, (label, cap) in zip(cands, mps):
                r = rows[cid]
                if r[4] == "green_context":
                    continue
                r[2], r[3], r[4] = label, cap, "mps_log"
                exp = next((v for k, v in MPS_CTX_EXPECT.items()
                            if label.lower().startswith(k.lower())), None)
                r[5] = int(bool(exp and r[7] in exp))
                if exp and r[7] and r[7] not in exp:
                    bad.append(f"{label}->ctx{cid} runs {r[7]}")
            if bad:
                log(f"    WARNING: MPS context mapping disagrees with kernel "
                    f"channels: {'; '.join(bad)}")
        else:
            log(f"    WARNING: {len(mps)} MPS context lines but "
                f"{len(cands)} candidate contexts; SM caps not applied")

    b.st.executemany("INSERT INTO contexts VALUES(?,?,?,?,?,?,?,?)",
                     [tuple(r) for r in rows.values()])
    capped = [r for r in rows.values() if r[3]]
    if capped:
        log("    SM caps: " + ", ".join(
            f"ctx{r[0]} {r[2] or r[7] or '?'}={r[3]}"
            + ("" if r[5] or r[4] == "green_context" else "?")
            for r in sorted(capped)))
    elif rows:
        log("    no SM caps found (no [DRV.CTX] lines, no green contexts) - "
            "spare analysis will treat every context as uncapped")
    return len(capped)


def build_store(capture_dir: Path, out: Path, log=print) -> None:
    cap = Capture.discover(Path(capture_dir))
    if cap.nsys_rep and not cap.sqlite:
        cap.sqlite = export_nsys_rep(cap.nsys_rep, log)
    if not (cap.sqlite or cap.ktrace_log or cap.nvlog):
        raise SystemExit(
            f"no .sqlite / kernel_printf_*.log / cuphy_*.log under {capture_dir}\n"
            f"(if only a .nsys-rep is present, export it first:\n"
            f"   nsys export --type sqlite -o run.sqlite run.nsys-rep)")
    log(f"building {out.name} from {cap.root}")
    log(f"  sources: {cap.describe()}")

    t_start = time.time()
    if out.exists():
        out.unlink()
    st = sqlite3.connect(out)
    st.executescript(STORE_DDL)
    b = _Build(st)

    n_ktrace = 0
    samples: dict = {}
    counts: dict = {}
    durations: dict = {}
    if cap.ktrace_log:
        n_ktrace, samples, counts, durations = _read_ktrace(cap.ktrace_log, b, log)

    ticks, tasks, n_cells, misses, mps = [], [], 0, [], []
    if cap.nvlog:
        ticks, tasks, n_cells, misses, mps = _read_nvlog(cap.nvlog, log)
    if cap.ctrl_log:
        mps = read_mps_contexts(cap.ctrl_log) or _dedupe_mps(mps)
    else:
        mps = _dedupe_mps(mps)

    n_nsys, nsys_by, info = 0, {}, {"gpu_name": "Unknown GPU", "sm_count": 132,
                                    "max_warps": 132 * 64, "utc_epoch_ns": None}
    if cap.sqlite:
        log(f"  reading {cap.sqlite.name} "
            f"({cap.sqlite.stat().st_size/1e9:.1f} GB)")
        sq = sqlite3.connect(f"file:{cap.sqlite}?mode=ro", uri=True)
        sq.execute("PRAGMA mmap_size=1073741824")
        # NVTX payloads are not guaranteed UTF-8 — a handful of ranges in the
        # 20-cell capture carry raw bytes. The default text_factory raises
        # OperationalError on the first one, which silently truncated the phase
        # table at 18 rows out of 779,451.
        sq.text_factory = lambda raw: raw.decode("utf-8", "replace")
        n_nsys, nsys_by, info = _read_nsys(sq, b, log)
        sq.close()

    # ---- place KTRACE launches on the nsys timeline -------------------------
    offset = iqr = calib_kernel = None
    calib_n = 0
    if n_ktrace:
        if nsys_by:
            offset, iqr, calib_kernel, calib_n, considered = _clock_offset(
                nsys_by, samples, counts)
            if offset is not None:
                log(f"  clock offset {offset:+,} ns (IQR {iqr:,} ns) via "
                    f"'{calib_kernel}' over {calib_n:,} matched launches "
                    f"[{len(considered)} candidates]")
                for cd_iqr, _o, cd_k, cd_n in considered[1:5]:
                    log(f"      runner-up: {cd_k:<38} IQR {cd_iqr:>12,} ns "
                        f"({cd_n:,})")
            else:
                log("  WARNING: no kernel appears in both sources with a "
                    "matching launch count; KTRACE launches cannot be placed "
                    "on the nsys timeline and are dropped")
        else:
            # KTRACE alone: %globaltimer IS the timeline. Rebase to zero so the
            # UI does not have to reason about a 1e18 origin.
            offset = -(st.execute("SELECT MIN(t0) FROM ktrace_raw").fetchone()[0] or 0)
            iqr, calib_kernel, calib_n = 0, "(ktrace-only, rebased)", n_ktrace
            log("  no nsys source; using %globaltimer as the timeline")

        if offset is not None:
            # KTRACE and nsys both record the 36 host-launched kernels. Keeping
            # both would double every one of them — inflating launch counts,
            # busy time and occupancy, and drawing each bar twice. nsys wins
            # where it can see: it has a real end timestamp, the stream id, the
            # register count and the shared-memory size, none of which the
            # device printf carries. KTRACE is kept only for what nsys cannot
            # see at all — the 23 device-launched PUSCH/UL kernels.
            dup = [b.kid[k] for k in nsys_by if k in b.kid]
            keep = ("" if not dup else
                    " WHERE kernel_id NOT IN (%s)" % ",".join(map(str, dup)))
            n_dup = st.execute(
                "SELECT COUNT(*) FROM ktrace_raw" +
                ("" if not dup else " WHERE kernel_id IN (%s)"
                 % ",".join(map(str, dup)))).fetchone()[0] if dup else 0

            med = {k: sorted(v)[len(v) // 2] for k, v in durations.items() if v}
            dur_case = " ".join(
                f"WHEN {b.kid[k]} THEN {v}" for k, v in med.items() if k in b.kid)
            dexpr = (f"t0+{offset}+(CASE kernel_id {dur_case} ELSE NULL END)"
                     if dur_case else "NULL")
            st.execute(
                f"INSERT INTO events SELECT t0+{offset},{dexpr},kernel_id,"
                f"{SRC_KTRACE},{DUR_ESTIMATED},NULL,blocks,tpb,warps,"
                f"NULL,NULL,NULL,NULL,NULL,NULL,NULL FROM ktrace_raw{keep}")
            n_ktrace -= n_dup
            if n_dup:
                log(f"  {n_dup:,} KTRACE launches dropped as duplicates of "
                    f"nsys rows; {n_ktrace:,} kept (device-launched only)")
        else:
            n_ktrace = 0
    st.execute("DROP TABLE IF EXISTS ktrace_raw")

    # ---- duration statistics per kernel ------------------------------------
    # Two independent measurement sources, never mixed:
    #   nsys rows carry a real end timestamp, so t1-t0 IS the duration;
    #   KTRACE rows only have one where KERNEL_TIME_END_PRINT was instrumented.
    # nsys wins when both exist — it is a hardware timestamp pair, whereas the
    # KTRACE duration is measured by the kernel about itself.
    st.execute("CREATE INDEX idx_ev_kern_tmp ON events(kernel_id,src)")
    for name, kid in b.kid.items():
        n = st.execute("SELECT COUNT(*) FROM events WHERE kernel_id=? AND "
                       "src=? AND t1 IS NOT NULL", (kid, SRC_NSYS)).fetchone()[0]
        if n:
            q = ("SELECT t1-t0 d FROM events WHERE kernel_id=? AND src=? "
                 "AND t1 IS NOT NULL ORDER BY d LIMIT 1 OFFSET ?")
            pick = lambda o: st.execute(q, (kid, SRC_NSYS, o)).fetchone()[0]
            st.execute("UPDATE kernels SET dur_n=?,dur_min=?,dur_med=?,"
                       "dur_p99=?,dur_max=? WHERE id=?",
                       (n, pick(0), pick(n // 2),
                        pick(min(n - 1, int(n * .99))), pick(n - 1), kid))
            continue
        v = durations.get(name)
        if v:
            v.sort()
            n = len(v)
            st.execute("UPDATE kernels SET dur_n=?,dur_min=?,dur_med=?,"
                       "dur_p99=?,dur_max=? WHERE id=?",
                       (n, v[0], v[n // 2], v[min(n - 1, int(n * .99))],
                        v[-1], kid))
    # ---- provenance, from the capture rather than from a name list ---------
    # nsys_visible is stamped at kernel-insert time from DEVICE_GRAPH_HIDDEN,
    # which is a statement about the stock run config, not about this capture.
    # With pusch_workCancelMode/pusch_deviceGraphLaunchEn set to 0 those kernels
    # ARE host-launched and nsys records them, so the stamped value is wrong in
    # exactly the runs that were taken to observe them. Overwrite it with what
    # the event rows say. src_mask: bit0 = seen by nsys, bit1 = seen by KTRACE.
    st.execute("UPDATE kernels SET nsys_visible = COALESCE(("
               "  SELECT MAX(e.src = ?) FROM events e WHERE e.kernel_id = kernels.id"
               "), 0)", (SRC_NSYS,))
    st.execute("UPDATE kernels SET src_mask = COALESCE(("
               "  SELECT SUM(DISTINCT 1 << e.src) FROM events e"
               "  WHERE e.kernel_id = kernels.id"
               "), 0)")
    st.execute("DROP INDEX idx_ev_kern_tmp")

    # ---- host-side tasks ----------------------------------------------------
    epoch = info.get("utc_epoch_ns")
    n_task = 0
    if tasks and epoch:
        # Same clip as the slots: a host task outside the traced window has no
        # kernels to be shown against.
        _sp = _traced_span(b)
        _rows = [(t0 - epoch, t1 - epoch, nm, sfn, sl, cell, wk, stg)
                 for t0, t1, nm, sfn, sl, cell, wk, stg in tasks
                 if not _sp or (t1 - epoch >= _sp[0] and t0 - epoch <= _sp[1])]
        st.executemany(
            "INSERT INTO tasks(t0,t1,name,sfn,slot,cell,worker,slot_idx,stages) "
            "VALUES(?,?,?,?,?,?,?,NULL,?)", _rows)
        n_task = len(_rows)
        if len(_rows) != len(tasks):
            log(f"  {len(tasks)-len(_rows):,} of {len(tasks):,} host task "
                f"records fall outside the traced window and were not stored")
    elif tasks:
        log("  note: nvlog task records found but no nsys UTC epoch to anchor "
            "them; host pipeline view unavailable")

    # ---- slots, attribution, rollup ----------------------------------------
    n_slot = _build_slots(b, ticks, epoch, log)
    _write_tti_miss(b, misses, epoch, log)
    n_capped = _write_contexts(b, info, mps, log) if cap.sqlite else 0
    _assign_slots(b, log)
    _compute_occupancy(b, log)
    _rollup(b, int(info["max_warps"]), log)

    slot_dur = st.execute(
        "SELECT AVG(t1-t0) FROM slots").fetchone()[0] or 500_000

    log("  indexing")
    st.executescript(STORE_INDICES)
    meta = [("schema", SCHEMA_VERSION), ("label", Path(capture_dir).name),
            ("capture_dir", str(cap.root)),
            ("n_events", n_nsys + n_ktrace), ("n_nsys", n_nsys),
            ("n_ktrace", n_ktrace), ("n_slots", n_slot), ("n_tasks", n_task),
            ("n_kernels", len(b.kid)), ("n_cells", n_cells),
            ("slot_dur_ns", int(slot_dur)),
            ("clock_offset_ns", offset), ("clock_iqr_ns", iqr),
            ("calib_kernel", calib_kernel), ("calib_n", calib_n),
            ("gpu_name", info["gpu_name"]), ("sm_count", info["sm_count"]),
            ("max_warps", info["max_warps"]), ("utc_epoch_ns", epoch),
            ("n_ctx_capped", n_capped),
            # Longest launch, so windowed queries can look back far enough
            # to catch a kernel that started before the window and is still
            # running in it (gpu_comm_trigger_send_doca_cx7 runs ~7.8 ms).
            ("max_dur_ns", st.execute(
                "SELECT MAX(t1-t0) FROM events WHERE t1 IS NOT NULL"
            ).fetchone()[0] or 0),
            ("max_cluster", info.get("max_cluster", 1)),
            ("built_at", time.strftime("%Y-%m-%d %H:%M:%S"))]
    # the per-SM capacity table, when the capture carried it
    meta += [(k, info[k]) for k in
             ("max_warps_per_sm", "max_blocks_per_sm", "regs_per_sm",
              "smem_per_sm", "smem_per_block_optin", "warp_size",
              "max_threads_per_block", "compute_major", "compute_minor",
              "total_memory", "chip_name") if k in info]
    st.executemany("INSERT INTO meta VALUES(?,?)",
                   [(k, str(v)) for k, v in meta])
    st.commit()
    st.execute("VACUUM")
    st.close()
    log(f"built {out.name}: {n_nsys+n_ktrace:,} events "
        f"({n_nsys:,} nsys + {n_ktrace:,} ktrace), {len(b.kid)} kernels, "
        f"{n_slot:,} slots, {n_task:,} host tasks, "
        f"{out.stat().st_size/1e6:.0f} MB, {time.time()-t_start:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Build a TTI-Scope session store")
    ap.add_argument("capture_dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-f", "--force", action="store_true",
                    help="rebuild even if a current store exists")
    a = ap.parse_args()
    st = SessionStore.open_or_build(a.capture_dir, a.out, force=a.force)

    print(f"\nstore      : {st.path}")
    print(f"gpu        : {st.gpu_name}  ({st.sm_count} SMs, "
          f"{st.max_warps} max resident warps)")
    print(f"events     : {st.n_events:,}")
    print(f"slots      : {st.n_slots:,}   ({st.slot_dur_ns/1000:.0f} us each)")
    print(f"cells      : {st.n_cells}")
    print(f"clock off  : {st.clock_offset_ns} ns (IQR {st.clock_iqr_ns})")

    vis = sum(1 for v in st.kernels.values() if v[3])
    dfn = sum(1 for v in st.kernels.values() if v[4] == 'device_fn')
    print(f"kernels    : {len(st.kernels)-dfn}  "
          f"({vis} nsys-visible, {len(st.kernels)-vis-dfn} KTRACE-only)"
          + (f"  + {dfn} instrumented device function(s)" if dfn else ""))

    if st.n_slots:
        mid = st.n_slots // 2
        lo, hi = max(0, mid - 20), min(st.n_slots - 1, mid + 20)
        occ = st.occupancy_series(lo, hi)
        if occ:
            print(f"\nmid-capture slots {lo}..{hi}:")
            print(f"  SM occupancy  mean {sum(o[1] for o in occ)/len(occ):6.2f}%"
                  f"   peak {max(o[1] for o in occ):6.2f}%")
            print(f"  GPU busy      mean {sum(o[2] for o in occ)/len(occ):6.2f}%"
                  f"   peak {max(o[2] for o in occ):6.2f}%")
        print(f"\nchannel breakdown, slots {lo}..{hi} "
              f"({st.slot_label(lo)} .. {st.slot_label(hi)}):")
        print(f"  {'channel':<10}{'launches':>10}{'busy us':>12}{'% warp-time':>13}")
        rows = st.channel_histogram(lo, hi)
        tw = sum(r[3] or 0 for r in rows) or 1
        for ch, n, busy, warp in rows:
            print(f"  {ch:<10}{n:>10,}{(busy or 0)/1000:>12.1f}"
                  f"{100.0*(warp or 0)/tw:>13.1f}")


if __name__ == "__main__":
    main()
