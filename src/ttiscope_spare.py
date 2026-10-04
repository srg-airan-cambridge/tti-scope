#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
Spare-GPU quantification from an nsys trace, and the phantom-kernel scheduler.
primarily developed for GH200. GB10 architecture needs further development.

"""
from __future__ import annotations

import csv
import glob
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ttiscope_occupancy import (SM90, _ceil_to, _floor_to, block_smem_bytes,
                                smallest_carveout, theoretical_occupancy)

MEASURED = "MEASURED"
DERIVED = "DERIVED"
MODELLED = "MODELLED"

MIN_REGS_PER_THREAD = 16      # the lowest any real kernel in these traces uses
# Dynamic shared memory above this needs cudaFuncSetAttribute(
# cudaFuncAttributeMaxDynamicSharedMemorySize) before launch.
SMEM_DEFAULT_LIMIT = 48 * 1024

SMEM_MODES = ("conservative", "optimistic")


# ─────────────────────────────────────────────────────────────────────────────
# Roofline


@dataclass
class Roofline:
    """Device capacity. Per-SM limits come from the capture's TARGET_INFO_GPU;
    only the allocation granularities and the carveout table are architecture
    constants, because nsys does not record them."""
    num_sm: int = 132
    max_warps_per_sm: int = SM90["max_warps_per_sm"]
    max_blocks_per_sm: int = SM90["max_blocks_per_sm"]
    max_threads_per_sm: int = SM90["max_threads_per_sm"]
    regs_per_sm: int = SM90["regs_per_sm"]
    smem_per_sm: int = SM90["smem_per_sm"]
    warp_size: int = SM90["warp_size"]
    reg_alloc_unit: int = SM90["reg_alloc_unit"]
    warp_alloc_granularity: int = SM90["warp_alloc_granularity"]
    smem_alloc_unit: int = SM90["smem_alloc_unit"]
    max_regs_per_thread: int = SM90["max_regs_per_thread"]
    max_threads_per_block: int = 1024
    reserved_smem_per_block: int = SM90["reserved_smem_per_block"]
    smem_per_block_optin: int = SM90["smem_per_block_optin"]
    carveouts: tuple = SM90["carveouts"]
    fb_total_mib: Optional[int] = None
    gpu_name: str = "Unknown GPU"
    chip_name: str = ""
    compute_cap: str = ""
    src: str = MODELLED
    table_src: str = MODELLED     # where the per-SM limits came from
    reserve_src: str = MODELLED   # where the per-block smem reserve came from

    def dev(self) -> dict:
        """The dict shape ttiscope_occupancy.theoretical_occupancy expects."""
        return dict(SM90,
                    max_warps_per_sm=self.max_warps_per_sm,
                    max_blocks_per_sm=self.max_blocks_per_sm,
                    max_threads_per_sm=self.max_threads_per_sm,
                    regs_per_sm=self.regs_per_sm,
                    smem_per_sm=self.smem_per_sm,
                    warp_size=self.warp_size,
                    reg_alloc_unit=self.reg_alloc_unit,
                    warp_alloc_granularity=self.warp_alloc_granularity,
                    smem_alloc_unit=self.smem_alloc_unit,
                    max_threads_per_block=self.max_threads_per_block,
                    reserved_smem_per_block=self.reserved_smem_per_block,
                    smem_per_block_optin=self.smem_per_block_optin,
                    carveouts=self.carveouts)

    def block_smem(self, smem) -> int:
        return block_smem_bytes(smem, self.dev())

    # Register and warp allocation granularity are the one part of the table
    # nsys does NOT record. They are architecture constants: 256 registers per
    # warp and warps in groups of 4 on every compute capability from 7.0
    # through 12.x. Kept as an explicit map so an architecture that changes
    # them is a one-line edit rather than a silent wrong answer.
    _ALLOC = {None: (256, 4)}
    # Legal L1/shared carveouts per compute capability, bytes per SM. Only
    # architectures whose table has been checked are listed; anything else
    # gets a single-entry table (no rounding), which is stated in describe().
    _CARVEOUTS = {"9.0": SM90["carveouts"]}

    @classmethod
    def from_store(cls, st, dcgm: "DcgmSeries" = None) -> "Roofline":
        """Read the device's real capacity out of the capture.

        Stores built before the per-SM table was kept fall back to the
        capture's own .sqlite; only if both are unavailable does the sm_90
        table stand in, and then `table_src` says MODELLED.
        """
        rf = cls()
        got_table = False
        try:
            rf.num_sm = int(st.sm_count) or 132
            rf.gpu_name = st.gpu_name
            # sm_count has a 132 default; only a real TARGET_INFO_GPU read
            # also yields a GPU name, so that is what makes it MEASURED.
            if rf.gpu_name and rf.gpu_name != "Unknown GPU":
                rf.src = MEASURED
        except Exception:
            pass
        optin = {}

        def take(src):
            """src(key) -> value or None. Returns True if a real table was found."""
            n = 0
            for attr, key in (("max_warps_per_sm", "max_warps_per_sm"),
                              ("max_blocks_per_sm", "max_blocks_per_sm"),
                              ("regs_per_sm", "regs_per_sm"),
                              ("smem_per_sm", "smem_per_sm"),
                              ("warp_size", "warp_size")):
                v = src(key)
                if v:
                    setattr(rf, attr, int(v))
                    n += 1
            v = src("max_threads_per_block")
            if v:
                rf.max_threads_per_block = int(v)
            v = src("smem_per_block_optin")
            if v:
                optin["v"] = int(v)
            cm, cn = src("compute_major"), src("compute_minor")
            if cm:
                rf.compute_cap = f"{int(cm)}.{int(cn or 0)}"
            v = src("chip_name")
            if v:
                rf.chip_name = (v.decode(errors="replace")
                                if isinstance(v, bytes) else str(v))
            v = src("total_memory")
            if v and rf.fb_total_mib is None:
                rf.fb_total_mib = int(v) // (1024 * 1024)
            return n >= 4

        got_table = take(lambda k: st._m(k, str, None))

        if not got_table:
            try:
                import sqlite3
                cap = st._meta.get("capture_dir")
                hits = sorted(glob.glob(str(Path(cap) / "*.sqlite"))) if cap else []
                if hits:
                    sq = sqlite3.connect(f"file:{hits[0]}?mode=ro", uri=True)
                    sq.text_factory = bytes
                    cols = [c[1].decode() for c in
                            sq.execute("PRAGMA table_info(TARGET_INFO_GPU)")]
                    row = sq.execute(
                        "SELECT * FROM TARGET_INFO_GPU LIMIT 1").fetchone()
                    sq.close()
                    gi = dict(zip(cols, row)) if row else {}
                    m = {"max_warps_per_sm": "maxWarpsPerSm",
                         "max_blocks_per_sm": "maxBlocksPerSm",
                         "regs_per_sm": "maxRegistersPerSm",
                         "smem_per_sm": "maxShmemPerSm",
                         "smem_per_block_optin": "maxShmemPerBlockOptin",
                         "warp_size": "threadsPerWarp",
                         "max_threads_per_block": "maxThreadsPerBlock",
                         "compute_major": "computeMajor",
                         "compute_minor": "computeMinor",
                         "total_memory": "totalMemory",
                         "chip_name": "chipName"}
                    got_table = take(lambda k: gi.get(m.get(k, k)))
            except Exception:
                pass

        rf.max_threads_per_sm = rf.max_warps_per_sm * rf.warp_size
        ra, wg = cls._ALLOC[None]
        rf.reg_alloc_unit, rf.warp_alloc_granularity = ra, wg
        # The reserve is not recorded directly, but it is exactly the gap
        # between what an SM holds and what one block may opt in to.
        if optin.get("v") and rf.smem_per_sm > optin["v"]:
            rf.smem_per_block_optin = optin["v"]
            rf.reserved_smem_per_block = rf.smem_per_sm - optin["v"]
            rf.reserve_src = DERIVED
        rf.carveouts = cls._CARVEOUTS.get(rf.compute_cap, (rf.smem_per_sm,))
        rf.table_src = MEASURED if got_table else MODELLED
        if dcgm is not None and dcgm.ok:
            fb = dcgm.median("fb_total_mib")
            if fb:
                rf.fb_total_mib = int(fb)
        return rf

    @classmethod
    def from_file(cls, path) -> Optional["Roofline"]:
        """Parse the `<report>.roofline` the BenchmarkingApp writes at startup
        from cudaGetDeviceProperties, so scope and harvester agree by
        construction rather than by both hardcoding sm_90."""
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        kv = {}
        for line in p.read_text(errors="replace").splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            kv[k.strip()] = v.strip()
        if not kv:
            return None

        def g(*names):
            return next((kv[n] for n in names if n in kv), None)

        def _i(val, dflt):
            try:
                return int(str(val).split()[0])
            except (TypeError, ValueError, IndexError):
                return dflt

        rf = cls(src=MEASURED, table_src=MEASURED)
        rf.num_sm = _i(g("multiProcessorCount", "numSM", "SM"), rf.num_sm)
        rf.max_threads_per_sm = _i(g("maxThreadsPerMultiProcessor",
                                     "maxThreadsPerSM"), rf.max_threads_per_sm)
        rf.max_blocks_per_sm = _i(g("maxBlocksPerMultiProcessor",
                                    "maxBlocksPerSM"), rf.max_blocks_per_sm)
        rf.regs_per_sm = _i(g("regsPerMultiprocessor", "regsPerSM"),
                            rf.regs_per_sm)
        rf.smem_per_sm = _i(g("sharedMemPerMultiprocessor", "smemPerSM"),
                            rf.smem_per_sm)
        rf.warp_size = _i(g("warpSize"), rf.warp_size)
        rf.gpu_name = g("name", "gpu") or rf.gpu_name
        rf.max_warps_per_sm = rf.max_threads_per_sm // max(rf.warp_size, 1)
        optin = _i(g("sharedMemPerBlockOptin"), None)
        res = _i(g("reservedSharedMemPerBlock"), None)
        if optin:
            rf.smem_per_block_optin = optin
        if res is not None:
            rf.reserved_smem_per_block, rf.reserve_src = res, MEASURED
        elif optin and rf.smem_per_sm > optin:
            rf.reserved_smem_per_block = rf.smem_per_sm - optin
            rf.reserve_src = DERIVED
        mj, mn = _i(g("major"), None), _i(g("minor"), None)
        if mj is not None:
            rf.compute_cap = f"{mj}.{mn or 0}"
        rf.carveouts = cls._CARVEOUTS.get(rf.compute_cap, (rf.smem_per_sm,))
        mib = g("totalGlobalMem_MiB", "totalGlobalMem", "fb_total_mib")
        if mib:
            rf.fb_total_mib = _i(mib, None)
        return rf

    def describe(self) -> str:
        fb = f"{self.fb_total_mib:,} MiB" if self.fb_total_mib else "unknown"
        cc = f" sm_{self.compute_cap.replace('.', '')}" if self.compute_cap else ""
        warn = "" if self.table_src == MEASURED else \
            "   [!] per-SM limits ASSUMED sm_90, not read from the capture"
        if len(self.carveouts) <= 1:
            warn += "   [!] no carveout table for this architecture"
        return (f"{self.gpu_name}{cc}  {self.num_sm} SM x "
                f"{self.max_warps_per_sm} warps  ·  "
                f"{self.max_blocks_per_sm} blk/SM  ·  "
                f"{self.smem_per_sm/1024:.0f} KiB smem/SM "
                f"(+{self.reserved_smem_per_block} B reserved/block)  ·  "
                f"{self.regs_per_sm:,} regs/SM  ·  fb {fb}{warn}")


# ─────────────────────────────────────────────────────────────────────────────
# DCGM telemetry
# ─────────────────────────────────────────────────────────────────────────────

DCGM_GLOB = ("gputel_*.csv", "dcgm_*.csv", "*_dcgm.csv", "gpu_telemetry*.csv")


class DcgmSeries:
    """The gputel_*.csv a capture carries, aligned onto the nsys timeline.

    Alignment is exact, not fitted: the CSV stamps every row with epoch_ns and
    the store records utc_epoch_ns for nsys t=0, so nsys_ns = epoch_ns - epoch.
    """

    FIELDS = ("gr_engine_active", "sm_active", "sm_occupancy", "tensor_active",
              "dram_active", "fp64_active", "fp32_active", "fp16_active",
              "int_active", "power_w", "fb_free_mib", "fb_used_mib",
              "fb_total_mib", "sm_clock_mhz")

    def __init__(self, rows=None, path=None, epoch_ns=None):
        self.rows = rows or []           # [(nsys_ns, {field: float})]
        self.path = path
        self.epoch_ns = epoch_ns
        self._med = {}

    @property
    def ok(self) -> bool:
        return bool(self.rows)

    @property
    def n(self) -> int:
        return len(self.rows)

    @classmethod
    def load(cls, capture_dir, epoch_ns: Optional[int]) -> "DcgmSeries":
        """Read the telemetry CSV beside a capture. Lazy and cheap (~150 rows),
        so this deliberately does NOT live in the .tti store."""
        d = Path(capture_dir) if capture_dir else None
        if not d or not d.is_dir():
            return cls()
        hits = []
        for pat in DCGM_GLOB:
            hits += sorted(glob.glob(str(d / pat)))
        if not hits:
            return cls()
        path = Path(hits[0])
        out = []
        try:
            with open(path, newline="", errors="replace") as fh:
                for r in csv.DictReader(fh):
                    try:
                        t = int(r["epoch_ns"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    vals = {}
                    for f in cls.FIELDS:
                        v = r.get(f, "")
                        if v in ("", "N/A", None):
                            continue
                        try:
                            vals[f] = float(v)
                        except ValueError:
                            continue
                    if vals:
                        out.append(((t - epoch_ns) if epoch_ns is not None
                                    else t, vals))
        except OSError:
            return cls()
        out.sort(key=lambda x: x[0])
        return cls(out, path, epoch_ns)

    def _vals(self, field_name, t0, t1):
        return [v[field_name] for t, v in self.rows
                if field_name in v and (t0 is None or t >= t0)
                and (t1 is None or t <= t1)]

    def median(self, field_name: str, t0=None, t1=None) -> Optional[float]:
        if not self.rows:
            return None
        if t0 is None and t1 is None and field_name in self._med:
            return self._med[field_name]
        vals = sorted(self._vals(field_name, t0, t1))
        if not vals:
            return None
        m = vals[len(vals) // 2]
        if t0 is None and t1 is None:
            self._med[field_name] = m
        return m

    def mean(self, field_name: str, t0=None, t1=None):
        """(mean, n_samples) over [t0,t1]. Samples are evenly spaced, so the
        plain mean is the time-weighted one."""
        vals = self._vals(field_name, t0, t1)
        return ((sum(vals) / len(vals)) if vals else None), len(vals)

    def span_ns(self):
        return (self.rows[0][0], self.rows[-1][0]) if self.rows else (None, None)

    def describe(self) -> str:
        if not self.ok:
            return "no DCGM telemetry beside this capture"
        a, b = self.span_ns()
        bits = [f"{self.n} samples over {(b - a)/1e9:.1f} s from {self.path.name}"]
        for f in ("sm_occupancy", "sm_active", "gr_engine_active"):
            v = self.median(f)
            if v is not None:
                bits.append(f"{f} {v:.3f}")
        return "  ·  ".join(bits)


# ─────────────────────────────────────────────────────────────────────────────
# Launch shapes
# ─────────────────────────────────────────────────────────────────────────────

class _ShapeCache:
    """Launch shape -> per-block resource cost and occupancy limit.

    Shapes repeat heavily (a few hundred across ~690k launches), so the
    occupancy calculator runs once per shape. Returns
        (wpb, blk_smem, blk_regs, k, carveout, carveout_src)
    where k is blocks/SM under the carveout the launch actually ran with.

    The carveout comes from CUPTI sharedMemoryExecuted when that value can
    physically hold the launch's blocks. It cannot always: the PUSCH LDPC
    decoder cubins opt in to 40 KiB blocks yet report 32 KiB, and the UL
    ping-pong kernel reports 0 with 177 B of static shared memory. Those are
    replaced by the smallest legal carveout that fits the occupancy-limited
    block count, and marked inferred.
    """

    def __init__(self, rf: Roofline):
        self.rf = rf
        self.dev = rf.dev()
        self._c = {}

    def __call__(self, tpb, regs, smem, cfg):
        key = (tpb, regs, smem, cfg)
        hit = self._c.get(key)
        if hit is not None:
            return hit
        rf = self.rf
        tpb = int(tpb or 0)
        if tpb <= 0:
            out = (0, 0, 0, 0, 0, "")
            self._c[key] = out
            return out
        wpb = _ceil_to(tpb, rf.warp_size) // rf.warp_size
        bsm = block_smem_bytes(smem, self.dev)
        rpw = _ceil_to(int(regs or 0) * rf.warp_size, rf.reg_alloc_unit)
        cfg = int(cfg or 0)
        if cfg >= bsm:
            d = dict(self.dev, smem_per_sm=min(cfg, rf.smem_per_sm))
            k = theoretical_occupancy(tpb, regs, smem, d)[1]
            car, why = cfg, "cupti"
        else:
            k = theoretical_occupancy(tpb, regs, smem, self.dev)[1]
            car = smallest_carveout(k * bsm, self.dev)
            why = "inferred"
        out = (wpb, bsm, wpb * rpw, k, car, why)
        self._c[key] = out
        return out


# ─────────────────────────────────────────────────────────────────────────────
# The demand profile across a span
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    """Aerial's per-SM demand sampled across a time span.

    Arrays are parallel, one entry per bin, each the MAXIMUM over the bin
    (a co-tenant has to fit the worst instant of a bin, not its average).
    `mean_occ` is the exact time-weighted warp occupancy over the span, the
    figure comparable with DCGM sm_occupancy. `scale` has been applied to the
    arrays when it is not 1.0.
    """
    bin_ns: int
    t0: int
    t1: int
    blk: list = field(default_factory=list)     # blocks / SM
    wrp: list = field(default_factory=list)     # warps / SM
    smem: list = field(default_factory=list)    # bytes / SM, incl. reserve
    regs: list = field(default_factory=list)    # registers / SM
    cfg: list = field(default_factory=list)     # conservative smem roof/SM
    n_launch: int = 0
    n_queued: int = 0          # segments where a live launch got < its want
    mean_occ: float = 0.0
    unshaped_ns: int = 0       # busy time of launches with no register data
    busy_ns: int = 0
    # The exact piecewise-constant demand the bins were taken from:
    # [(x, y, blk, wrp, smem, regs, smem_roof_conservative)] per SM, [t0, t1)
    # gap-free (idle stretches carry zero demand). The oracle works on these,
    # not on the bins, so its answer does not depend on the bin width.
    segs: list = field(default_factory=list)
    scale: float = 1.0
    anchor_src: str = MODELLED
    raw_occ_mean: float = 0.0
    dcgm_occ: Optional[float] = None

    @property
    def n(self) -> int:
        return len(self.wrp)

    def occ_mean(self, rf: Roofline) -> float:
        if not self.wrp:
            return 0.0
        return 100.0 * (sum(self.wrp) / len(self.wrp)) / rf.max_warps_per_sm

    def provenance(self) -> str:
        if self.anchor_src == MEASURED:
            return (f"MPS-capped allocation model, scaled x{self.scale:.3f} to "
                    f"DCGM sm_occupancy {self.dcgm_occ:.3f}")
        return "MPS-capped allocation model from nsys launch geometry, unscaled"


def _event_query(st) -> str:
    cols = {r[1] for r in st.db.execute("PRAGMA table_info(events)")}
    c = lambda n: f"e.{n}" if n in cols else "NULL"
    # Device functions report their parent's launch geometry and are not
    # launches; KTRACE rows carry no regs/smem and fall back to the per-kernel
    # figure (cuobjdump static maximum for kernels nsys cannot see).
    return ("SELECT e.t0,e.t1,e.blocks,e.tpb,COALESCE(e.regs,k.regs),"
            "COALESCE(e.smem,k.smem),"
            f"{c('ctx')},{c('gctx')},{c('smem_cfg')} "
            "FROM events e JOIN kernels k ON k.id=e.kernel_id "
            "WHERE e.t0 >= ? AND e.t0 < ? AND e.t1 IS NOT NULL AND e.t1 > ? "
            "AND k.kind IS NOT 'device_fn' ORDER BY e.t0")


def demand_profile(st, t0: int, t1: int, ctx: "SpareContext",
                   bin_ns: int = 1000) -> Profile:
    """Aerial's per-SM demand over [t0,t1), from a sweep over the live set.

    Between two consecutive launch boundaries the set of live kernels is
    constant, so residency is computed once per stretch: launches in start
    order are granted blocks against what remains of their context's SM cap
    and of the device, in all four resources at once. The per-SM figure is
    the device total over N_SM - a mean field, see the module docstring.
    """
    rf = ctx.rf
    N = rf.num_sm
    n = max(1, int((t1 - t0) // bin_ns))
    blk, wrp, sm, rg = ([0.0] * n for _ in range(4))
    S_full = float(rf.smem_per_sm)
    # Conservative shared-memory ceiling, per SM (mean field): MIN over the
    # bin. Idle bins keep the full array - no Aerial launch configured them.
    cf = [S_full] * n
    rows = st.db.execute(ctx.query, (t0 - ctx.lookback_ns, t1, t0)).fetchall()
    shape = ctx.shapes
    caps = ctx.caps
    cap_dev = (rf.max_blocks_per_sm * N, rf.max_warps_per_sm * N,
               rf.smem_per_sm * N, rf.regs_per_sm * N)

    evs = []
    unshaped = busy = 0
    for a, b, g, tpb, regs, smem, c, gc, cfg in rows:
        a, b = max(a, t0), min(b, t1)
        if b <= a or not g:
            continue
        wpb, bsm, brg, k, car, _why = shape(tpb, regs, smem, cfg)
        if k <= 0:
            continue
        key = gc or c
        nc = min(caps.get(key) or N, N)
        evs.append((a, b, int(g), k, nc, key, (1.0, wpb, bsm, brg), car))
        busy += b - a
        if not regs:
            unshaped += b - a

    # boundaries of constant live sets
    marks = sorted({t0, t1} | {e[0] for e in evs} | {e[1] for e in evs})
    by_start = sorted(range(len(evs)), key=lambda i: evs[i][0])
    live, p, occ_int, n_q = [], 0, 0.0, 0
    segs = []
    for x, y in zip(marks, marks[1:]):
        while p < len(by_start) and evs[by_start[p]][0] <= x:
            live.append(by_start[p])
            p += 1
        live = [i for i in live if evs[i][1] > x]
        if not live:
            segs.append((x, y, 0.0, 0.0, 0.0, 0.0, S_full))
            continue
        tot = [0.0, 0.0, 0.0, 0.0]
        used = {}
        occ_sm = car_sm = 0.0
        for i in live:                      # launch order = dispatch order
            a, b, g, k, nc, key, per, car = evs[i]
            want = min(g, k * nc)
            r = want
            u = used.setdefault(key, [0.0, 0.0, 0.0, 0.0])
            for d in range(4):
                if per[d] > 0:
                    cap_c = cap_dev[d] / N * nc
                    r = min(r, (cap_c - u[d]) / per[d],
                            (cap_dev[d] - tot[d]) / per[d])
            r = max(0.0, r)
            if r < want - 1e-9:
                n_q += 1
            if r > 0:
                # The block scheduler spreads a grid breadth-first, so a
                # launch with r resident blocks touches about min(N_ctx, r)
                # SMs - and only THOSE carry its L1/shared configuration.
                touched = min(float(nc), r)
                occ_sm += touched
                car_sm += touched * min(car, S_full)
                for d in range(4):
                    u[d] += r * per[d]
                    tot[d] += r * per[d]
        vals = (tot[0] / N, tot[1] / N, tot[2] / N, tot[3] / N)
        # Conservative ceiling: occupied SMs keep the carveout their Aerial
        # launches set (weighted mean over them); the rest are unconfigured
        # and offer the full array. Applying one launch's carveout to every
        # SM instead would let the 20-block UL ping-pong kernel's 8 KiB
        # carveout cap shared memory on all 132 SMs.
        if occ_sm > 0:
            n_occ = min(float(N), occ_sm)
            car_mean = car_sm / occ_sm
            roof_c = ((N - n_occ) * S_full + n_occ * car_mean) / N
        else:
            roof_c = S_full
        segs.append((x, y) + vals + (roof_c,))
        occ_int += vals[1] * (y - x)
        i0 = max(0, int((x - t0) // bin_ns))
        i1 = min(n, int((y - t0 + bin_ns - 1) // bin_ns))
        for i in range(i0, i1):
            if vals[1] > wrp[i]:
                wrp[i] = vals[1]
            if vals[0] > blk[i]:
                blk[i] = vals[0]
            if vals[2] > sm[i]:
                sm[i] = vals[2]
            if vals[3] > rg[i]:
                rg[i] = vals[3]
            if roof_c < cf[i]:
                cf[i] = roof_c

    span = max(1, t1 - t0)
    p = Profile(bin_ns=bin_ns, t0=t0, t1=t1, blk=blk, wrp=wrp, smem=sm,
                regs=rg, cfg=cf, n_launch=len(evs), n_queued=n_q,
                mean_occ=occ_int / span / rf.max_warps_per_sm,
                unshaped_ns=unshaped, busy_ns=busy, segs=segs)
    p.raw_occ_mean = p.mean_occ
    if ctx.apply_scale and ctx.scale != 1.0:
        _apply_scale(p, ctx)
    return p


def _apply_scale(p: Profile, ctx: "SpareContext"):
    s = ctx.scale
    for arr in (p.blk, p.wrp, p.smem, p.regs):
        for i in range(len(arr)):
            arr[i] *= s
    p.segs = [(x, y, b * s, w * s, m * s, r * s, c)
              for x, y, b, w, m, r, c in p.segs]
    p.scale, p.anchor_src, p.dcgm_occ = s, MEASURED, ctx.dcgm_occ
    p.mean_occ *= s


def spare_profile(p: Profile, rf: Roofline, smem_mode: str = "conservative"):
    """Per-bin remaining capacity: roofline - Aerial, clamped at zero.

    smem_mode 'conservative' limits shared memory to the carveout the live
    launches configured (no SM reconfiguration); 'optimistic' to the full
    228 KiB. Idle bins have no configured carveout, so both agree there.
    """
    n = p.n
    sb = [0.0] * n
    sw = [0.0] * n
    ss = [0.0] * n
    sr = [0.0] * n
    cons = smem_mode == "conservative"
    for i in range(n):
        sb[i] = max(0.0, rf.max_blocks_per_sm - p.blk[i])
        sw[i] = max(0.0, rf.max_warps_per_sm - p.wrp[i])
        roof = rf.smem_per_sm
        if cons and p.cfg:
            roof = min(roof, p.cfg[i])
        ss[i] = max(0.0, roof - p.smem[i])
        sr[i] = max(0.0, rf.regs_per_sm - p.regs[i])
    return sb, sw, ss, sr


# ─────────────────────────────────────────────────────────────────────────────
# The seven-dimensional spare vector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpareVector:
    """A launchable co-tenant kernel shape, in the harvester's own dimensions."""
    B: int = 0            # threads per block
    S: int = 0            # shared memory per block, bytes (excl. the reserve)
    R: int = 0            # registers per thread
    D: float = 0.0        # per-block duration, us
    n: int = 1            # waves
    P: float = 500.0      # period, us
    M: int = 0            # device memory, MiB
    # derived
    k: int = 0            # blocks per SM
    G: int = 0            # grid size
    n_sm: int = 0         # SMs the co-tenant may use (its own MPS cap)
    warps_per_sm: float = 0.0
    occ_pct: float = 0.0
    duty: float = 0.0
    f_hz: float = 0.0
    limiter: str = ""
    feasible: bool = False
    reason: str = ""          # when infeasible, which resource ran out
    smem_optin: bool = False  # S above 48 KiB: needs the opt-in attribute
    S_opt: Optional[int] = None   # S under the optimistic carveout
    # oracle phantoms only
    d_safe_us: Optional[float] = None   # D*: how long this k could have run
    S_room: Optional[int] = None        # extra smem/block that still fits
    R_room: Optional[int] = None        # extra regs/thread that still fit
    src_level: str = MODELLED
    src_M: str = MODELLED

    def as_tuple(self):
        return (self.B, self.S, self.R, self.D, self.n, self.P, self.M)

    def ctl_line(self) -> str:
        """The control vector in the order harvest_mobo.py writes it, so a
        shape found here can be handed straight to the co-tenant."""
        return (f"grid={self.G} block={self.B} smem={self.S} "
                f"onus={self.D:.0f} waves={self.n} "
                f"period={self.P:.0f} memmb={self.M}"
                + (" smem_optin=1" if self.smem_optin else ""))

    def describe(self) -> str:
        if not self.feasible:
            return (f"INFEASIBLE for D={self.D:.0f}us - no launchable shape "
                    f"survives; binding resource: {self.reason or 'unknown'}")
        opt = (f" (opt {self.S_opt}B)" if self.S_opt is not None
               and self.S_opt != self.S else "")
        return (f"B={self.B} S={self.S}B{opt} R={self.R} D={self.D:.0f}us "
                f"n={self.n} P={self.P:.0f}us M={self.M}MiB   ->  "
                f"k={self.k} G={self.G} on {self.n_sm} SM "
                f"{self.warps_per_sm:.1f} warps/SM "
                f"occ {self.occ_pct:.1f}% duty {self.duty:.2f} "
                f"f {self.f_hz/1000:.1f} kHz   [{self.limiter}]")


def shape_from_capacity(blk_free: float, wrp_free: float, smem_free: float,
                        reg_free: float, rf: Roofline,
                        B: Optional[int] = None,
                        n_sm: Optional[int] = None) -> SpareVector:
    """Turn a per-SM capacity 4-vector into the largest launchable kernel shape.

    Block size is SEARCHED: at a fixed free-warp count a large block wastes
    whatever does not divide into it. The search maximises warps claimed and
    breaks ties toward fewer block slots.

    Every block costs align(S + reserve) shared memory, so even S=0 holds
    1 KiB per block; S is offered net of that reserve and capped at the
    per-block opt-in limit. R is floored at MIN_REGS_PER_THREAD.
    """
    v = SpareVector(P=500.0)
    v.n_sm = int(n_sm or rf.num_sm)
    ws = rf.warp_size
    res = rf.reserved_smem_per_block
    unit = rf.smem_alloc_unit
    min_bsm = _ceil_to(res, unit) if res else unit
    wrp_free = max(0.0, wrp_free)
    blk_free = max(0.0, blk_free)
    smem_free = max(0.0, smem_free)
    if wrp_free < 1.0 or blk_free < 1.0 or smem_free < min_bsm:
        v.reason = ("warps" if wrp_free < 1.0 else
                    "blocks" if blk_free < 1.0 else "shared memory")
        return v

    def build(bsize):
        wpb = _ceil_to(bsize, ws) // ws
        lim = {"blocks": blk_free, "warps": wrp_free / wpb,
               "shared memory": smem_free / min_bsm}
        k = int(min(lim.values()))
        k0 = k
        while k >= 1:
            rpw_free = reg_free / float(k * wpb)
            # Affordable only if the ROUNDED-UP per-warp allocation fits.
            R = (int(rpw_free) // rf.reg_alloc_unit) * (rf.reg_alloc_unit // ws)
            R = int(min(rf.max_regs_per_thread, R))
            if R >= MIN_REGS_PER_THREAD:
                break
            k -= 1
        if k < 1:
            return None
        S = _floor_to(int(smem_free // k) - res, unit)
        S = max(0, min(S, rf.smem_per_block_optin))
        # What stopped k growing. S and R are then sized to absorb whatever
        # shared memory and registers remain, so leftover slack after the
        # fact would always name those two - it says nothing about k.
        why = "registers" if k < k0 else min(lim, key=lim.get)
        return bsize, wpb, k, S, R, why

    if B:
        got = build(int(B))
    else:
        cands = []
        for c in (32, 64, 128, 256, 512, 1024):
            if c > min(rf.max_threads_per_sm, rf.max_threads_per_block):
                continue
            g = build(c)
            if g:
                cands.append(g)
        got = max(cands, key=lambda g: (g[2] * g[1], -g[2])) if cands else None
    if not got:
        # Every block size failed the register floor: Aerial holds enough of
        # the register file that no co-tenant kernel of any shape fits.
        v.reason = "registers"
        return v
    bsize, wpb, k, S, R, why = got
    v.B, v.S, v.R = bsize, S, R
    v.k = k
    v.n = 1
    v.G = k * v.n_sm
    v.warps_per_sm = k * wpb
    v.occ_pct = 100.0 * v.warps_per_sm / rf.max_warps_per_sm
    v.smem_optin = S > SMEM_DEFAULT_LIMIT
    v.limiter = why
    v.feasible = True
    return v


def _sliding_min(a, w):
    """Minimum of every length-w window, O(n) via a monotonic deque."""
    dq, out = deque(), []
    for i, x in enumerate(a):
        while dq and a[dq[-1]] >= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - w:
            dq.popleft()
        if i >= w - 1:
            out.append(a[dq[0]])
    return out


def capacity_duration_frontier(sw, sb, ss, sr, bin_ns: int, rf: Roofline,
                               durations_us=(10, 25, 50, 100, 150, 200,
                                             250, 300, 400, 500)):
    """For each duration D, the largest capacity continuously free for D
    microseconds somewhere in the span. The sustainable capacity across a
    window is its MINIMUM, not its mean.

    Returns per duration: (D_us, warps/SM, blocks/SM, smem/SM, regs/SM,
    offset_us of the best window).
    """
    n = len(sw)
    out = []
    for D in durations_us:
        w = max(1, int(round(D * 1000.0 / bin_ns)))
        if w > n:
            out.append((D, 0.0, 0.0, 0.0, 0.0, None))
            continue
        mins = _sliding_min(sw, w)
        j = max(range(len(mins)), key=lambda i: mins[i])
        out.append((D, mins[j],
                    min(sb[j:j + w]), min(ss[j:j + w]), min(sr[j:j + w]),
                    j * bin_ns / 1000.0))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Phantom stream lane
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Phantom:
    """One hypothetical co-tenant kernel placed in the spare envelope."""
    i0: int
    i1: int
    t0: int
    t1: int
    vec: SpareVector
    retired_by: str = ""      # which dimension Aerial reclaimed first
    # Smallest gap between what this kernel holds and what was free, over its
    # whole life, per resource: what integer block placement cannot reach.
    margin: dict = field(default_factory=dict)

    @property
    def dur_us(self) -> float:
        return (self.t1 - self.t0) / 1000.0


def phantom_schedule(sb, sw, ss, sr, bin_ns: int, t0: int, rf: Roofline,
                     min_warps: float = 4.0, B: Optional[int] = None,
                     max_live: int = 1, min_dur_us: float = 10.0,
                     max_dur_us: Optional[float] = None,
                     n_sm: Optional[int] = None):
    """Greedily fill the spare envelope with hypothetical co-tenant kernels.

    A CUDA BLOCK CANNOT BE RESIZED OR EVICTED ONCE RESIDENT, so a phantom
    holding capacity c must END before Aerial reclaims c. Each bar's right
    edge is the instant that kernel would have had to have finished. Sizing
    looks ahead min_dur_us so every phantom could run at least that long.

    This is a feasibility ORACLE: perfect foreknowledge, no launch overhead,
    no scheduler latency, no cache or bandwidth interference. max_live=1
    makes each bar the upper limit; >1 shows how much of the idle GPU is
    reachable at all. max_dur_us reimposes a ceiling (e.g. one TTI).
    """
    n = len(sw)
    look = max(1, int(round(min_dur_us * 1000.0 / bin_ns)))
    cap_bins = (max(1, int(round(max_dur_us * 1000.0 / bin_ns)))
                if max_dur_us else None)
    live = []
    done = []

    def close(rec, i, why):
        j0, j1 = rec["i0"], max(i, rec["i0"] + 1)
        mg = {}
        for nm, arr, held in (("warps", sw, rec["wrp"]),
                              ("blocks", sb, rec["blk"]),
                              ("shared memory", ss, rec["smem"]),
                              ("registers", sr, rec["regs"])):
            seg = arr[j0:min(j1, len(arr))]
            if seg:
                mg[nm] = min(seg) - held
        done.append(Phantom(i0=rec["i0"], i1=i,
                            t0=t0 + rec["i0"] * bin_ns,
                            t1=t0 + i * bin_ns,
                            vec=rec["vec"], retired_by=why, margin=mg))

    for i in range(n):
        if cap_bins:
            keep = []
            for rec in live:
                if i - rec["i0"] >= cap_bins:
                    close(rec, i, "duration cap")
                else:
                    keep.append(rec)
            live = keep

        # retire whatever Aerial has taken the room back from, youngest first
        while live:
            hb = sum(r["blk"] for r in live)
            hw = sum(r["wrp"] for r in live)
            hs = sum(r["smem"] for r in live)
            hr = sum(r["regs"] for r in live)
            why = ("blocks" if hb > sb[i] + 1e-9 else
                   "warps" if hw > sw[i] + 1e-9 else
                   "shared memory" if hs > ss[i] + 1e-9 else
                   "registers" if hr > sr[i] + 1e-9 else None)
            if why is None:
                break
            close(live.pop(), i, why)

        if len(live) >= max_live:
            continue
        held_b = sum(r["blk"] for r in live)
        held_w = sum(r["wrp"] for r in live)
        held_s = sum(r["smem"] for r in live)
        held_r = sum(r["regs"] for r in live)
        j = min(n, i + look)
        hb = min(sb[i:j]) - held_b
        hw = min(sw[i:j]) - held_w
        hs = min(ss[i:j]) - held_s
        hr = min(sr[i:j]) - held_r
        if hw < min_warps or hb < 1.0:
            continue
        v = shape_from_capacity(hb, hw, hs, hr, rf, B=B, n_sm=n_sm)
        if v.k < 1 or v.warps_per_sm < min_warps:
            continue
        wpb = _ceil_to(v.B, rf.warp_size) // rf.warp_size
        rpw = _ceil_to(v.R * rf.warp_size, rf.reg_alloc_unit)
        live.append(dict(i0=i, blk=float(v.k), wrp=float(v.warps_per_sm),
                         smem=float(v.k * rf.block_smem(v.S)),
                         regs=float(v.k * wpb * rpw), vec=v))

    for rec in live:
        close(rec, n, "end of window")
    done.sort(key=lambda ph: (ph.i0, -ph.vec.warps_per_sm))
    return done


# ─────────────────────────────────────────────────────────────────────────────
# The oracle
# ─────────────────────────────────────────────────────────────────────────────
#
# Model. C(t) in R^4 is the per-SM spare (blocks, warps, shared bytes,
# registers), piecewise constant on the boundaries tau_0 < ... < tau_m that
# demand_profile() records. A co-tenant kernel of shape x = (B, S, R) costs
#     h(x) = (1, w, sigma, w*rho)    w = ceil(B/32), sigma = align(S+reserve),
#                                    rho = align(32R, 256)
# per block and per SM, and with k blocks/SM holds k*h(x). A placement
# (x, k, s, D) is feasible iff k*h(x) <= C(t) for all t in [s, s+D): blocks
# cannot be evicted or resized once resident.
#
# Exact oracle quantities:
#     K*(x; s, D) = min_d floor( min_{[s,s+D)} C_d / h_d(x) )
#     D*(x, k; s) = inf{ tau : k*h(x) not<= C(s+tau) }
#
# The lane is the OPTIMAL single-kernel schedule: non-overlapping intervals
# [s_i, e_i) on the boundaries, each with its own shape from the family X
# and k_i = K*(x_i; s_i, e_i - s_i), maximising captured warp-time
#     sum_i  w(x_i) * k_i * (e_i - s_i - lambda)
# where lambda is the relaunch cost (the first lambda of every kernel holds
# its resources but does no work) and e_i - s_i >= D_min. Weighted interval
# scheduling, solved exactly by dynamic programming over the boundaries:
# for a fixed k a kernel's value is linear in its endpoints and C is constant
# between boundaries, so an optimal schedule exists whose every endpoint is a
# boundary (or the previous kernel's end, itself a boundary).
#
# Assumptions, stated wherever a phantom is shown: perfect foreknowledge of
# Aerial's schedule; no scheduler latency or cache/bandwidth interference;
# the four resources are the only interaction; per-SM mean field; the
# co-tenant shares the GPU spatially (same MPS server or a green context).

LAUNCH_GAP_US = 3.0   # p5 back-to-back STREAM launch gap measured on A2
                      # (2.8 us; CUDA-graph nodes are ~0.35 us)
D_MIN_US = 10.0


@dataclass
class CotenantSpec:
    """The co-tenant kernel the oracle places.

    B None searches the block size (32..1024) per phantom; S and R are what
    the co-tenant NEEDS, and are held at that - they do not grow to fill the
    space, which would make every phantom brittle to the smallest change in
    Aerial's demand. How much further they COULD grow is reported per
    phantom as headroom.
    """
    B: Optional[int] = None
    S: int = 0
    R: int = MIN_REGS_PER_THREAD
    launch_gap_us: float = LAUNCH_GAP_US
    d_min_us: float = D_MIN_US

    def blocks(self, rf: Roofline):
        if self.B:
            return (int(self.B),)
        return tuple(b for b in (32, 64, 128, 256, 512, 1024)
                     if b <= min(rf.max_threads_per_block,
                                 rf.max_threads_per_sm))

    def describe(self) -> str:
        b = f"B={self.B}" if self.B else "B searched 32..1024"
        return (f"{b}, S={self.S} B, R={self.R}/thread, relaunch cost "
                f"{self.launch_gap_us:g} us, D_min {self.d_min_us:g} us")

    @classmethod
    def parse(cls, text: str, base: "CotenantSpec" = None) -> "CotenantSpec":
        """'B,S,R' with any field blank or 'auto' kept from `base`."""
        base = base or cls()
        sp = cls(base.B, base.S, base.R, base.launch_gap_us, base.d_min_us)
        parts = [p.strip() for p in (text or "").split(",")]
        for i, attr in enumerate(("B", "S", "R")):
            if i < len(parts) and parts[i] and parts[i].lower() != "auto":
                setattr(sp, attr, int(parts[i]))
            elif i < len(parts) and parts[i].lower() == "auto" and attr == "B":
                sp.B = None
        return sp

    @classmethod
    def from_env(cls) -> "CotenantSpec":
        sp = cls.parse(os.environ.get("TTISCOPE_COTENANT_SHAPE", ""))
        for attr, key in (("launch_gap_us", "TTISCOPE_LAUNCH_GAP_US"),
                          ("d_min_us", "TTISCOPE_DMIN_US")):
            v = os.environ.get(key)
            if v:
                try:
                    setattr(sp, attr, float(v))
                except ValueError:
                    pass
        return sp


def block_cost(rf: Roofline, B: int, S: int, R: int) -> tuple:
    """h(x): one block's cost per SM in (blocks, warps, smem, regs)."""
    w = _ceil_to(int(B), rf.warp_size) // rf.warp_size
    return (1.0, float(w), float(rf.block_smem(S)),
            float(w * _ceil_to(int(R) * rf.warp_size, rf.reg_alloc_unit)))


DIMS = ("blocks", "warps", "shared memory", "registers")


def kstar(cmin, h) -> int:
    """K* for an elementwise-minimum capacity and a block cost."""
    return int(min(c // x for c, x in zip(cmin, h)))


def spare_segments(p: Profile, rf: Roofline, smem_mode: str = "conservative"):
    """(tau[m+1], C[m][4]) from a profile's exact segments, with runs of
    identical spare merged so the DP works on genuine change points only."""
    cons = smem_mode == "conservative"
    tau, C = [], []
    for x, y, b, w, m, r, roof_c in p.segs:
        roof_s = rf.smem_per_sm
        if cons:
            roof_s = min(roof_s, roof_c)
        c = (max(0.0, rf.max_blocks_per_sm - b),
             max(0.0, rf.max_warps_per_sm - w),
             max(0.0, roof_s - m),
             max(0.0, rf.regs_per_sm - r))
        if C and C[-1] == c and tau[-1] == x:
            tau[-1] = y
            continue
        if tau and tau[-1] != x:                 # never happens; be safe
            C.append((0.0, 0.0, 0.0, 0.0))
            tau.append(x)
        if not tau:
            tau.append(x)
        C.append(c)
        tau.append(y)
    return tau, C


def oracle_schedule(tau, C, rf: Roofline, spec: CotenantSpec,
                    n_sm: Optional[int] = None):
    """Optimal single-co-tenant schedule over the envelope (Algorithm 3).

    Returns (phantoms_as_dicts, captured, available): captured is the DP's
    objective value in warp-ns/SM; available is the envelope's total spare
    warp-ns/SM, for an efficiency figure.
    """
    import numpy as np
    m = len(C)
    if m == 0:
        return [], 0.0, 0.0
    T = np.asarray(tau, dtype=np.float64)
    Cm = np.asarray(C, dtype=np.float64)
    lam = spec.launch_gap_us * 1000.0
    dmin = spec.d_min_us * 1000.0
    # Largest B first: np.argmax returns the FIRST maximum, so on a tie in
    # warp-time the kernel with fewer, larger blocks wins - the same rule
    # oracle_shape() applies, since block slots are the scarcer resource
    # beside Aerial's many small grids.
    shapes = [(B,) + block_cost(rf, B, spec.S, spec.R)
              for B in sorted(spec.blocks(rf), reverse=True)]
    H = np.asarray([s[1:] for s in shapes])            # (n_shape, 4)
    W = H[:, 1]
    V = np.zeros(m + 1)
    P = [None] * (m + 1)
    for e in range(1, m + 1):
        V[e], P[e] = V[e - 1], None
        rev = np.minimum.accumulate(Cm[e - 1::-1], axis=0)   # s = e-1 .. 0
        s_idx = np.arange(e - 1, -1, -1)
        lens = T[e] - T[s_idx]
        ok = (lens >= dmin) & (lens > lam)
        if not ok.any():
            continue
        # k[shape, s] = min_d floor(Cmin_d / h_d)
        k = np.floor(rev[None, :, :] / H[:, None, :]).min(axis=2)
        val = V[s_idx][None, :] + W[:, None] * k * (lens - lam)[None, :]
        val = np.where((k >= 1) & ok[None, :], val, -np.inf)
        j = np.unravel_index(int(np.argmax(val)), val.shape)
        if val[j] > V[e] + 1e-9:
            V[e] = val[j]
            P[e] = (int(s_idx[j[1]]), int(j[0]), int(k[j]))
    out = []
    e = m
    while e > 0:
        if P[e] is None:
            e -= 1
            continue
        s, si, k = P[e]
        out.append(dict(s=s, e=e, shape=shapes[si], k=k))
        e = s
    out.reverse()
    avail = float(np.sum(Cm[:, 1] * np.diff(T)))
    return out, float(V[m]), avail


def oracle_phantoms(tau, C, rf: Roofline, spec: CotenantSpec, n_sm: int,
                    clip=None, bin_ns: int = 1000, t_origin: int = 0):
    """Run the oracle and turn its intervals into Phantom records.

    clip (a, b): phantoms are computed on a padded envelope so the window's
    edges do not decide them, then clipped; one cut by the clip says so.
    """
    sched, captured, avail = oracle_schedule(tau, C, rf, spec, n_sm)
    res = []
    m = len(C)
    for it in sched:
        s, e, k = it["s"], it["e"], it["k"]
        B, *h = it["shape"]
        cmin = [min(C[j][d] for j in range(s, e)) for d in range(4)]
        # why it ends: extending one more segment would cost a block of k
        if e < m:
            ext = [min(cmin[d], C[e][d]) for d in range(4)]
            k_ext = kstar(ext, h)
            if k_ext < k:
                why = DIMS[min(range(4), key=lambda d: ext[d] // h[d])]
            else:
                why = "relaunch"
        else:
            why = "end of window"
        # D*: how long THIS k could have run from s
        j, cm = s, list(C[s])
        while j < m and kstar([min(cm[d], C[j][d]) for d in range(4)], h) >= k:
            cm = [min(cm[d], C[j][d]) for d in range(4)]
            j += 1
        t0, t1 = tau[s], tau[e]
        if clip is not None:
            if t1 <= clip[0] or t0 >= clip[1]:
                continue
            if t1 > clip[1]:
                why = "end of window"            # continues past the view
            t0, t1 = max(t0, clip[0]), min(t1, clip[1])
        w = int(h[1])
        v = SpareVector(B=B, S=spec.S, R=spec.R, k=k, n_sm=n_sm, G=k * n_sm,
                        warps_per_sm=float(k * w),
                        occ_pct=100.0 * k * w / rf.max_warps_per_sm,
                        limiter=DIMS[min(range(4), key=lambda d: cmin[d] // h[d])],
                        feasible=True, smem_optin=spec.S > SMEM_DEFAULT_LIMIT)
        v.D = (t1 - t0) / 1000.0
        v.d_safe_us = (tau[j] - tau[s]) / 1000.0
        margin = {DIMS[d]: cmin[d] - k * h[d] for d in range(4)}
        # headroom: how far S and R could grow for this interval and k
        v.S_room = int(_floor_to(int(margin["shared memory"] // k),
                                 rf.smem_alloc_unit)) if k else 0
        rpw_room = margin["registers"] / (k * w) if k else 0
        v.R_room = int(rpw_room // rf.reg_alloc_unit) * (
            rf.reg_alloc_unit // rf.warp_size)
        i0 = max(0, -(-(int(t0) - t_origin) // bin_ns))
        i1 = max(i0 + 1, (int(t1) - t_origin) // bin_ns)
        res.append(Phantom(i0=i0, i1=i1, t0=int(t0), t1=int(t1), vec=v,
                           retired_by=why, margin=margin))
    return res, captured, avail


# ─────────────────────────────────────────────────────────────────────────────
# Cached per-store context
# ─────────────────────────────────────────────────────────────────────────────

_CTX = {}


@dataclass
class SpareContext:
    """Everything constant for a store: roofline, telemetry, MPS caps, the
    DCGM validation and the shape cache. Built once, because the TTI tab
    asks for the phantom lane on every pan.

    `apply_scale` and `cotenant_sms` are user settings and may be changed on
    the cached object; everything else is a property of the capture.
    """
    rf: Roofline
    dcgm: DcgmSeries
    caps: dict = field(default_factory=dict)      # ctx -> SM cap
    cap_rows: dict = field(default_factory=dict)  # ctx -> (cap,label,src,ok)
    query: str = ""
    lookback_ns: int = 10_000_000
    shapes: object = None
    # validation against DCGM, over the same wall-clock interval
    model_occ: Optional[float] = None
    dcgm_occ: Optional[float] = None
    dcgm_n: int = 0
    val_window: tuple = (0, 0)
    scale: float = 1.0            # dcgm_occ / model_occ, applied only if asked
    apply_scale: bool = False
    cotenant_sms: Optional[int] = None
    spec: CotenantSpec = field(default_factory=CotenantSpec.from_env)
    max_cluster: int = 1
    unshaped_frac: float = 0.0

    # kept for callers that read the old attribute names
    @property
    def anchor_src(self) -> str:
        return MEASURED if (self.apply_scale and self.dcgm_occ) else MODELLED

    @property
    def raw_occ_mean(self) -> float:
        return self.model_occ or 0.0

    @property
    def n_sm_cotenant(self) -> int:
        return int(self.cotenant_sms or self.rf.num_sm)

    def caps_note(self) -> str:
        cap = [(k, r) for k, r in sorted(self.cap_rows.items()) if r[0]]
        if not cap:
            return ("NO SM CAPS - no [DRV.CTX] lines or green contexts; every "
                    "context treated as able to use all "
                    f"{self.rf.num_sm} SMs, which overstates residency")
        return "SM caps " + ", ".join(
            f"{r[1] or ('ctx%d' % k)} {r[0]}" + ("" if r[3] else "?")
            for k, r in cap)

    def validation_note(self) -> str:
        if self.model_occ is None:
            return "no model validation"
        if self.dcgm_occ is None:
            return (f"model occupancy {self.model_occ:.3f}; no DCGM "
                    f"sm_occupancy to check it against")
        st = "APPLIED" if self.apply_scale else "not applied"
        return (f"model {self.model_occ:.3f} vs DCGM {self.dcgm_occ:.3f} "
                f"({self.dcgm_n} samples, same interval): "
                f"x{self.scale:.2f} {st}")

    def note(self) -> str:
        bits = [self.validation_note()]
        if self.max_cluster > 1:
            bits.append(f"[!] clusters of {self.max_cluster} CTAs present - "
                        f"not modelled")
        if self.unshaped_frac > 0.01:
            bits.append(f"{100*self.unshaped_frac:.0f}% of busy time has no "
                        f"register data")
        return "; ".join(bits)


def context(st, bin_ns: int = 1000, roofline_file=None,
            validate_s: float = 1.0) -> SpareContext:
    """Build (once per store) the roofline, caps and DCGM validation.

    Validation runs the model over `validate_s` seconds at the middle of the
    traced span and compares its time-weighted mean warp occupancy with the
    mean DCGM sm_occupancy over exactly that interval. The middle, because
    the head and tail of a sweep are ramp-up and tear-down.
    """
    key = (id(st), str(roofline_file))
    hit = _CTX.get(key)
    if hit is not None:
        return hit
    dcgm = DcgmSeries.load(st._meta.get("capture_dir"), st.utc_epoch_ns)
    rf = (Roofline.from_file(roofline_file) if roofline_file else None) \
        or Roofline.from_store(st, dcgm)
    if rf.fb_total_mib is None and dcgm.ok:
        fb = dcgm.median("fb_total_mib")
        if fb:
            rf.fb_total_mib = int(fb)
    cap_rows = st.context_caps() if hasattr(st, "context_caps") else {}
    ctx = SpareContext(
        rf=rf, dcgm=dcgm, cap_rows=cap_rows,
        caps={k: r[0] for k, r in cap_rows.items() if r[0]},
        query=_event_query(st),
        lookback_ns=max(2_000_000, int(getattr(st, "max_dur_ns", 0) or 0)
                        + 1000),
        shapes=_ShapeCache(rf),
        max_cluster=int(getattr(st, "max_cluster", 1) or 1))
    env = os.environ.get("TTISCOPE_COTENANT_SMS")
    if env and env.isdigit():
        ctx.cotenant_sms = int(env)

    if st.n_slots:
        first, last = st.slot(0), st.slot(st.n_slots - 1)
        span = last[2] - first[1]
        w = min(span, int(validate_s * 1e9))
        a = first[1] + (span - w) // 2
        b = a + w
        prof = demand_profile(st, a, b, ctx, bin_ns=100_000)
        ctx.model_occ = prof.mean_occ
        ctx.unshaped_frac = (prof.unshaped_ns / prof.busy_ns
                             if prof.busy_ns else 0.0)
        ctx.val_window = (a, b)
        if dcgm.ok:
            m, k = dcgm.mean("sm_occupancy", a, b)
            ctx.dcgm_occ, ctx.dcgm_n = m, k
            if m is not None and prof.mean_occ > 1e-9:
                ctx.scale = m / prof.mean_occ
    _CTX[key] = ctx
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Top-level analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpareReport:
    rf: Roofline
    dcgm: DcgmSeries
    prof: Profile                   # FOLDED shape, for plotting
    spare: tuple                    # per-bin spare of the folded profile
    frontier: list                  # [(D_us, p5, p50, p95, shape_at_p5)]
    sustained: SpareVector          # fits in EVERY TTI at the p5 level
    peak: SpareVector               # fits at the most idle instant
    phantoms: list
    n_tti: int = 0
    win_spare: tuple = ()           # per-bin spare of the real (unfolded) window
    win_t0: int = 0
    win_bin_ns: int = 1000
    ctx: Optional[SpareContext] = None
    smem_mode: str = "conservative"
    oracle_captured: float = 0.0     # warp-ns/SM the oracle lane captures
    oracle_avail: float = 0.0        # warp-ns/SM of spare in the window

    def summary_lines(self) -> list:
        c = self.ctx
        L = [f"Roofline   {self.rf.describe()}",
             f"DCGM       {self.dcgm.describe()}",
             f"Model      {self.prof.provenance()}"]
        if c is not None:
            L += [f"MPS        {c.caps_note()}",
                  f"Check      {c.note()}",
                  f"Co-tenant  may use {c.n_sm_cotenant} SMs "
                  f"(set TTISCOPE_COTENANT_SMS to its own MPS cap)"]
        L += [f"Shared mem {self.smem_mode} - S is net of the "
              f"{self.rf.reserved_smem_per_block} B per-block reserve; "
              f"(opt N) shows the optimistic carveout",
              f"Sampled    {self.n_tti} TTI · {self.prof.n_launch} launches", ""]
        L.append("Sustained spare, whole TTI, THIS slot only"
                 if self.n_tti <= 1 else
                 "Sustained spare, whole TTI, p5 across TTI")
        L.append(f"   {self.sustained.describe()}")
        L.append("Peak spare, single instant")
        L.append(f"   {self.peak.describe()}")
        return L


def _pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(q * (len(xs) - 1))))]


def fold_profile(st, lo_slot: int, hi_slot: int, ctx: SpareContext,
                 bin_ns: int, max_tti: int = 200) -> Profile:
    """Average the per-bin demand of every TTI in [lo,hi] onto one TTI-length
    canvas: the SHAPE of the pipeline inside a slot. Not for hard bounds -
    the minimum of an average is optimistic; analyse() uses per-TTI minima."""
    dur = st.slot_dur_ns
    n = max(1, int(dur // bin_ns))
    acc = {k: [0.0] * n for k in ("blk", "wrp", "smem", "regs", "cfg")}
    cnt = n_launch = 0
    occ = 0.0
    for idx in list(range(lo_slot, hi_slot + 1))[:max_tti]:
        r = st.slot(idx)
        if not r:
            continue
        p = demand_profile(st, r[1], r[2], ctx, bin_ns)
        if p.n < n:
            continue
        for k in acc:
            src = getattr(p, k)
            a = acc[k]
            for i in range(n):
                a[i] += src[i]
        n_launch += p.n_launch
        occ += p.mean_occ
        cnt += 1
    if cnt:
        for a in acc.values():
            for i in range(n):
                a[i] /= cnt
    p = Profile(bin_ns=bin_ns, t0=0, t1=dur, n_launch=n_launch,
                mean_occ=occ / cnt if cnt else 0.0, **acc)
    p.raw_occ_mean = p.mean_occ
    if ctx.apply_scale:
        p.scale, p.anchor_src, p.dcgm_occ = ctx.scale, MEASURED, ctx.dcgm_occ
    return p


def oracle_shape(cmin, rf: Roofline, spec: CotenantSpec,
                 n_sm: int) -> SpareVector:
    """The largest kernel of the co-tenant family that fits a capacity
    held for the whole interval: max over B of w(B)*K*(B,S,R), ties toward
    fewer blocks. S and R are the co-tenant's own; S_room/R_room say how far
    they could grow at that k."""
    best = None
    for B in spec.blocks(rf):
        h = block_cost(rf, B, spec.S, spec.R)
        k = kstar(cmin, h)
        if k >= 1 and (best is None or
                       (k * h[1], -k) > (best[1] * best[2][1], -best[1])):
            best = (B, k, h)
    v = SpareVector(S=spec.S, R=spec.R, n_sm=n_sm)
    if best is None:
        hs = [block_cost(rf, B, spec.S, spec.R) for B in spec.blocks(rf)]
        h = min(hs, key=lambda h: h[1])
        v.reason = DIMS[min(range(4), key=lambda d: cmin[d] / h[d])]
        return v
    B, k, h = best
    w = int(h[1])
    mg = [cmin[d] - k * h[d] for d in range(4)]
    v.B, v.k, v.G = B, k, k * n_sm
    v.warps_per_sm = float(k * w)
    v.occ_pct = 100.0 * k * w / rf.max_warps_per_sm
    v.limiter = DIMS[min(range(4), key=lambda d: cmin[d] // h[d])]
    v.smem_optin = spec.S > SMEM_DEFAULT_LIMIT
    v.S_room = int(_floor_to(int(mg[2] // k), rf.smem_alloc_unit))
    v.R_room = int((mg[3] / (k * w)) // rf.reg_alloc_unit) * (
        rf.reg_alloc_unit // rf.warp_size)
    v.feasible = True
    return v


def window_oracle(st, t0: int, t1: int, ctx: SpareContext, bin_ns: int,
                  smem_mode: str = "conservative", pad_ns: int = None):
    """Oracle phantoms for [t0,t1), computed on an envelope padded by one
    TTI each side so the view's edges do not decide where kernels start."""
    pad = st.slot_dur_ns if pad_ns is None else pad_ns
    pp = demand_profile(st, t0 - pad, t1 + pad, ctx, bin_ns=max(bin_ns, 10_000))
    tau, C = spare_segments(pp, ctx.rf, smem_mode)
    ph, cap, avail = oracle_phantoms(tau, C, ctx.rf, ctx.spec,
                                     ctx.n_sm_cotenant, clip=(t0, t1),
                                     bin_ns=bin_ns, t_origin=t0)
    return ph, cap, avail


def analyse(st, lo_slot: int, hi_slot: int, ctx: SpareContext = None,
            bin_ns: int = 1000, B: Optional[int] = None,
            want_phantom: bool = True, max_tti: int = 200,
            durations_us=(10, 25, 50, 100, 150, 200, 250, 300, 400, 500),
            smem_mode: str = "conservative"):
    """End-to-end spare-GPU analysis for a slot range.

    The frontier is computed PER TTI and then aggregated across TTIs: a
    co-tenant has to survive the WORST TTI it meets, not the average one.
    Percentiles are taken per dimension, so the shape is a per-dimension
    bound, not a joint p5.
    """
    s_lo, s_hi = st.slot(lo_slot), st.slot(hi_slot)
    if not s_lo or not s_hi:
        return None
    ctx = ctx or context(st, bin_ns=bin_ns)
    rf, dcgm = ctx.rf, ctx.dcgm
    nsm = ctx.n_sm_cotenant
    other = "optimistic" if smem_mode == "conservative" else "conservative"

    prof = fold_profile(st, lo_slot, hi_slot, ctx, bin_ns, max_tti=max_tti)
    sb, sw, ss, sr = spare_profile(prof, rf, smem_mode)

    per_D = {D: [] for D in durations_us}
    shp_D = {D: [] for D in durations_us}
    n_tti = 0
    for idx in list(range(lo_slot, hi_slot + 1))[:max_tti]:
        r = st.slot(idx)
        if not r:
            continue
        pr = demand_profile(st, r[1], r[2], ctx, bin_ns)
        if pr.n < 2:
            continue
        b2, w2, s2, r2 = spare_profile(pr, rf, smem_mode)
        s2o = spare_profile(pr, rf, other)[2]
        fr = capacity_duration_frontier(w2, b2, s2, r2, bin_ns, rf,
                                        durations_us)
        fro = capacity_duration_frontier(w2, b2, s2o, r2, bin_ns, rf,
                                         durations_us)
        for (D, wv, bv, sv, rv, _o), (_D, _w, _b, svo, _r, _oo) in zip(fr, fro):
            per_D[D].append(wv)
            shp_D[D].append((bv, wv, sv, rv, svo))
        n_tti += 1

    frontier = []
    for D in durations_us:
        vals = per_D[D]
        p5, p50, p95 = _pct(vals, .05), _pct(vals, .50), _pct(vals, .95)
        cols = (list(zip(*shp_D[D])) if shp_D[D]
                else ((0,), (0,), (0,), (0,), (0,)))
        pick = tuple(_pct(list(c), .05) for c in cols)
        spec = ctx.spec if not B else CotenantSpec.parse(str(B), ctx.spec)
        v = oracle_shape((pick[0], pick[1], pick[2], pick[3]), rf, spec, nsm)
        vo = oracle_shape((pick[0], pick[1], pick[4], pick[3]), rf, spec, nsm)
        v.S_opt = (spec.S + vo.S_room) if vo.feasible else None
        v.D = float(D)
        frontier.append((D, p5, p50, p95, v))

    full = st.slot_dur_ns / 1000.0
    sustained = next((f[4] for f in frontier if f[0] == max(durations_us)),
                     SpareVector())
    sustained.D = full
    if sw:
        j = max(range(len(sw)), key=lambda i: sw[i])
        peak = oracle_shape((sb[j], sw[j], ss[j], sr[j]), rf, ctx.spec, nsm)
        peak.D = bin_ns / 1000.0
    else:
        peak = SpareVector()

    fb_free = dcgm.median("fb_free_mib") if dcgm.ok else None

    def _finish(v):
        v.P = full
        if fb_free is not None:
            v.M, v.src_M = int(fb_free), MEASURED
        v.src_level = ctx.anchor_src
        if v.P > 0:
            v.duty = (v.n * v.D) / v.P
            v.f_hz = 1e6 / v.P

    _finish(sustained)
    _finish(peak)
    for _D, _a, _b, _c, v in frontier:
        _finish(v)

    # phantom lane over the REAL window, not the fold
    ph, win, wt0 = [], (), s_lo[1]
    if want_phantom:
        wp = demand_profile(st, s_lo[1], s_hi[2], ctx, bin_ns)
        win = spare_profile(wp, rf, smem_mode)
        ph, captured, avail = window_oracle(st, s_lo[1], s_hi[2], ctx,
                                            bin_ns, smem_mode)
        for pp in ph:
            _finish(pp.vec)
            pp.vec.D = pp.dur_us
            pp.vec.duty = pp.vec.D / full if full else 0.0

    rep = SpareReport(rf, dcgm, prof, (sb, sw, ss, sr), frontier,
                      sustained, peak, ph, n_tti=n_tti,
                      win_spare=win, win_t0=wt0, win_bin_ns=bin_ns,
                      ctx=ctx, smem_mode=smem_mode)
    if want_phantom:
        rep.oracle_captured, rep.oracle_avail = captured, avail
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# The light path the phantom lane uses on every redraw
# ─────────────────────────────────────────────────────────────────────────────

def window_spare(st, t0: int, t1: int, ctx: SpareContext, bin_ns: int = 1000,
                 smem_mode: str = "conservative"):
    """Aerial's demand and the spare envelope across one wall-clock window.
    No per-TTI frontier: this is the path every redraw takes."""
    p = demand_profile(st, t0, t1, ctx, bin_ns)
    return p, spare_profile(p, ctx.rf, smem_mode)


def window_phantoms(st, t0: int, t1: int, ctx: SpareContext,
                    bin_ns: int = 1000, B: Optional[int] = None,
                    min_dur_us: float = 10.0, max_live: int = 1,
                    max_dur_us: Optional[float] = None,
                    smem_mode: str = "conservative"):
    """max_live == 1: the exact oracle (optimal single co-tenant schedule).
    max_live > 1: the greedy stacked fill, a HEURISTIC - no optimality claim.
    Sets p.oracle = (captured, available) warp-ns/SM in oracle mode."""
    p, (sb, sw, ss, sr) = window_spare(st, t0, t1, ctx, bin_ns, smem_mode)
    P = st.slot_dur_ns / 1000.0
    if max_live == 1:
        ph, cap, avail = window_oracle(st, t0, t1, ctx, bin_ns, smem_mode)
        p.oracle = (cap, avail)
    else:
        ph = phantom_schedule(sb, sw, ss, sr, bin_ns, t0, ctx.rf, B=B,
                              max_live=max_live, min_dur_us=min_dur_us,
                              max_dur_us=max_dur_us, n_sm=ctx.n_sm_cotenant)
        p.oracle = None
    fb = ctx.dcgm.median("fb_free_mib") if ctx.dcgm.ok else None
    for q in ph:
        q.vec.D = q.dur_us
        q.vec.P = P
        if fb is not None:
            q.vec.M, q.vec.src_M = int(fb), MEASURED
        q.vec.src_level = ctx.anchor_src
        q.vec.duty = q.vec.D / P if P else 0.0
        q.vec.f_hz = 1e6 / P if P else 0.0
    return ph, (sb, sw, ss, sr), p


def auto_bin_ns(t0: int, t1: int, target_bins: int = 5000,
                floor_ns: int = 1000) -> int:
    """Bin width for an interactive redraw: never finer than one bin per
    pixel is useful, never coarser than needed. Bins hold the MAXIMUM demand
    of their span, so a coarser bin is conservative, never optimistic."""
    span = max(1, int(t1) - int(t0))
    b = span // max(1, target_bins)
    b = (b // 1000) * 1000
    return max(floor_ns, b)
