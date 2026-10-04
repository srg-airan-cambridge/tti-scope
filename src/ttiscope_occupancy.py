#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
Theoretical SM occupancy and the energy model.

Theoretical Occupancy is never accurate so we didn't report it. Energy-line, specially per Energy per TTI-slot is computed
rather than measured. 

    E_slot = P_IDLE * slot_duration
           + SUM_k (P_COMPUTE*wf_k + P_MEM_BW*mf_k) * busy_k
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Device limits
# ─────────────────────────────────────────────────────────────────────────────

# sm_90 (GH100 die — the GH200 480GB in these captures reports chipName GH100).
# Values are the CUDA occupancy calculator's device constants for compute
# capability 9.0.
SM90 = dict(
    name="sm_90",
    max_threads_per_sm=2048,
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    regs_per_sm=65536,
    max_regs_per_thread=255,
    reg_alloc_unit=256,        # registers are allocated to a warp in units of 256
    warp_alloc_granularity=4,  # ...and warps in groups of 4
    smem_per_sm=233472,        # 228 KB, the opt-in maximum on H100/GH200
    smem_alloc_unit=128,
    warp_size=32,
    max_threads_per_block=1024,
    # cudaDevAttrReservedSharedMemoryPerBlock: the runtime holds 1 KiB of
    # shared memory per resident block on sm_80+. It is why the opt-in
    # per-block maximum (232,448) is exactly 1 KiB under the per-SM total
    # (233,472) - both read straight out of TARGET_INFO_GPU in the GH200
    # captures. Every resident block costs smem + 1 KiB against the SM.
    reserved_smem_per_block=1024,
    smem_per_block_optin=232448,
    # Legal L1/shared carveouts on sm_90, bytes per SM. The driver configures
    # one of these per launch (CUPTI sharedMemoryExecuted); the rest of the
    # 256 KB unified array is L1.
    carveouts=tuple(k * 1024 for k in (0, 8, 16, 32, 64, 100, 132, 164,
                                       196, 228)),
)

LIMITER = ("warps", "blocks", "registers", "shared memory")


def _ceil_to(x: int, unit: int) -> int:
    return ((x + unit - 1) // unit) * unit


def _floor_to(x: int, unit: int) -> int:
    return (x // unit) * unit


def block_smem_bytes(smem: int, dev=SM90) -> int:
    """Shared memory one resident block holds against its SM: its own
    static + dynamic bytes plus the runtime reserve, in allocation units."""
    return _ceil_to(int(smem or 0) + dev.get("reserved_smem_per_block", 0),
                    dev["smem_alloc_unit"])


def smallest_carveout(need: int, dev=SM90) -> int:
    """The smallest legal carveout holding `need` bytes per SM."""
    for c in dev.get("carveouts") or (dev["smem_per_sm"],):
        if c >= need:
            return c
    return dev["smem_per_sm"]


def theoretical_occupancy(tpb: int, regs: int, smem: int, dev=SM90) -> tuple:
    """(occupancy_pct, active_blocks_per_sm, active_warps_per_sm, limiter).

    `limiter` names the resource that caps it — the same annotation Nsight's
    occupancy view gives, and the only part of the number that is actionable.
    """
    tpb = int(tpb or 0)
    if tpb <= 0 or tpb > dev.get("max_threads_per_block", 1024):
        return 0.0, 0, 0, "invalid"
    warps_per_block = _ceil_to(tpb, dev["warp_size"]) // dev["warp_size"]
    if warps_per_block <= 0:
        return 0.0, 0, 0, "invalid"

    limits = [
        dev["max_warps_per_sm"] // warps_per_block,   # warps
        dev["max_blocks_per_sm"],                     # blocks
    ]

    regs = int(regs or 0)
    if regs > 0:
        regs_per_warp = _ceil_to(regs * dev["warp_size"], dev["reg_alloc_unit"])
        warps_by_regs = _floor_to(dev["regs_per_sm"] // regs_per_warp,
                                  dev["warp_alloc_granularity"])
        limits.append(warps_by_regs // warps_per_block)
    else:
        limits.append(dev["max_blocks_per_sm"])

    # Every block holds its own shared memory plus the runtime's per-block
    # reserve, so even a kernel with none still costs 1 KiB per block. That
    # never binds at 32 blocks/SM against 228 KB, but it does once the SM is
    # configured to a small carveout, and it matters for every kernel that
    # does use shared memory: the README's worked example falls from 12 to 11
    # blocks/SM (18.8% -> 17.2%) once it is charged.
    smem_block = block_smem_bytes(smem, dev)
    limits.append(dev["smem_per_sm"] // smem_block if smem_block
                  else dev["max_blocks_per_sm"])

    active_blocks = min(limits)
    if active_blocks <= 0:
        # A launch that cannot place even one block per SM. Real, and worth
        # showing as 0 rather than crashing or clamping to 1.
        return 0.0, 0, 0, LIMITER[limits.index(min(limits))]
    active_warps = active_blocks * warps_per_block
    occ = 100.0 * active_warps / dev["max_warps_per_sm"]
    return occ, active_blocks, active_warps, LIMITER[limits.index(active_blocks)]


# ─────────────────────────────────────────────────────────────────────────────
# Static resources from cuobjdump
# ─────────────────────────────────────────────────────────────────────────────

# Set TTISCOPE_KERNEL_RESOURCES to point elsewhere.
DEFAULT_RESOURCE_CSV = (Path.home() / ".config" / "tti-scope"
                        / "all_kernel_resources_sm90.csv")


def load_static_resources(path=None) -> dict:
    """kernel name -> (max_regs_per_thread, max_static_smem_bytes).

    Produced by dump_all_kernel_resources.py (cuobjdump -res-usage over the
    built libcuphy/libcuphydriver/libnvipc .so files). Optional: without it,
    KTRACE-only kernels simply have no occupancy figure, which is reported as
    unknown rather than guessed.
    """
    p = Path(path or os.environ.get("TTISCOPE_KERNEL_RESOURCES")
             or DEFAULT_RESOURCE_CSV)
    if not p.exists():
        return {}
    out = {}
    try:
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    out[row["kernel_name"]] = (
                        int(row["max_regs_per_thread"]),
                        int(row["max_static_smem_bytes"]))
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return out



# Power / energy. Optimized for GH200. GB10 will need attention

P_IDLE = 72.0
P_MEM_BW = 120.0
P_TDP = float(os.environ.get("TTISCOPE_TDP_W",
                           os.environ.get("AIRAN_TDP_W", 700.0)))
P_COMPUTE = P_TDP - P_IDLE - P_MEM_BW          # 508 W on a 700 W part


def access_rates(name: str, warps: int, max_warps: int) -> tuple:
    """(wf, mf) — compute and memory access rates, both in [0,1]."""
    wf = min((warps or 0) / float(max_warps or 1), 1.0)
    n = name or ""
    mf = (wf * 0.8 if "memset" in n else
          0.60 if "compress" in n else
          0.30 if "prepare" in n else
          wf * 0.20)
    return wf, min(mf, 1.0)


def dynamic_watts(name: str, warps: int, max_warps: int) -> float:
    """Marginal power of a kernel ABOVE the idle floor.

    The floor is charged once per slot by slot_energy_uj, not once per kernel;
    summing a P_IDLE-inclusive figure over 24 concurrent streams would make
    the per-slot energy several times too large.
    """
    wf, mf = access_rates(name, warps, max_warps)
    return P_COMPUTE * wf + P_MEM_BW * mf


def slot_energy_uj(slot_dur_ns: int, dynamic_watt_ns: float) -> float:
    """Energy in microjoules.

    dynamic_watt_ns must be the INTEGRAL of instantaneous dynamic power, with
    the concurrent sum already clamped to (P_TDP - P_IDLE) — see the sweep in
    ttiscope_ingest._rollup. Naively summing each kernel's power over its own
    duration instead produced a mean of 892 W on a 700 W part, because a GPU
    running twenty concurrent kernels does not draw the sum of what each would
    draw alone; it draws what the board permits.

    1 W*ns = 1 nJ = 1e-3 uJ.
    """
    idle_nj = P_IDLE * float(slot_dur_ns)
    return (idle_nj + float(dynamic_watt_ns)) * 1e-3


def energy_provenance() -> str:
    return (f"Estimated, not measured — no capture here was taken with "
            f"--gpu-metrics-device.\n"
            f"E_slot = P_idle x slot + SUM_k (P_compute*wf + P_mem*mf) x busy_k\n"
            f"P_idle {P_IDLE:.0f} W (Antepara SC'25 Tab.2, GH200 P_const)   "
            f"P_mem {P_MEM_BW:.0f} W (H100 power-capping study)   "
            f"P_TDP {P_TDP:.0f} W   P_compute {P_COMPUTE:.0f} W (derived)\n"
            f"wf = warp residency; mf = name-based heuristic — unmeasurable "
            f"under MPS, which cuPHY requires.")


if __name__ == "__main__":
    res = load_static_resources()
    print(f"{len(res)} kernels in the static resource table\n")
    print(f"{'kernel':<40}{'tpb':>5}{'regs':>6}{'smem':>7}"
          f"{'occ%':>7}{'blk/SM':>8}  limiter")
    for name in ("crcUplinkPuschCodeBlocksKernel", "eqMmseSoftDemapKernel",
                 "rm_decoder_3", "listPolarDecoderKernel",
                 "ldpc_encode_in_bit_kernel", "memset_kernel"):
        r, s = res.get(name, (0, 0))
        for tpb in (128, 256):
            occ, blk, _w, lim = theoretical_occupancy(tpb, r, s)
            print(f"{name:<40}{tpb:>5}{r:>6}{s:>7}{occ:>7.1f}{blk:>8}  {lim}")
    print()
    print(energy_provenance())
