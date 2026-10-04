#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
TTI-Scope kernel taxonomy — all 59 kernels observed executing in a 20-cell 59c
capture, mapped to 5G NR channel and pipeline role.

"""
from __future__ import annotations

import re

# channel -> display colour
CHANNEL_COLOR = {
    "FH-DL":   "#7F77DD",
    "FH-UL":   "#5D55BB",
    "PDSCH":   "#378ADD",
    "PDCCH":   "#4FA3E8",
    "PUSCH":   "#D4537E",
    "PUCCH":   "#BA7517",
    "PRACH":   "#1D9E75",
    "CSI-RS":  "#59C3A5",
    "SSB":     "#9FE1CB",
    "UCI":     "#E07A5F",
    "Graph":   "#B4B2A9",
    "Utility": "#B4B2A9",
    "Unknown": "#888780",
}

# name -> (channel, role)
KERNEL_TAXONOMY: dict[str, tuple[str, str]] = {

    # ── Fronthaul DL (DOCA GPUNetIO) ────────────────────────────────────────
    "gpu_comm_pre_prepare_send_doca": ("FH-DL", "Pre-stage TX descriptors"),
    "gpu_comm_prepare_send_doca":     ("FH-DL", "Finalize TX descriptors"),
    "gpu_comm_trigger_send_doca_cx7": ("FH-DL", "DMA doorbell -> ConnectX-7"),
    "kernel_compress":                ("FH-DL", "BFP IQ compression (O-RAN WG4 7.7)"),
    "kernel_order":                   ("FH-DL", "RE ordering into eCPRI layout"),
    "kernel_write":                   ("FH-DL", "Completion flag write"),

    # ── Fronthaul UL ────────────────────────────────────────────────────────
    "order_kernel_doca_single_subSlot_pingpong":
        ("FH-UL", "GPU-resident sub-slot ping-pong (NIC poll)"),

    # ── PDSCH (DL shared channel) ───────────────────────────────────────────
    "prepare_crc_buffers":                  ("PDSCH", "CRC buffer init"),
    "crcDownlinkPdschTransportBlockKernel": ("PDSCH", "TB CRC-24A (TS 38.212 5.1)"),
    "crcDownlinkPdschCodeBlocksKernel":     ("PDSCH", "CB CRC-24B after segmentation"),
    "ldpc_encode_in_bit_kernel":            ("PDSCH", "LDPC encode BG1/BG2"),
    "fused_dl_rm_and_modulation":           ("PDSCH", "Rate-match + scramble + modulate"),
    "fused_dmrs":                           ("PDSCH", "DMRS generation (TS 38.211 7.4.1)"),

    # ── PDCCH (DL control) ──────────────────────────────────────────────────
    "genPdcchTfSignalKernel":            ("PDCCH", "TF-signal mapping"),
    "genScramblingSeqKernel":            ("PDCCH", "Scrambling sequence"),
    "encodeRateMatchMultipleDCIsKernel": ("PDCCH", "Polar encode + rate-match DCIs"),

    # ── CSI-RS ──────────────────────────────────────────────────────────────
    "genCsirsTfSignalKernel": ("CSI-RS", "CSI-RS TF-signal generation"),
    "genScramblingKernel":    ("CSI-RS", "CSI-RS scrambling"),
    "genCsirsReMap":          ("CSI-RS", "RE mapping"),
    "postProcessCsirsReMap":  ("CSI-RS", "RE-map post-process"),
    "zero_memset_kernel":     ("CSI-RS", "RE-map buffer clear"),

    # ── SSB ─────────────────────────────────────────────────────────────────
    "ssbModTfSigKernel":                ("SSB", "SSB modulation + TF mapping"),
    "encodeRateMatchMultipleSSBsKernel": ("SSB", "Polar encode + rate-match PBCH"),

    # ── PUSCH receive chain — KTRACE-only (device-launched graph) ──────────
    "windowedChEstPreNoDftSOfdmKernel":   ("PUSCH", "Channel est. pre-stage (windowed)"),
    "chEstFilterNoDftSOfdmDispatchKernel": ("PUSCH", "Channel est. filter dispatch"),
    "noiseIntfEstNoDftSOfdmKernel":       ("PUSCH", "Noise + interference estimate"),
    "noiseIntfEstNoDftSOfdmKernelInner":  ("PUSCH", "Noise/intf inner helper"),
    "cfoTaEstLowMimoKernel":              ("PUSCH", "CFO + timing-advance estimate"),
    "eqMmseCoefCompLowMimoKernel":        ("PUSCH", "MMSE equalizer coefficients"),
    "eqMmseIrcCoefCompLowMimoKernel":     ("PUSCH", "MMSE-IRC equalizer coefficients"),
    "eqMmseSoftDemapKernel":              ("PUSCH", "Equalize + soft LLR demap"),
    "de_rate_matching_reset_buffer":      ("PUSCH", "De-rate-match buffer reset"),
    "de_rate_matching_clamp_buffer":      ("PUSCH", "De-rate-match LLR clamp"),
    "de_rate_matching_global2":           ("PUSCH", "De-rate-match + HARQ combine"),
    "crcUplinkPuschCodeBlocksKernel":     ("PUSCH", "CB CRC check"),
    "crcUplinkPuschTransportBlockKernel": ("PUSCH", "TB CRC check -> ACK/NACK"),
    "rsrpMeasKernel_v1":                  ("PUSCH", "RSRP measurement"),
    "rssiMeasKernel":                     ("PUSCH", "RSSI measurement"),

    # ── UCI on PUSCH / control decoding ─────────────────────────────────────
    "compCwTreeTypesKernel":    ("UCI", "Codeword tree-type computation"),
    "polSegDeRmDeItlKernel":    ("UCI", "Polar seg. de-rate-match + de-interleave"),
    "listPolarDecoderKernel":   ("UCI", "Polar list decoder"),
    "rm_decoder_3":             ("UCI", "Reed-Muller decoder"),
    "simplex_decoder_kernel":   ("UCI", "Simplex decoder"),
    "uciOnPuschCsi2CtrlKernel": ("UCI", "UCI-on-PUSCH CSI part-2 control"),
    "uciOnPuschSegLLRs0Kernel": ("UCI", "UCI LLR segmentation (part 0)"),
    "uciOnPuschSegLLRs2Kernel": ("UCI", "UCI LLR segmentation (part 2)"),

    # ── PUCCH ───────────────────────────────────────────────────────────────
    "pucchF1RxKernel": ("PUCCH", "Format-1 receiver"),

    # ── PRACH ───────────────────────────────────────────────────────────────
    "prach_compute_correlation": ("PRACH", "Preamble correlation"),
    "block_fft_kernel":          ("PRACH", "Block FFT (cuFFTDx)"),
    "prach_compute_pdp":         ("PRACH", "Power-delay profile"),
    "prach_search_pdp":          ("PRACH", "PDP peak search"),
    "prach_compute_rssi":        ("PRACH", "RSSI computation"),
    "memsetRssi":                ("PRACH", "RSSI buffer clear"),
    "memcpyRssi":                ("PRACH", "RSSI buffer copy"),

    # ── CUDA-graph structure / utility ──────────────────────────────────────
    # kDeviceGraphLauncher is the trampoline that makes the PUSCH chain
    # invisible to nsys: it runs ON the GPU and calls cudaGraphLaunch, so the
    # inner graph never produces a host-side launch event.
    "kDeviceGraphLauncher": ("Graph", "Device-side graph launch trampoline"),
    "graphs_empty_kernel":  ("Graph", "Graph topology placeholder"),
    "memset_kernel":        ("Utility", "Slot-boundary buffer clear"),
    "warmup_kernel":        ("Utility", "GPU warmup"),
    "warmup":               ("Utility", "Fronthaul warmup"),
}

# NOT kernels. These two are __device__ functions called from inside a parent
# kernel; the AERIAL_KTRACE macro was placed in their bodies, so each emits a
# trace record that looks exactly like a launch.
#
# They are still worth showing — each marks a real phase boundary inside its
# parent's execution — but they must never be counted as launches, and they
# have no occupancy of their own.
DEVICE_FUNCTIONS = {
    "noiseIntfEstNoDftSOfdmKernelInner":  "noiseIntfEstNoDftSOfdmKernel",
    "eqMmseIrcCoefCompLowMimoKernel":     "eqMmseCoefCompLowMimoKernel",
}

# Kernels that the STOCK Aerial config hides from nsys, because PUSCH runs as
# a device-launched CUDA graph (pusch_workCancelMode: 2, pusch_deviceGraphLaunchEn: 1)
# and a fire-and-forget graph launch emits no host-side launch event for CUPTI
# to record. This is a property of the RUN CONFIG, not of the kernel, and not of
# how a given capture was taken: with both keys set to 0 every name below is
# launched from the host and nsys records it normally.
# There might be a different way to extract these kernels in the runtime. Can be consulted with NVIDIA.

DEVICE_GRAPH_HIDDEN = {
    "cfoTaEstLowMimoKernel", "chEstFilterNoDftSOfdmDispatchKernel",
    "compCwTreeTypesKernel", "crcUplinkPuschCodeBlocksKernel",
    "crcUplinkPuschTransportBlockKernel", "de_rate_matching_clamp_buffer",
    "de_rate_matching_global2", "de_rate_matching_reset_buffer",
    "eqMmseCoefCompLowMimoKernel", "eqMmseIrcCoefCompLowMimoKernel",
    "eqMmseSoftDemapKernel", "listPolarDecoderKernel",
    "noiseIntfEstNoDftSOfdmKernel", "noiseIntfEstNoDftSOfdmKernelInner",
    "polSegDeRmDeItlKernel", "rm_decoder_3", "rsrpMeasKernel_v1",
    "rssiMeasKernel", "simplex_decoder_kernel", "uciOnPuschCsi2CtrlKernel",
    "uciOnPuschSegLLRs0Kernel", "uciOnPuschSegLLRs2Kernel",
    "windowedChEstPreNoDftSOfdmKernel",
}

# Backwards-compatible alias: the old name is a misnomer kept only so existing
# imports resolve. Prefer DEVICE_GRAPH_HIDDEN.
KTRACE_ONLY = DEVICE_GRAPH_HIDDEN

_SUBSTRING_FALLBACK = [
    ("doca",     ("FH-DL",   "Fronthaul/DOCA (unclassified)")),
    ("prach",    ("PRACH",   "PRACH (unclassified)")),
    ("pucch",    ("PUCCH",   "PUCCH (unclassified)")),
    ("pusch",    ("PUSCH",   "PUSCH (unclassified)")),
    ("pdsch",    ("PDSCH",   "PDSCH (unclassified)")),
    ("pdcch",    ("PDCCH",   "PDCCH (unclassified)")),
    ("csirs",    ("CSI-RS",  "CSI-RS (unclassified)")),
    # LDPC decode is UPLINK (PUSCH), LDPC encode is DOWNLINK (PDSCH). They
    # share the "ldpc" token, so the specific rules must precede the bare one.
    ("ldpc2_",      ("PUSCH",   "LDPC decode (cubin)")),
    ("ldpc_decode", ("PUSCH",   "LDPC decode")),
    ("ldpc_encode", ("PDSCH",   "LDPC encode")),
    ("ldpc",        ("PDSCH",   "LDPC (unclassified)")),
    ("polar",    ("UCI",     "Polar (unclassified)")),
    ("uci",      ("UCI",     "UCI (unclassified)")),
    ("chest",    ("PUSCH",   "Channel estimation (unclassified)")),
    ("eqmmse",   ("PUSCH",   "Equalizer (unclassified)")),
    ("crc",      ("PDSCH",   "CRC (unclassified)")),
    ("fft",      ("PRACH",   "FFT (unclassified)")),
    ("graph",    ("Graph",   "Graph structure")),
    ("warmup",   ("Utility", "Warmup")),
    ("memset",   ("Utility", "Buffer clear")),
]


# The PUSCH LDPC decoder is not compiled with cuPHY: it ships as prebuilt SASS
# in /opt/nvidia/ldpc_decoder_cubin and is loaded with cuModuleLoadData, so its
# name reaches a capture straight from the cubin. Format: ldpc2_BG<n>_<algo>_tb.
_LDPC2_RE = re.compile(r"^ldpc2_BG(\d+)_(.+?)_tb$", re.IGNORECASE)


def classify_kernel(name: str) -> tuple[str, str]:
    """(channel, role). Exact match first, then substring, then Unknown.

    Unknown is returned rather than guessed: an unmapped kernel showing up as
    'Unknown' in the UI is a prompt to extend this table, whereas a wrong
    channel silently corrupts the per-channel breakdown.
    """
    if name in KERNEL_TAXONOMY:
        return KERNEL_TAXONOMY[name]
    m = _LDPC2_RE.match(name or "")
    if m:
        algo = m.group(2)
        # Trim the trailing SM tag: it names the cubin the algorithm was built
        # for, not the GPU it ran on (the "_sm86" variants are sm_90 binaries).
        algo = re.sub(r"_sm\d+$", "", algo)
        return ("PUSCH", f"LDPC decode BG{m.group(1)} ({algo})")
    ln = (name or "").lower()
    for token, val in _SUBSTRING_FALLBACK:
        if token in ln:
            return val
    return ("Unknown", "unmapped kernel")


def channel_color(channel: str) -> str:
    return CHANNEL_COLOR.get(channel, CHANNEL_COLOR["Unknown"])


if __name__ == "__main__":
    from collections import Counter
    c = Counter(v[0] for v in KERNEL_TAXONOMY.values())
    print(f"{len(KERNEL_TAXONOMY)} traced entities "
          f"({len(KERNEL_TAXONOMY)-len(DEVICE_FUNCTIONS)} kernels + "
          f"{len(DEVICE_FUNCTIONS)} device functions) across {len(c)} channels")
    for ch, n in c.most_common():
        print(f"  {ch:<9} {n:>3}   {channel_color(ch)}")
    print(f"\nKTRACE-only (invisible to nsys): {len(KTRACE_ONLY)}")
