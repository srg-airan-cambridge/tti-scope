#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
CPU-side profile from `perf record` captures.

WHAT THIS CAN AND CANNOT ANSWER
-------------------------------
The sweeps recorded `perf record -F 99 -a -g -- sleep 45`, i.e. system-wide
cycle sampling at 99 Hz across all 72 cores, one capture per cell count. From
that we can say, accurately:

  * how busy each core was, over the traffic window;
  * which thread ran on which core, and for how much of it;
  * how much of the whole machine the DU consumed.

We cannot say WHERE INSIDE A 500 us TTI the CPU was busy, and it is worth being
explicit about why, because it is two independent blockers:
  1. perf stamps samples with CLOCK_MONOTONIC (seconds since boot) and this
     capture was taken without `-k`, so no realtime reference is stored. The
     only anchor is the header's "captured on" line, at 1-second granularity —
     2,000 TTI periods. Folding to a 500 us phase needs the offset to
     microseconds; it is known to a second. The phase is simply not there.
  2. At 99 Hz there is one sample per core per ~10 ms, i.e. 0.05 samples per
     core per TTI. Even with a perfect clock, intra-slot structure could only
     come from folding many slots together — which needs (1).

So this module reports per-core and per-thread utilisation, and the UI labels
it as such. Getting intra-TTI CPU phase would need a re-run with
`perf record -k CLOCK_REALTIME` (or nsys `--cpuctxsw`), not better analysis.

THE TRAFFIC WINDOW
------------------
`sleep 45` bounds perf, not the DU. On the 20-cell capture the PHY driver
threads stop at ~32 s while perf keeps sampling to 74.8 s; computing over the
full span understates every utilisation figure by ~2.5x. The window in which
the PHY threads are actually scheduled is therefore detected and used.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SAMPLE_HZ = 99.0          # default only; the real rate is read per capture
CACHE_VERSION = 4         # bump when the parse or the idle rule changes

# Idle is not only "swapper". On this kernel/perf pair a large share of idle
# samples arrive with an UNRESOLVED comm, printed as ":0" with pid/tid 0/0 —
# 23,245 of them in the 1-cell capture, which is more samples than every DU
# thread combined. Counting those as busy is what made "machine %" jump around
# between 3% and 12% with no relation to cell count. pid 0 IS the idle task, so
# tid == 0 is the reliable test and the comm string is only a fallback.
IDLE_COMM = {"swapper", "swapper/0", ":0", ":-1"}

PERF_DIR_RE = re.compile(r"^perf_percore_(\d+)C_(\w+?)_(\d{8})_(\d{6})$")

# nvlog: "... [L2A.TICK_TIMES] SFN 262.10 current_time=<ns>, tick=<ns>"
# `tick` is the ideal 500 us slot boundary; `current_time` is when the timer
# thread actually woke, several us late. The grid is the former.
TICK_RE = re.compile(
    r"\[L2A\.TICK_TIMES\]\s+SFN\s+(\d+)\.(\d+)\s+current_time=(\d+),\s*tick=(\d+)")
# clock_anchor.txt: "pre mono=<ns> real=<ns> tai=<ns> read_ns=<n>"
ANCHOR_RE = re.compile(
    r"^(\S+)\s+mono=(\d+)\s+real=(\d+)\s+tai=(\d+)\s+read_ns=(\d+)")

# Thread name -> role. cuphycontroller pins each worker to its own core, so
# these also name the cores.
ROLE_RULES = [
    (re.compile(r"^DlPhyDriver"),      "DL PHY worker"),
    (re.compile(r"^UlPhyDriver"),      "UL PHY worker"),
    (re.compile(r"^phy_main$"),        "PHY main"),
    (re.compile(r"^h2dcpy_thread$"),   "H2D copy"),
    (re.compile(r"^timer_thread$"),    "L2A slot timer"),
    (re.compile(r"^msg_processing"),   "L2A FAPI"),
    (re.compile(r"^DebugWorker"),      "Debug worker"),
    (re.compile(r"^bg_fmtlog$"),       "nvlog writer"),
    (re.compile(r"^(cuda-|nvidia-cuda|cuda_)"), "CUDA / MPS"),
    (re.compile(r"^(nvipc|fh_|ru_)"),  "Fronthaul / nvipc"),
    (re.compile(r"^(test_?mac|testmac)"), "testMAC"),
]

ROLE_COLOR = {
    "DL PHY worker":     "#378ADD",
    "UL PHY worker":     "#D4537E",
    "PHY main":          "#7F77DD",
    "H2D copy":          "#9FE1CB",
    "L2A slot timer":    "#BA7517",
    "L2A FAPI":          "#E07A5F",
    "Debug worker":      "#59C3A5",
    "nvlog writer":      "#888780",
    "CUDA / MPS":        "#1D9E75",
    "Fronthaul / nvipc": "#5D55BB",
    "testMAC":           "#B4B2A9",
    "other":             "#6E7681",
}

# Roles that constitute the DU itself, as opposed to whatever else the box was
# doing (udisks, tmux, ssh, the perf process...).
DU_ROLES = {"DL PHY worker", "UL PHY worker", "PHY main", "H2D copy",
            "L2A slot timer", "L2A FAPI", "Debug worker", "nvlog writer",
            "CUDA / MPS", "Fronthaul / nvipc"}

PHY_ROLES = {"DL PHY worker", "UL PHY worker", "PHY main", "H2D copy"}


def classify_thread(comm: str) -> str:
    for rx, role in ROLE_RULES:
        if rx.match(comm or ""):
            return role
    return "other"


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CpuCapture:
    """A perf_percore_<N>C_<pat>_<stamp> directory."""
    path: Path
    perf_data: Path
    cells: Optional[int] = None
    pattern: str = ""
    when: str = ""
    nvlog: Optional[Path] = None
    anchor: Optional[Path] = None   # clock_anchor.txt, if the capture has one

    @property
    def tti_capable(self) -> bool:
        """Whether this capture can be folded into TTI slots at all.

        Needs both halves: the monotonic->realtime anchor written by the sweep,
        and the run's own nvlog to supply the slot grid. Captures from the
        earlier sweeps have the nvlog but no anchor, so for those only per-core
        utilisation is available - the phase genuinely is not in the data.
        """
        return bool(self.anchor and self.nvlog)

    @property
    def key(self):
        return (self.cells, self.pattern)


def discover_cpu_captures(root, max_depth: int = 3) -> dict:
    """{(cells, pattern): CpuCapture} for every perf capture under `root`."""
    root = Path(root).expanduser()
    out: dict = {}
    if not root.exists():
        return out

    def walk(d: Path, depth: int):
        if depth > max_depth:
            return
        try:
            entries = list(d.iterdir())
        except OSError:
            return
        m = PERF_DIR_RE.match(d.name)
        if m:
            pd = [p for p in entries if p.name.endswith(".perf.data")]
            if pd:
                logs = d / "logs"
                nv = None
                if logs.is_dir():
                    nv = next((p for p in logs.iterdir()
                               if p.name.startswith("cuphy_")
                               and p.suffix == ".log"), None)
                anc = d / "clock_anchor.txt"
                c = CpuCapture(path=d, perf_data=pd[0], cells=int(m.group(1)),
                               pattern=m.group(2),
                               when=f"{m.group(3)}-{m.group(4)}", nvlog=nv,
                               anchor=anc if anc.exists() else None)
                # On a key collision keep the NEWEST capture. A results tree
                # accumulates re-runs of the same cell count, and taking
                # whichever the directory walk reached first silently analysed
                # a superseded sweep - the symptom was an updated capture set
                # producing byte-identical numbers to the one it replaced.
                prev = out.get(c.key)
                if prev is None or c.when > prev.when:
                    out[c.key] = c
                return
        for p in entries:
            if p.is_dir():
                walk(p, depth + 1)

    walk(root, 0)
    return out


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CpuProfile:
    """Per-core and per-thread utilisation over the traffic window."""
    label: str = ""
    cells: Optional[int] = None
    n_cores: int = 0
    span_s: float = 0.0            # whole perf capture
    window_s: float = 0.0          # traffic window actually used
    window: tuple = (0.0, 0.0)
    cores: list = field(default_factory=list)   # per-core dicts
    threads: list = field(default_factory=list)  # per-thread dicts
    sample_hz: float = SAMPLE_HZ
    note: str = ""

    # -- headline numbers ---------------------------------------------------
    @property
    def machine_pct(self) -> float:
        """Busy fraction of the WHOLE box: all non-idle samples over all cores."""
        if not self.n_cores or not self.window_s:
            return 0.0
        cap = self.sample_hz * self.window_s * self.n_cores
        return 100.0 * sum(c["busy"] for c in self.cores) / cap if cap else 0.0

    @property
    def du_pct(self) -> float:
        cap = self.sample_hz * self.window_s * self.n_cores
        if not cap:
            return 0.0
        du = sum(t["samples"] for t in self.threads if t["role"] in DU_ROLES)
        return 100.0 * du / cap

    @property
    def cores_busy(self) -> float:
        """Aggregate expressed as 'this many cores fully occupied'."""
        if not self.window_s:
            return 0.0
        return sum(c["busy"] for c in self.cores) / (self.sample_hz * self.window_s)

    def phy_cores(self):
        return [c for c in self.cores if c["role"] in PHY_ROLES]


def perf_sample_hz(perf_data: Path) -> float:
    """Sampling rate recorded in the perf.data header.

    Not assumed: the 2026-08-03 sweep used -F 99 and the 2026-08-11 one -F 997,
    and taking the wrong one scales every "cores busy" figure by 10x.
    """
    try:
        hdr = subprocess.run(
            [shutil.which("perf") or "perf", "report", "-i", str(perf_data),
             "--header-only"], capture_output=True, text=True, timeout=180).stdout
    except (OSError, subprocess.TimeoutExpired):
        return SAMPLE_HZ
    m = re.search(r"sample_freq\s*\}\s*=\s*(\d+)", hdr)
    if m:
        return float(m.group(1))
    m = re.search(r"sample_period,\s*sample_freq\s*\}\s*=\s*(\d+)", hdr)
    return float(m.group(1)) if m else SAMPLE_HZ


def _run_perf_script(perf_data: Path) -> Optional[str]:
    exe = shutil.which("perf")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "script", "-i", str(perf_data),
             "-F", "comm,pid,tid,cpu,time", "--no-demangle"],
            capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 and r.stdout else None


# "  comm  pid/tid  [cpu]  time:"   — comm may contain spaces ("tmux: server")
LINE_RE = re.compile(
    r"^\s*(.+?)\s+(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+\.\d+):")


def _cache_path(perf_data: Path) -> Path:
    d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = d / "ttiscope" / "cpu"
    d.mkdir(parents=True, exist_ok=True)
    st = perf_data.stat()
    return d / (f"{perf_data.parent.name}_v{CACHE_VERSION}"
                f"_{int(st.st_mtime)}_{st.st_size}.json")


def build_profile(cap: CpuCapture, log=print) -> Optional[CpuProfile]:
    """Parse (or load from cache) one perf capture."""
    cache = _cache_path(cap.perf_data)
    if cache.exists():
        try:
            return _from_dict(json.loads(cache.read_text()))
        except (OSError, ValueError, KeyError):
            pass

    log(f"  decoding {cap.perf_data.name} with perf script…")
    text = _run_perf_script(cap.perf_data)
    if text is None:
        return CpuProfile(
            label=cap.path.name, cells=cap.cells,
            note="`perf` is not installed, or it could not read this "
                 "perf.data (it was recorded by perf 6.8 on aarch64). "
                 "Install linux-tools matching the recording kernel.")

    per_bucket = Counter()          # 1 s bucket -> PHY-thread samples
    rows = []
    t0 = None
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        comm, tid = m.group(1).strip(), int(m.group(3))
        cpu, t = int(m.group(4)), float(m.group(5))
        if tid == 0:
            comm = "swapper"          # unresolved idle task
        if t0 is None:
            t0 = t
        rows.append((t - t0, cpu, comm))
        if classify_thread(comm) in PHY_ROLES:
            per_bucket[int(t - t0)] += 1
    if not rows:
        return None
    span = rows[-1][0]

    # ── traffic window ───────────────────────────────────────────────────
    # `sleep 45` bounds perf, not the DU: on the 20-cell capture the PHY
    # threads stop at ~32 s while sampling runs to 74.8 s. Use the span in
    # which PHY threads are actually scheduled, or every figure is diluted by
    # the idle tail.
    active = [b for b, n in per_bucket.items() if n >= 5]
    if active:
        w0, w1 = float(min(active)), float(max(active) + 1)
    else:
        w0, w1 = 0.0, span
    window_s = max(1e-9, w1 - w0)

    per_core = defaultdict(Counter)
    for t, cpu, comm in rows:
        if w0 <= t < w1:
            per_core[cpu][comm] += 1

    n_cores = 72
    try:
        hdr = subprocess.run(["perf", "report", "-i", str(cap.perf_data),
                              "--header-only"], capture_output=True, text=True,
                             timeout=120).stdout
        mm = re.search(r"nrcpus online\s*:\s*(\d+)", hdr)
        if mm:
            n_cores = int(mm.group(1))
    except (OSError, subprocess.TimeoutExpired):
        pass

    rate = perf_sample_hz(cap.perf_data)
    cap_per_core = rate * window_s
    cores = []
    for cpu in range(n_cores):
        cn = per_core.get(cpu, Counter())
        idle = sum(v for k, v in cn.items() if k in IDLE_COMM)
        busy = sum(cn.values()) - idle
        top = [(k, v) for k, v in cn.most_common() if k not in IDLE_COMM]
        named = [(k, v) for k, v in top if classify_thread(k) != "other"]
        role = classify_thread(named[0][0]) if named else (
            classify_thread(top[0][0]) if top else "other")
        if named:
            # Lead the label with the recognised DU thread, so the tick reads
            # "cpu9 DlPhyDriver09" and not "cpu9 <unclassified>".
            top = named + [x for x in top if x not in named]
        cores.append(dict(
            cpu=cpu, busy=busy, idle=idle,
            pct=100.0 * busy / cap_per_core if cap_per_core else 0.0,
            role=role,
            top=[{"comm": k, "samples": v,
                  "pct": 100.0 * v / cap_per_core if cap_per_core else 0.0}
                 for k, v in top[:4]]))

    tot = Counter()
    tcore = defaultdict(Counter)
    for cpu, cn in per_core.items():
        for comm, v in cn.items():
            if comm in IDLE_COMM:
                continue
            tot[comm] += v
            tcore[comm][cpu] += v
    threads = [dict(comm=c, samples=v, role=classify_thread(c),
                    pct_of_core=100.0 * v / cap_per_core if cap_per_core else 0.0,
                    cores=[cpu for cpu, _ in tcore[c].most_common(3)])
               for c, v in tot.most_common()]

    prof = CpuProfile(label=cap.path.name, cells=cap.cells, n_cores=n_cores,
                      span_s=span, window_s=window_s, window=(w0, w1),
                      cores=cores, threads=threads, sample_hz=rate)
    try:
        cache.write_text(json.dumps(_to_dict(prof)))
    except OSError:
        pass
    return prof


# ─────────────────────────────────────────────────────────────────────────────
# Intra-TTI fold
# ─────────────────────────────────────────────────────────────────────────────

def read_clock_offset(path: Path) -> tuple:
    """(offset_ns, drift_ns, n_readings) mapping CLOCK_MONOTONIC -> realtime.

    This kernel refuses CLOCK_REALTIME/CLOCK_TAI/CLOCK_BOOTTIME for perf events
    ("wrong clockid (0)"), so the sweep records CLOCK_MONOTONIC and writes
    paired readings of both clocks before, during and after the capture. Each
    pairing reads monotonic either side of realtime, so its own uncertainty is
    known; measured spread is 112-176 ns and pre-to-final drift was 0 ns over a
    45 s capture. The median is used, and the pre-to-final difference is
    reported as the drift so the caller can judge it rather than assume it.
    """
    pre, post, fin = [], [], []
    try:
        for line in open(path):
            m = ANCHOR_RE.match(line.strip())
            if not m:
                continue
            tag, mono, real = m.group(1), int(m.group(2)), int(m.group(3))
            {"pre": pre, "post": post, "final": fin}.get(tag, post).append(
                real - mono)
    except OSError:
        return None, None, 0
    allv = pre + post + fin
    if not allv:
        return None, None, 0
    allv.sort()
    off = allv[len(allv) // 2]
    drift = 0
    if pre and fin:
        pre.sort(); fin.sort()
        drift = fin[len(fin) // 2] - pre[len(pre) // 2]
    return off, drift, len(allv)


def read_tick_grid(path: Path) -> list:
    """Sorted slot-boundary timestamps (realtime ns) from an nvlog.

    Uses `tick`, the ideal 500 us boundary, not `current_time`, which is when
    the L2 timer thread actually woke - several microseconds late and jittery.
    """
    ticks = set()
    try:
        with open(path, errors="replace", buffering=1 << 20) as fh:
            for line in fh:
                if "TICK_TIMES" in line:
                    m = TICK_RE.search(line)
                    if m:
                        ticks.add(int(m.group(4)))
    except OSError:
        return []
    return sorted(ticks)


def intra_tti_profile(cap: CpuCapture, n_bins: int = 25,
                      max_slots: int = 1000, log=print) -> Optional[dict]:
    """Where inside the 500 us TTI the CPU is busy, per thread role.

    Folds every perf sample that falls inside the slot grid onto slot phase.
    Returns per-bin, per-role "cores busy": samples / (bin_duration * rate),
    which is directly comparable across bins and captures because it does not
    depend on how many slots were folded.
    """
    if not cap.tti_capable:
        return None
    off, drift, n_anchor = read_clock_offset(cap.anchor)
    if off is None:
        return None
    ticks = read_tick_grid(cap.nvlog)
    if len(ticks) < 4:
        return None

    text = _run_perf_script(cap.perf_data)
    if text is None:
        return None
    samples = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        if int(m.group(3)) == 0:          # idle task
            continue
        # perf prints seconds with 6 decimals; go via integer us to avoid
        # float rounding drifting the phase across a 45 s capture.
        sec, _, frac = m.group(5).partition(".")
        t = (int(sec) * 1_000_000_000 + int(frac.ljust(9, "0")[:9])) + off
        samples.append((t, m.group(1).strip()))
    if not samples:
        return None
    samples.sort()

    # The overlap is what limits this: the perf window is ~45 s but the nvlog's
    # tick coverage ends when the run does, and on a run that crashed early it
    # may be only a couple of seconds.
    lo = max(ticks[0], samples[0][0])
    hi = min(ticks[-1], samples[-1][0])
    if hi <= lo:
        return None
    period = ticks[1] - ticks[0] if len(ticks) > 1 else 500_000
    overlap_s = (hi - lo) / 1e9
    avail = int((hi - lo) // period)
    # Fold the LAST `max_slots` of the overlap: the head of a run is ramp-up.
    if max_slots and avail > max_slots:
        lo = hi - max_slots * period
    n_slots = max(1, int((hi - lo) // period))
    rate = perf_sample_hz(cap.perf_data)

    roles = {}
    used = 0
    for t, comm in samples:
        if not (lo <= t < hi):
            continue
        i = bisect.bisect_right(ticks, t) - 1
        if i < 0:
            continue
        ph = t - ticks[i]
        if not (0 <= ph < period):
            continue          # a gap in the tick log; not a real slot
        b = min(n_bins - 1, int(ph * n_bins // period))
        roles.setdefault(classify_thread(comm), [0] * n_bins)[b] += 1
        used += 1
    if not used:
        return None

    bin_s = (period / n_bins) * 1e-9
    # samples -> cores busy: a core sampled at `rate` for the whole of every
    # folded bin would contribute rate * bin_s * n_slots samples.
    scale = 1.0 / (rate * bin_s * n_slots)
    out_roles = {r: [c * scale for c in v] for r, v in roles.items()}
    total = [sum(v[b] for v in out_roles.values()) for b in range(n_bins)]
    return dict(
        label=cap.path.name, cells=cap.cells, n_bins=n_bins,
        period_us=period / 1000.0, n_slots=n_slots, n_samples=used,
        clock_offset_ns=off, clock_drift_ns=drift, n_anchor=n_anchor,
        sample_hz=rate, overlap_s=overlap_s, slots_available=avail,
        edges_us=[period / 1000.0 * i / n_bins for i in range(n_bins + 1)],
        roles=out_roles, total=total)


def _to_dict(p: CpuProfile) -> dict:
    return dict(label=p.label, cells=p.cells, n_cores=p.n_cores,
                span_s=p.span_s, window_s=p.window_s, window=list(p.window),
                cores=p.cores, threads=p.threads, sample_hz=p.sample_hz,
                note=p.note)


def _from_dict(d: dict) -> CpuProfile:
    return CpuProfile(label=d["label"], cells=d.get("cells"),
                      n_cores=d["n_cores"], span_s=d["span_s"],
                      window_s=d["window_s"], window=tuple(d["window"]),
                      cores=d["cores"], threads=d["threads"],
                      sample_hz=d.get("sample_hz", SAMPLE_HZ),
                      note=d.get("note", ""))


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Summarise perf_percore_* captures")
    ap.add_argument("root")
    ap.add_argument("-c", "--cells", type=int, default=None)
    a = ap.parse_args()
    caps = discover_cpu_captures(a.root)
    print(f"{len(caps)} CPU capture(s)")
    for k in sorted(caps, key=lambda x: (x[0] or 0)):
        print(f"  {caps[k].cells:>3}C {caps[k].pattern:<6} {caps[k].path.name}")
    if a.cells is None:
        return
    cap = next((c for c in caps.values() if c.cells == a.cells), None)
    if not cap:
        print("no capture for that cell count")
        return
    p = build_profile(cap)
    if p is None or p.note:
        print(p.note if p else "failed")
        return
    print(f"\n{p.label}")
    print(f"  perf span {p.span_s:.1f}s, traffic window "
          f"{p.window[0]:.0f}-{p.window[1]:.0f}s ({p.window_s:.0f}s used)")
    print(f"  machine {p.machine_pct:.1f}% of {p.n_cores} cores "
          f"({p.cores_busy:.1f} cores busy);  DU threads {p.du_pct:.1f}%")
    print(f"\n  {'core':>5}{'util %':>9}  role / top thread")
    for c in sorted(p.cores, key=lambda c: -c["pct"])[:16]:
        t = c["top"][0]["comm"] if c["top"] else "-"
        print(f"  {c['cpu']:>5}{c['pct']:>9.1f}  {c['role']:<18} {t}")
    print(f"\n  {'thread':<18}{'% of a core':>12}  cores")
    for t in p.threads[:14]:
        print(f"  {t['comm']:<18}{t['pct_of_core']:>12.1f}  {t['cores']}")


if __name__ == "__main__":
    main()
