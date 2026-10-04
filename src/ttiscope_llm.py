# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""LLM inference traces: read an nsys SQLite export and find the phases.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 
# Kernel taxonomy
# 
#Only tested against Qwen and LLama. We need to revisit for mutiple llms and Physical AI later. 
# Ordered: the lane order in the view is the order of this list, which is
# roughly the order a token flows through a transformer block. Patterns are
# matched against the demangled short name, lowercased, first hit wins.

FAMILIES = [
    # Before Embedding: "rotary_embedding" contains "embedding", and the
    # first matching family wins.
    ("RoPE",        [r"rotary", r"rope_", r"_rope"]),
    ("Embedding",   [r"embedding", r"embed_tokens"]),
    # ONLY genuinely phase-specific names belong in these two. FlashAttention
    # (flash_fwd / FlashAttnFwdSm90 / varlen / fmha) is NOT one of them: vLLM
    # V1 on Hopper runs the same FA3 kernel for prefill and decode, so putting
    # it here made every step match "prefill" and none match "decode" - a trace
    # that is 84% decode reported as 100% prefill. Those go to Attention.
    ("Attn prefill", [r"prefill", r"chunked_prefill"]),
    ("Attn decode", [r"paged_attention", r"single_query", r"decode_attn",
                     r"flash_decod", r"attention_decode", r"packed_decode",
                     r"_decode_kernel", r"decode_kernel"]),
    ("Attention",   [r"flash_fwd", r"flashattn", r"fmha", r"mha_fwd",
                     r"varlen", r"attn", r"attention", r"softmax_lse"]),
    ("GEMM",        [r"gemm", r"cutlass", r"nvjet", r"sm\d+_xmma", r"ampere_",
                     r"turing_", r"volta_", r"gemv", r"matmul", r"^s\d*gemm",
                     r"cublas", r"marlin", r"awq", r"gptq", r"machete",
                     r"scaled_mm", r"w4a16", r"w8a8"]),
    ("MoE",         [r"moe", r"expert", r"topk_softmax", r"grouped_gemm"]),
    ("Norm",        [r"rms_norm", r"layer_norm", r"layernorm", r"rmsnorm"]),
    ("Activation",  [r"silu", r"gelu", r"swiglu", r"act_and_mul", r"relu"]),
    ("KV cache",    [r"reshape_and_cache", r"copy_blocks", r"swap_blocks",
                     r"cache_kernel", r"kv_cache"]),
    ("Sampling",    [r"sampl", r"topk", r"top_k", r"top_p", r"multinomial",
                     r"argmax", r"logits", r"penalt"]),
    ("Comm",        [r"nccl", r"all_reduce", r"allreduce", r"all_gather",
                     r"reduce_scatter", r"custom_ar"]),
    ("Quant",       [r"quant", r"dequant", r"per_token", r"scaled_fp8",
                     r"int8", r"fp8"]),
    ("Elementwise", [r"elementwise", r"vectorized", r"copy", r"cat_", r"fill",
                     r"memset", r"transpose", r"reshape", r"index_", r"gather",
                     r"scatter", r"add_", r"mul_", r"where"]),
]
OTHER = "Other"
FAMILY_ORDER = [n for n, _ in FAMILIES] + [OTHER]

_COMPILED = [(name, re.compile("|".join(pats))) for name, pats in FAMILIES]

# Phase-bearing families: seeing one of these in a step is direct evidence of
# what the step was, independent of how long it took.
PREFILL_MARK = "Attn prefill"
DECODE_MARK = "Attn decode"

FAMILY_COLOR = {
    "Embedding":    "#8B7BD8",
    "Attn prefill": "#E8544F",
    "Attn decode":  "#F2994A",
    # Phase-agnostic attention, kept visually distinct from the two above so a
    # figure cannot imply a phase the kernel does not carry.
    "Attention":    "#C97BB0",
    "GEMM":         "#4FA3E8",
    "MoE":          "#B36AE2",
    "Norm":         "#3FA796",
    "Activation":   "#5BC8A0",
    "RoPE":         "#C9A227",
    "KV cache":     "#7B9E4F",
    "Sampling":     "#E85D9E",
    "Comm":         "#D4553A",
    "Quant":        "#6E8CA8",
    "Elementwise":  "#8b939c",
    OTHER:          "#5c6470",
}


def classify_llm_kernel(name: str) -> str:
    """Family for one kernel name. First pattern wins, so order matters:
    the prefill/decode attention entries sit above the generic Attention one
    precisely so a paged-attention kernel is not swallowed by it."""
    n = (name or "").lower()
    for fam, rx in _COMPILED:
        if rx.search(n):
            return fam
    return OTHER


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    """One forward pass."""
    idx: int
    t0: int
    t1: int
    n_kernels: int
    busy_ns: int
    phase: str = "decode"          # 'prefill' | 'decode'
    marked: bool = False           # typed by an attention kernel, not duration
    families: set = field(default_factory=set)

    @property
    def dur_ns(self) -> int:
        return max(0, self.t1 - self.t0)


class LLMTrace:
    """An nsys SQLite export of an inference run.

    `provenance` records how the steps were found, in words, because the
    figures made from this are claims about prefill and decode and the reader
    is entitled to know whether that came from a marker or from a threshold.
    """

    # A step boundary needs the GPU to actually go quiet; below this the gap is
    # just the launch latency between two kernels of one pass.
    GAP_NS = 200_000                    # 200 us

    def __init__(self, path: Path, prefill_factor: float = 3.0):
        self.path = Path(path)
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.prefill_factor = prefill_factor
        self.provenance = ""
        self.warnings: list[str] = []
        self.name_family: dict[int, str] = {}
        self.family_names: dict[str, str] = {}
        self.steps: list[Step] = []
        self.phase_basis = ""
        self.t0 = self.t1 = 0
        self.n_kernels = 0
        self._check_schema()
        self._classify_names()
        self._build_steps()

    # -- setup ---------------------------------------------------------------
    def _check_schema(self):
        have = {r[0] for r in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        need = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
        missing = need - have
        if missing:
            raise ValueError(
                "This does not look like an nsys CUDA trace: missing "
                + ", ".join(sorted(missing))
                + ".\nExport with:  nsys export --type sqlite <run>.nsys-rep")

    def _name_col(self) -> str:
        cols = {c[1] for c in self.db.execute(
            "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        # demangledName is the readable one; shortName is the fallback on
        # exports that dropped it.
        return "demangledName" if "demangledName" in cols else "shortName"

    def _classify_names(self):
        """Classify the distinct kernel NAMES, not the launches.

        A trace with two million launches still has only a few hundred
        distinct kernels, so the regex work is done once per name and every
        launch is then a dict lookup.
        """
        col = self._name_col()
        rows = self.db.execute(
            f"SELECT DISTINCT k.{col}, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            f"JOIN StringIds s ON s.id = k.{col}").fetchall()
        for sid, value in rows:
            fam = classify_llm_kernel(value)
            self.name_family[sid] = fam
            self.family_names.setdefault(fam, value)
        if not rows:
            raise ValueError("The trace contains no CUDA kernels.")

    # -- step segmentation ---------------------------------------------------
    def _build_steps(self):
        """One streaming pass: segment the kernel stream into forward passes.

        Two signals, in order of preference:

          1. the sampling kernel. vLLM runs it exactly once per forward pass,
             so its launches ARE the step boundaries - an exact segmentation
             that needs no threshold.
          2. idle gaps. Without a sampling kernel (a trace that captured only
             part of the pipeline, or a backend that fuses sampling away), a
             quiet GPU is the only remaining evidence that one pass ended.

        Whichever was used is recorded in `provenance`.
        """
        col = self._name_col()
        cur = self.db.execute(
            f"SELECT start, end, {col} FROM CUPTI_ACTIVITY_KIND_KERNEL "
            f"ORDER BY start")

        sampling_ids = {sid for sid, f in self.name_family.items()
                        if f == "Sampling"}
        use_sampling = bool(sampling_ids)

        steps: list[Step] = []
        cur_start = cur_end = None
        n = busy = 0
        fams: set = set()
        total = 0
        last_end = None

        def close(t_end):
            if cur_start is None:
                return
            steps.append(Step(len(steps), cur_start, t_end, n, busy,
                              families=set(fams)))

        for st, en, sid in cur:
            total += 1
            fam = self.name_family.get(sid, OTHER)
            if cur_start is None:
                cur_start, cur_end = st, en
                n = busy = 0
                fams = set()
            else:
                boundary = (last_end is not None
                            and st - last_end > self.GAP_NS) if not use_sampling \
                    else (sid in sampling_ids)
                if boundary and n > 0:
                    if use_sampling:
                        # The sampling kernel ENDS the pass it belongs to.
                        n += 1
                        busy += max(0, en - st)
                        fams.add(fam)
                        cur_end = max(cur_end, en)
                        close(cur_end)
                        cur_start = None
                        last_end = en
                        continue
                    close(cur_end)
                    cur_start, cur_end = st, en
                    n = busy = 0
                    fams = set()
            n += 1
            busy += max(0, en - st)
            fams.add(fam)
            cur_end = max(cur_end or en, en)
            last_end = en
        close(cur_end)

        self.n_kernels = total
        self.steps = steps
        if steps:
            self.t0, self.t1 = steps[0].t0, steps[-1].t1
        self.provenance = (
            f"steps from the sampling kernel "
            f"({self.family_names.get('Sampling', 'sampling')}), one per "
            f"forward pass"
            if use_sampling else
            f"no sampling kernel in this trace; steps split at GPU idle gaps "
            f"longer than {self.GAP_NS/1000:.0f} us")
        if not use_sampling:
            self.warnings.append(
                "Step boundaries are inferred from idle gaps, not from a "
                "per-step kernel. Under continuous batching the GPU may never "
                "go idle between passes, in which case steps here are not "
                "individual forward passes.")
        self._label_phases()

    def _label_phases(self):
        """Prefill or decode, per step - or neither, said plainly.

        The obvious signal is the attention kernel, and on stacks that use a
        dedicated decode kernel (paged_attention) it is exact. It is also a
        trap: vLLM V1 on Hopper runs FlashAttention-3 for BOTH phases, so every
        step matches the prefill pattern and nothing matches decode. Trusting
        the marker there reports 100% prefill on a trace that is mostly decode.
        So the marker is used ONLY when both kinds appear - when it can
        actually discriminate.

        The fallback is not step duration. Measured on a real 60 s Qwen trace,
        duration correlates with per-step attention work at r = 0.02: at a low
        arrival rate most of a long step is the engine waiting, not computing.
        What IS bimodal there is the kernel COUNT per step (~979 against
        ~1855), which is the batch composition showing through. That is used
        when the split is clean, and when it is not, the phase is left
        undetermined rather than guessed.
        """
        if not self.steps:
            return
        fams_all = set(self.name_family.values())
        can_mark = PREFILL_MARK in fams_all and DECODE_MARK in fams_all
        self.phase_basis = ""

        if can_mark:
            unmarked = []
            for s in self.steps:
                if PREFILL_MARK in s.families:
                    s.phase, s.marked = "prefill", True
                elif DECODE_MARK in s.families:
                    s.phase, s.marked = "decode", True
                else:
                    s.marked = False
                    unmarked.append(s)
            for s in unmarked:                      # rare: no attention at all
                s.phase = "decode"
            self.phase_basis = "attention kernel"
            self.provenance += (
                f"; phases from the attention kernel "
                f"({len(self.steps)-len(unmarked):,} of {len(self.steps):,} "
                f"steps typed directly)")
            return

        # No discriminating marker. Try the batch-composition split.
        cut, sep = self._count_split()
        if cut is not None:
            for s in self.steps:
                s.phase = "prefill" if s.n_kernels >= cut else "decode"
                s.marked = False
            self.phase_basis = "kernel count"
            self.provenance += (
                f"; this stack runs one attention kernel for both phases, so "
                f"phases come from the per-step kernel count, which splits "
                f"cleanly at {cut:,} ({sep:.1f}x between the two groups)")
            self.warnings.append(
                "Prefill/decode here is the batch COMPOSITION, not a marker "
                "the engine emitted: steps with the larger kernel count are "
                "the ones carrying prompt tokens. With chunked prefill enabled "
                "such a step also carries decode tokens, so the two phases are "
                "not cleanly separable in time - state that in any figure.")
            return

        for s in self.steps:
            s.phase, s.marked = "undetermined", False
        self.phase_basis = "none"
        self.provenance += ("; phases UNDETERMINED - no decode-specific kernel "
                            "and no clean split in per-step kernel count")
        self.warnings.append(
            "This trace carries no signal that separates prefill from decode. "
            "The steps are real forward passes, but do not label them as "
            "phases from this view.")

    def _count_split(self):
        """Otsu threshold on per-step kernel count, if the split is real.

        Returns (cut, separation) or (None, 0). Separation is the ratio of the
        two group means; anything close to 1 means one population wearing a
        threshold, which is exactly the false structure worth refusing.
        """
        vals = sorted(s.n_kernels for s in self.steps)
        n = len(vals)
        if n < 20 or vals[0] == vals[-1]:
            return None, 0.0
        total = float(sum(vals))
        best, cut = -1.0, None
        run = 0.0
        for i in range(1, n):
            run += vals[i - 1]
            if vals[i] == vals[i - 1]:
                continue
            w0, w1 = i / n, (n - i) / n
            m0 = run / i
            m1 = (total - run) / (n - i)
            between = w0 * w1 * (m0 - m1) ** 2
            if between > best:
                best, cut = between, vals[i]
        if cut is None:
            return None, 0.0
        lo = [v for v in vals if v < cut]
        hi = [v for v in vals if v >= cut]
        if not lo or not hi:
            return None, 0.0
        sep = (sum(hi) / len(hi)) / max(1.0, sum(lo) / len(lo))
        # Below this the "two groups" are one distribution split arbitrarily.
        if sep < 1.35 or len(hi) < max(2, 0.01 * n):
            return None, 0.0
        return cut, sep

    # -- queries -------------------------------------------------------------
    def kernels_in(self, t0: int, t1: int, limit: int = 40000):
        """(start, end, family) for kernels overlapping [t0,t1]."""
        col = self._name_col()
        rows = self.db.execute(
            f"SELECT start, end, {col} FROM CUPTI_ACTIVITY_KIND_KERNEL "
            f"WHERE start < ? AND end > ? ORDER BY start LIMIT ?",
            (t1, t0, limit)).fetchall()
        return [(a, b, self.name_family.get(s, OTHER)) for a, b, s in rows]

    def step_at(self, i: int) -> Optional[Step]:
        return self.steps[i] if 0 <= i < len(self.steps) else None

    def phase_runs(self):
        """Consecutive steps of one phase, merged: (phase, t0, t1, n_steps).

        A decode phase is hundreds of steps and drawing hundreds of identical
        bands is both slow and unreadable; what the figure needs is where the
        phase changed.
        """
        out = []
        for s in self.steps:
            if out and out[-1][0] == s.phase:
                p, a, _b, k = out[-1]
                out[-1] = (p, a, s.t1, k + 1)
            else:
                out.append((s.phase, s.t0, s.t1, 1))
        return out

    def summary(self) -> dict:
        pre = [s for s in self.steps if s.phase == "prefill"]
        dec = [s for s in self.steps if s.phase == "decode"]
        def med(v):
            v = sorted(v)
            return v[len(v) // 2] if v else 0
        return dict(
            n_kernels=self.n_kernels,
            n_steps=len(self.steps),
            n_prefill=len(pre),
            n_decode=len(dec),
            span_ns=max(0, self.t1 - self.t0),
            prefill_ns=sum(s.dur_ns for s in pre),
            decode_ns=sum(s.dur_ns for s in dec),
            prefill_med_ns=med([s.dur_ns for s in pre]),
            decode_med_ns=med([s.dur_ns for s in dec]),
            busy_ns=sum(s.busy_ns for s in self.steps),
        )

    def family_totals(self, t0: int, t1: int) -> list:
        """(family, launches, busy_ns) over a window, busiest first."""
        agg: dict = {}
        for a, b, fam in self.kernels_in(t0, t1, limit=400000):
            n, busy = agg.get(fam, (0, 0))
            agg[fam] = (n + 1, busy + max(0, min(b, t1) - max(a, t0)))
        return sorted(((f, n, d) for f, (n, d) in agg.items()),
                      key=lambda r: -r[2])

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass
