#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
Headless smoke test for TTI-Scope.

Renders every view offscreen against a real store so a broken query or a
pyqtgraph API change fails here rather than in front of the user.

    QT_QPA_PLATFORM=offscreen python3 ttiscope_selftest.py <capture_dir>
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication          # noqa: E402

from ttiscope_app import (ChannelPanel, GlobalTimeline, KernelInventory,   # noqa: E402
                          MainWindow, StatsPanel, TTIView)
from ttiscope_ingest import SessionStore          # noqa: E402
from ttiscope_occupancy import theoretical_occupancy   # noqa: E402
import ttiscope_spare as spare                         # noqa: E402
from ttiscope_taxonomy import (DEVICE_FUNCTIONS, KERNEL_TAXONOMY,  # noqa: E402
                               classify_kernel)

FAIL = []


def check(name, fn):
    try:
        r = fn()
    except Exception:
        FAIL.append(name)
        print(f"  FAIL  {name}\n{traceback.format_exc()}")
        return None
    print(f"  ok    {name}" + (f"   {r}" if r else ""))
    return r


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    app = QApplication([])
    st = SessionStore.open_or_build(sys.argv[1], log=lambda m: None)

    print(f"\nstore: {st.path.name}")
    print(f"  {st.n_events:,} events / {st.n_slots:,} slots / "
          f"{len(st.kernels)} kernels / {st.n_cells} cells\n")

    print("store API")
    mid = st.n_slots // 2
    lo, hi = max(0, mid - 2), min(st.n_slots - 1, mid + 2)
    check("slot()", lambda: st.slot(mid))
    check("slot_label()", lambda: st.slot_label(mid))
    check("slot_of_ns()", lambda: st.slot_of_ns(st.slot(mid)[1] + 100))
    check("slot_stats()", lambda: f"{len(st.slot_stats(lo, hi))} rows")
    check("occupancy_series()", lambda: f"{len(st.occupancy_series(lo, hi))} rows")
    check("events_in_slot()", lambda: f"{len(st.events_in_slot(mid))} rows")
    check("events_in_window()",
          lambda: f"{len(st.events_in_window(st.slot(lo)[1], st.slot(hi)[2]))} rows")
    check("phases_in_window()",
          lambda: f"{len(st.phases_in_window(st.slot(lo)[1], st.slot(hi)[2]))} rows")
    check("phase_menu()", lambda: f"{len(st.phase_menu())} phases")
    check("tasks_in_slot()", lambda: f"{len(st.tasks_in_slot(mid))} rows")
    check("kernel_histogram()", lambda: f"{len(st.kernel_histogram(lo, hi))} rows")
    check("channel_histogram()", lambda: f"{len(st.channel_histogram(lo, hi))} rows")
    check("kernel_table()", lambda: f"{len(st.kernel_table())} rows")
    check("stream_map()", lambda: f"{len(st.stream_map())} streams")
    check("kernel_slot_durations()",
          lambda: f"{len(st.kernel_slot_durations(lo, hi))} (kernel,slot) rows")
    w_lo = max(0, mid - 500)
    w_hi = min(st.n_slots - 1, w_lo + 999)
    prof = check("intra_slot_profile() over 1000 TTI",
                 lambda: st.intra_slot_profile(w_lo, w_hi))
    if prof:
        # The five sub-windows partition the slot, so their energies must add
        # up to a slot's worth and none may exceed its own share of the cap.
        from ttiscope_occupancy import P_TDP
        bin_us = prof["slot_us"] / prof["n_bins"]
        cap = P_TDP * bin_us * 1e-6 * 1e6
        over = [i for i, e in enumerate(prof["e_max"]) if e > cap + 1]
        if over:
            FAIL.append("bin energy exceeds cap")
            print(f"  FAIL  bins {over} exceed {cap:,.0f} uJ per {bin_us:.0f} us")
        else:
            print(f"  ok    every bin within {cap:,.0f} uJ per {bin_us:.0f} us")
        bad = [i for i, o in enumerate(prof["occ"]) if not (0 <= o <= 100)]
        if bad:
            FAIL.append("bin occupancy out of range")
            print(f"  FAIL  bins {bad} have occupancy outside [0,100]")
        else:
            b = max(range(prof["n_bins"]), key=lambda i: prof["occ"][i])
            print(f"  ok    busiest window {prof['edges_us'][b]:.0f}-"
                  f"{prof['edges_us'][b+1]:.0f} us at {prof['occ'][b]:.1f}% "
                  f"occupancy, {prof['conc'][b]:.1f} concurrent kernels")

    print("\noccupancy + energy")
    kt = st.kernel_table()
    with_occ = [r for r in kt if r[10] is not None]
    print(f"  ok    {len(with_occ)}/{len(kt)} entries have theoretical occupancy")
    bad = [r[0] for r in with_occ if not (0 < r[10] <= 100)]
    if bad:
        FAIL.append("occupancy range")
        print(f"  FAIL  occupancy outside (0,100]: {bad[:5]}")
    else:
        print("  ok    all occupancies in (0,100]")
    # A device function has no launch config of its own, so it must not carry
    # an occupancy figure.
    df = [r for r in kt if r[15] == "device_fn"]
    if any(r[10] is not None for r in df):
        FAIL.append("device_fn occupancy")
        print("  FAIL  a device function was given an occupancy figure")
    else:
        print(f"  ok    {len(df)} device function(s), none given occupancy")
    # Check every slot in the capture, not just the sample window: the power
    # ceiling is a physical invariant, and a mean of 892 W on a 700 W part is
    # exactly the bug this exists to catch.
    from ttiscope_occupancy import P_TDP
    worst_w, worst_slot = 0.0, None
    for r in st.slot_stats(0, st.n_slots - 1):
        if not r[11]:
            continue
        w = r[11] * 1e-6 / ((r[2] - r[1]) * 1e-9)
        if w > worst_w:
            worst_w, worst_slot = w, r[0]
    if worst_slot is not None:
        if worst_w > P_TDP + 1.0:
            FAIL.append("energy exceeds TDP")
            print(f"  FAIL  slot {worst_slot} implies {worst_w:.0f} W, above "
                  f"the {P_TDP:.0f} W board limit")
        else:
            print(f"  ok    peak implied power {worst_w:.0f} W "
                  f"(slot {worst_slot}), within the {P_TDP:.0f} W limit")
    bad_slot = None
    for r in st.slot_stats(0, st.n_slots - 1):
        dur = r[2] - r[1]
        if r[7] > dur or r[10] > dur or r[7] + r[10] != dur:
            bad_slot = r
            break
    if bad_slot:
        FAIL.append("busy/idle vs slot length")
        print(f"  FAIL  slot {bad_slot[0]}: busy {bad_slot[7]} + idle "
              f"{bad_slot[10]} != slot length {bad_slot[2]-bad_slot[1]}")
    else:
        print("  ok    busy union + idle == slot length, every slot")

    print("\nwidgets")

    def mk(cls):
        w = cls()
        w.set_store(st)
        return w

    tl = check("GlobalTimeline", lambda: mk(GlobalTimeline))
    tv = check("TTIView", lambda: mk(TTIView))
    cp = check("ChannelPanel", lambda: mk(ChannelPanel))
    check("KernelInventory", lambda: mk(KernelInventory))
    sp = check("StatsPanel", lambda: mk(StatsPanel))
    if sp:
        sp.spin.setValue(min(1000, max(2, st.n_slots - 1)))
        check("StatsPanel.set_window (1000 TTI)",
              lambda: sp.set_window(lo, hi))
        check("StatsPanel metric rows",
              lambda: f"{sp.tbl.rowCount()} metrics, "
                      f"{sp.ktbl.rowCount()} kernels")
    if tv:
        check("TTIView.set_window (channel lanes)",
              lambda: tv.set_window(lo, hi))
        tv.lane_mode.setCurrentIndex(1)
        check("TTIView.set_window (stream lanes)",
              lambda: tv.set_window(lo, hi))
        tv.show_ktrace.setChecked(False)
        check("TTIView without KTRACE", lambda: tv.redraw())
        tv.show_ktrace.setChecked(True)
        if tv._bars:
            check("TTIView._describe",
                  lambda: f"{len(tv._describe(tv._bars[0][3]).splitlines())} lines")
    if cp:
        check("ChannelPanel.set_window", lambda: cp.set_window(lo, hi))
    if tl:
        check("GlobalTimeline window emit", lambda: tl._emit())

    print("\ncoverage")
    unmapped = [v[0] for v in st.kernels.values() if v[1] == "Unknown"]
    if unmapped:
        FAIL.append("taxonomy coverage")
        print(f"  FAIL  {len(unmapped)} kernels unmapped: {unmapped[:8]}")
    else:
        print(f"  ok    all {len(st.kernels)} kernels classified")
    stale = [k for k in KERNEL_TAXONOMY
             if k not in {v[0] for v in st.kernels.values()}]
    print(f"  note  {len(stale)} taxonomy entries not exercised by this "
          f"capture{': ' + str(stale[:5]) if stale else ''}")

    print("\ncpu profile")
    import os as _os
    from ttiscope_cpu import (IDLE_COMM, build_profile, discover_cpu_captures,
                              SAMPLE_HZ)
    root = _os.environ.get("TTISCOPE_PERF_ROOT", "")
    caps = discover_cpu_captures(root) if root else {}
    if not caps:
        print(f"  note  no perf captures (set TTISCOPE_PERF_ROOT); CPU panel untested")
    else:
        cells = st._m("n_cells", int, 0)
        cap = next((c for c in caps.values() if c.cells == cells), None)
        if cap is None:
            print(f"  note  {len(caps)} captures indexed, none at {cells}C")
        else:
            prof = check(f"build_profile({cells}C)",
                         lambda: build_profile(cap, log=lambda m: None))
            if prof and not prof.note:
                # The traffic window must be a real subset: `sleep 45` bounds
                # perf, not the DU, and using the full span dilutes every
                # figure by the idle tail.
                if not (0 < prof.window_s <= prof.span_s):
                    FAIL.append("cpu traffic window")
                    print(f"  FAIL  window {prof.window_s} vs span {prof.span_s}")
                else:
                    print(f"  ok    traffic window {prof.window[0]:.0f}-"
                          f"{prof.window[1]:.0f}s of {prof.span_s:.0f}s")
                bad = [c["cpu"] for c in prof.cores if not (0 <= c["pct"] <= 105)]
                if bad:
                    FAIL.append("cpu util range")
                    print(f"  FAIL  cores {bad[:5]} outside 0-100%")
                else:
                    print(f"  ok    all {prof.n_cores} cores within 0-100%")
                # Idle must never be counted as busy — pid 0 arrives with an
                # unresolved comm and outnumbers every DU thread.
                leak = [t["comm"] for t in prof.threads if t["comm"] in IDLE_COMM]
                if leak:
                    FAIL.append("idle counted as busy")
                    print(f"  FAIL  idle comms present in busy threads: {leak}")
                else:
                    print("  ok    idle tasks excluded from busy time")
                phy = prof.phy_cores()
                if not phy:
                    FAIL.append("no PHY cores identified")
                    print("  FAIL  no core attributed to a PHY worker")
                else:
                    print(f"  ok    {len(phy)} PHY cores, peak "
                          f"{max(c['pct'] for c in phy):.0f}%, DU "
                          f"{prof.du_pct:.1f}% of {prof.n_cores} cores")

    # ── spare GPU --> rendered as Oracle Phantom Kernel stream ────────────────────────────────
    print("\nspare GPU")
    ctx = check("spare.context()", lambda: spare.context(st))
    if ctx:
        rf = ctx.rf
        check("roofline from store",
              lambda: f"{rf.gpu_name} {rf.num_sm} SM, fb {rf.fb_total_mib}")

        def _table_measured():
            # Every per-SM limit must come from TARGET_INFO_GPU in the capture,
            # never from the hardcoded sm_90 table. Falling back silently would
            # score a non-Hopper capture against Hopper's limits.
            assert rf.table_src == spare.MEASURED, \
                "per-SM limits fell back to the assumed sm_90 table"
            assert rf.max_threads_per_sm == rf.max_warps_per_sm * rf.warp_size
            for nm in ("max_warps_per_sm", "max_blocks_per_sm", "regs_per_sm",
                       "smem_per_sm", "warp_size"):
                assert getattr(rf, nm) > 0, f"{nm} not read"
            return (f"sm_{rf.compute_cap.replace('.', '')} {rf.chip_name}: "
                    f"{rf.max_warps_per_sm}w/SM {rf.max_blocks_per_sm}blk/SM "
                    f"{rf.smem_per_sm//1024}KiB {rf.regs_per_sm//1024}Kreg")
        check("per-SM limits read from the capture", _table_measured)
        check("anchor provenance", lambda: ctx.note())
        check("MPS caps", lambda: ctx.caps_note())

        def _occ_calc():
            # The occupancy calculator, with the 1 KiB/block runtime reserve:
            # README's worked example, 32 thr, 114 regs, 18,752 B smem.
            occ, blk, _w, lim = theoretical_occupancy(32, 114, 18752)
            assert blk == 11 and lim == "shared memory", (blk, lim)
            assert theoretical_occupancy(1025, 16, 0)[3] == "invalid"
            return f"{occ:.1f}% / {blk} blk/SM, {lim}-limited"
        check("occupancy calculator (reserve, per-block limit)", _occ_calc)

        def _reserve():
            # Derived from the capture: per-SM smem minus the opt-in per-block
            # maximum. 1,024 B on every sm_80+ part.
            assert rf.reserved_smem_per_block == 1024, rf.reserved_smem_per_block
            return f"{rf.reserved_smem_per_block} B/block ({rf.reserve_src})"
        check("per-block smem reserve", _reserve)

        def _caps_applied():
            n = st._m("n_ctx_capped", int, 0)
            if n:
                assert ctx.caps, "store has SM caps but the model ignored them"
                assert all(0 < c <= rf.num_sm for c in ctx.caps.values())
            return f"{len(ctx.caps)} capped context(s)"
        check("MPS SM caps reach the model", _caps_applied)

        def _demand_physical():
            # With allocation against device capacity, UNSCALED demand can no
            # longer exceed the roofline in any resource - the impossible
            # 280-warps/SM peaks of the summation model are gone by design.
            a, b = st.slot(lo)[1], st.slot(hi)[2]
            was = ctx.apply_scale
            ctx.apply_scale = False
            try:
                pr = spare.demand_profile(st, a, b, ctx)
            finally:
                ctx.apply_scale = was
            for nm, arr, cap in (("warps", pr.wrp, rf.max_warps_per_sm),
                                 ("blocks", pr.blk, rf.max_blocks_per_sm),
                                 ("smem", pr.smem, rf.smem_per_sm),
                                 ("regs", pr.regs, rf.regs_per_sm)):
                assert max(arr) <= cap + 1e-6, f"{nm} demand {max(arr)} > {cap}"
            return (f"peak {max(pr.wrp):.1f} warps/SM, mean occ "
                    f"{pr.mean_occ:.3f}, {pr.n_queued} queued stretches")
        check("unscaled demand within roofline", _demand_physical)

        def _validated():
            if ctx.dcgm_occ is None:
                return "no DCGM in this capture - not validated"
            r = ctx.model_occ / ctx.dcgm_occ
            # Wave tails make the model read high; more than 2x either way
            # means the caps or the allocation are not doing their job.
            assert 0.5 <= r <= 2.0, f"model/DCGM = {r:.2f}"
            return ctx.validation_note()
        check("model vs DCGM within 2x", _validated)

        def _anchor_sane():
            # The whole point of the anchor. Unanchored, the residency model
            # exceeds the roofline: measured on A2_20C-59c it peaks at 280
            # warps/SM on a 64-warp part and its integral form reaches 172% of
            # a 100% machine. If anchored mean occupancy ever comes back above
            # 100% the correction has silently stopped being applied.
            a, b = st.slot(lo)[1], st.slot(hi)[2]
            pr, _sp = spare.window_spare(st, a, b, ctx)
            occ = pr.occ_mean(rf)
            assert occ <= 100.0, f"anchored occupancy {occ:.1f}% exceeds 100%"
            return f"{occ:.1f}% mean occupancy, <= 100% as required"
        check("anchored occupancy within roofline", _anchor_sane)

        def _spare_clamped():
            a, b = st.slot(lo)[1], st.slot(hi)[2]
            _pr, (sb, sw, ss, sr) = spare.window_spare(st, a, b, ctx)
            for nm, arr, cap in (("blocks", sb, rf.max_blocks_per_sm),
                                 ("warps", sw, rf.max_warps_per_sm),
                                 ("smem", ss, rf.smem_per_sm),
                                 ("regs", sr, rf.regs_per_sm)):
                assert min(arr) >= 0.0, f"{nm} spare went negative"
                assert max(arr) <= cap + 1e-6, f"{nm} spare above roofline"
            return "all four dimensions within [0, roofline]"
        check("spare envelope clamped", _spare_clamped)

        def _shapes_launchable():
            # A shape with R=0 or k=0 marked feasible would be handed to the
            # harvester and fail at cudaLaunchKernel.
            rep = spare.analyse(st, lo, hi, bin_ns=1000, max_tti=20)
            n_f = 0
            for D, _p5, _p50, _p95, v in rep.frontier:
                if not v.feasible:
                    assert v.reason, f"D={D} infeasible with no reason given"
                    continue
                n_f += 1
                assert v.k >= 1 and v.B >= 32
                assert v.R >= spare.MIN_REGS_PER_THREAD
                assert v.S <= rf.smem_per_block_optin
                assert v.smem_optin == (v.S > spare.SMEM_DEFAULT_LIMIT)
                assert v.G == v.k * ctx.n_sm_cotenant
                assert v.warps_per_sm <= rf.max_warps_per_sm + 1e-6
                assert v.k <= rf.max_blocks_per_sm
            return f"{n_f} feasible frontier points, all launchable"
        check("frontier shapes launchable", _shapes_launchable)

        def _frontier_monotone():
            # Capacity sustainable for longer can never exceed capacity
            # sustainable for less: a sliding minimum over a wider window is
            # non-increasing. A violation means the window logic is wrong.
            rep = spare.analyse(st, lo, hi, bin_ns=1000, max_tti=20)
            prev = None
            for D, p5, _p50, _p95, _v in rep.frontier:
                if prev is not None:
                    assert p5 <= prev + 1e-6, f"p5 rose at D={D}"
                prev = p5
            return "p5 non-increasing in D"
        check("frontier monotone in duration", _frontier_monotone)

        def _phantom_respects_envelope():
            a, b = st.slot(lo)[1], st.slot(hi)[2]
            bn = spare.auto_bin_ns(a, b)
            ph, (sb, sw, ss, sr), _pr = spare.window_phantoms(
                st, a, b, ctx, bin_ns=bn)
            for q in ph:
                assert q.i1 > q.i0, "zero-length phantom"
                for i in range(q.i0, min(q.i1, len(sw))):
                    assert q.vec.warps_per_sm <= sw[i] + 1e-6, \
                        f"phantom overruns the warp envelope at bin {i}"
            return f"{len(ph)} phantoms, none overrunning the envelope"
        check("phantom lane inside spare envelope", _phantom_respects_envelope)

        def _phantom_toggle():
            v = TTIView()
            v.set_store(st)
            v.show_phantom.setChecked(False)
            v.set_window(lo, hi)
            n_off = len(getattr(v, "_phantoms", []))
            v.show_phantom.setChecked(True)
            v.redraw()
            n_on = len(getattr(v, "_phantoms", []))
            assert n_off == 0, "phantom lane drew while switched off"
            assert n_on > 0, "phantom lane drew nothing while switched on"
            v.show_spare()
            assert "SPARE GPU" in v.info.toPlainText()
            return f"off={n_off} on={n_on}, Spare GPU panel renders"
        check("TTIView phantom toggle + Spare GPU panel", _phantom_toggle)

        def _phantom_click():
            # Clicking a phantom must produce a record. The lane is an ORACLE:
            # bars are uncapped and end only when Aerial reclaims a resource
            # or the window does, so a bar longer than a TTI is expected, not
            # a fault. What must hold is that it never overruns the envelope
            # and that it took everything integer block placement allowed.
            from PyQt6.QtCore import QPointF
            v = TTIView()
            v.set_store(st)
            v.show_phantom.setChecked(True)
            v.set_window(lo, hi)
            assert v._phantoms, "no phantoms to click"
            # MAXIMALITY. Free capacity left over is not evidence of a
            # missed opportunity: 24 idle block slots are unusable when the
            # warps are gone, and 11.9 idle warps are unusable when the
            # register file is. The real test is that ONE MORE BLOCK of this
            # shape would have exceeded at least one resource.
            ws = rf.warp_size
            for _xa, _xb, _b0, _b1, q in v._phantoms:
                assert q.retired_by in ("warps", "blocks", "shared memory",
                                        "registers", "end of window",
                                        "duration cap", "relaunch"), \
                    f"unknown retirement reason {q.retired_by!r}"
                vv = q.vec
                if not (vv.feasible and q.margin):
                    continue
                wpb = max(1, -(-vv.B // ws))
                rpw = spare._ceil_to(vv.R * ws, rf.reg_alloc_unit)
                need = {"blocks": 1.0, "warps": float(wpb),
                        "shared memory": float(rf.block_smem(vv.S)),
                        "registers": float(wpb * rpw)}
                room = [k for k, n in need.items()
                        if q.margin.get(k, 0.0) + 1e-6 >= n]
                assert len(room) < len(need), (
                    f"another {vv.B}-thread block fitted in every dimension "
                    f"({q.margin}) - not an upper limit")
            xa, xb, b0, b1, q = max(v._phantoms, key=lambda t: t[4].dur_us)

            class _Ev:
                def __init__(self, sp):
                    self._sp = sp

                def scenePos(self):
                    return self._sp

            v._on_click(_Ev(v.p_kern.vb.mapViewToScene(
                QPointF((xa + xb) / 2.0, (b0 + b1) / 2.0))))
            txt = v.info.toPlainText()
            assert v.info_title.text() == "Phantom kernel", "wrong panel title"
            for want in ("PHANTOM", "NOT CAPTURED", "SPARE VECTOR",
                         "hand this to the harvester as:", "grid=", "block=",
                         "HOW THIS SHAPE WAS DERIVED", "WHAT THIS IS NOT"):
                assert want in txt, f"phantom record missing {want!r}"
            return (f"record renders, longest phantom {q.dur_us:.0f}us, "
                    f"{len(v._phantoms)} shapes all maximal")
        check("phantom click record + oracle upper limit", _phantom_click)

        def _oracle_brute():
            # Algorithm 3 against exhaustive enumeration on random small
            # envelopes: the DP value must equal the true optimum.
            import random
            from functools import lru_cache
            rnd = random.Random(7)
            for trial in range(120):
                m = rnd.randint(1, 7)
                tau = [0]
                for _ in range(m):
                    tau.append(tau[-1] + rnd.choice([2000, 8000, 20000, 60000]))
                C = [(rnd.choice([0, 3, 32]), rnd.choice([0, 6, 40, 64]),
                      rnd.choice([0, 4096, 233472]),
                      rnd.choice([0, 8192, 65536])) for _ in range(m)]
                spec = spare.CotenantSpec(B=rnd.choice([None, 128]),
                                          S=rnd.choice([0, 2048]), R=16)
                _s, dp, _a = spare.oracle_schedule(tau, C, rf, spec)
                lam, dmin = spec.launch_gap_us * 1e3, spec.d_min_us * 1e3
                hs = [spare.block_cost(rf, B, spec.S, spec.R)
                      for B in spec.blocks(rf)]

                @lru_cache(None)
                def best(t):
                    if t >= m:
                        return 0.0
                    r = best(t + 1)
                    for e in range(t + 1, m + 1):
                        L_ = tau[e] - tau[t]
                        if L_ < dmin or L_ <= lam:
                            continue
                        cm = [min(C[j][d] for j in range(t, e))
                              for d in range(4)]
                        v = max((h[1] * spare.kstar(cm, h) * (L_ - lam)
                                 for h in hs), default=0.0)
                        if v > 0:
                            r = max(r, v + best(e))
                    return r
                assert abs(dp - best(0)) < 1e-3, (trial, dp, best(0))
            return "DP equals brute-force optimum on 120 random envelopes"
        check("oracle optimal (DP vs brute force)", _oracle_brute)

        def _oracle_invariants():
            a, b = st.slot(lo)[1], st.slot(hi)[2]
            pp = spare.demand_profile(st, a - st.slot_dur_ns,
                                      b + st.slot_dur_ns, ctx, 10_000)
            tau, C = spare.spare_segments(pp, rf, "conservative")
            sched, cap, avail = spare.oracle_schedule(tau, C, rf, ctx.spec)
            prev_e = -1
            for it in sched:
                s_, e_, k = it["s"], it["e"], it["k"]
                h = it["shape"][1:]
                assert s_ >= prev_e, "oracle kernels overlap"
                prev_e = e_
                assert tau[e_] - tau[s_] >= ctx.spec.d_min_us * 1e3 - 1e-6
                cm = [min(C[j][d] for j in range(s_, e_)) for d in range(4)]
                assert k == spare.kstar(cm, h), "k is not K* of its interval"
                assert all(k * h[d] <= cm[d] + 1e-6 for d in range(4))
            # The greedy fill, scored by the same objective, cannot beat it.
            bn = 1000
            p1, env = spare.window_spare(st, a, b, ctx, bn)
            greedy = spare.phantom_schedule(*env, bn, a, rf, max_live=1)
            lam = ctx.spec.launch_gap_us * 1e3
            g_val = sum(q.vec.warps_per_sm * max(0.0, q.t1 - q.t0 - lam)
                        for q in greedy)
            _ph, o_cap, _av = spare.window_oracle(st, a, b, ctx, bn)
            assert o_cap + 1e-6 >= g_val * 0.999, (o_cap, g_val)
            return (f"{len(sched)} kernels, non-overlapping, k = K* each; "
                    f"oracle {o_cap/1e3:,.0f} >= greedy {g_val/1e3:,.0f} "
                    f"warp-us/SM")
        check("oracle invariants + beats greedy", _oracle_invariants)

    print("\nmain window")
    mw = check("MainWindow", lambda: MainWindow())
    if mw:
        import ttiscope_app as app_mod
        base = app_mod.SCALE

        def zoom(n, d):
            for _ in range(n):
                mw.zoom_ui(d)
            return f"scale {app_mod.SCALE:.1f}"
        check("Ctrl+ zoom in", lambda: zoom(5, +1))
        check("Ctrl- zoom out", lambda: zoom(10, -1))
        check("Ctrl+0 reset", lambda: zoom(1, 0))
        if abs(app_mod.SCALE - 1.0) > 1e-6:
            FAIL.append("zoom reset")
            print(f"  FAIL  Ctrl+0 left scale at {app_mod.SCALE}")
        # Clamps must hold, or a held-down key walks the UI off the screen.
        zoom(60, +1)
        hi_ok = app_mod.SCALE <= app_mod.SCALE_MAX + 1e-9
        zoom(60, -1)
        lo_ok = app_mod.SCALE >= app_mod.SCALE_MIN - 1e-9
        zoom(1, 0)
        if hi_ok and lo_ok:
            print(f"  ok    scale clamped to "
                  f"[{app_mod.SCALE_MIN}, {app_mod.SCALE_MAX}]")
        else:
            FAIL.append("zoom clamp")
            print("  FAIL  scale escaped its clamp")

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} — {FAIL}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
