#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
TTI-Scope — full-pipeline of UL and DL in the context of 5G RAN TTI. GPU-Kernel launching pattern viewer for NVIDIA Aerial
cuBB.


"""
from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import (Qt, QEvent, QThread, QMarginsF, QRectF, QSize,
                          pyqtSignal)
from PyQt6.QtGui import (QAction, QColor, QFont, QFontMetrics, QKeySequence,
                         QShortcut,
                         QPageLayout, QPageSize, QPainter, QPdfWriter)
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from ttiscope_ingest import SessionStore, export_nsys_rep
from ttiscope_llm import (FAMILY_COLOR, FAMILY_ORDER, LLMTrace)
from ttiscope_cpu import ROLE_COLOR, build_profile, intra_tti_profile
from ttiscope_library import SessionEntry, SessionLibrary
from ttiscope_occupancy import P_IDLE, P_TDP, energy_provenance
import ttiscope_spare as spare
from ttiscope_taxonomy import CHANNEL_COLOR, channel_color

ENERGY_NOTE = (
    "Energy is MODELLED, not measured: no capture here was taken with "
    "--gpu-metrics-device, so there is no power telemetry in the data. "
    f"E = P_idle x slot + sum of marginal kernel power x busy time, with "
    f"P_idle {P_IDLE:.0f} W and a {P_TDP:.0f} W TDP budget. See Help > "
    "Energy model for the provenance of each constant.")

APP = "TTI-Scope"

BG = "#14161a"
BG2 = "#1a1d22"
# Controls sat on BG2 inside a GRID border - three values within a few points
# of each other, so a dropdown was a dark rectangle on a dark ground and read
# as decoration rather than something you could click. These two lift the
# interactive furniture clear of the panel it sits on.
BG3 = "#252a32"           # control fill
EDGE = "#4a525d"          # control border
FG = "#e6e8ea"
MUTED = "#8b939c"
# Labels next to a control. MUTED is right for footnotes and wrong for the
# thing telling you what a dropdown does.
LABEL = "#c8d0d8"
GRID = "#2a2f37"
ACCENT = "#4FA3E8"
# Kernel-lane furniture that lives ABOVE the top lane. TOP_PAD is the Y
# headroom reserved for it: SFN.slot captions at +0.42 and TTI-MISS crosses at
# MISS_Y. Without the reserve the default range spans only the lanes and both
# are drawn outside the visible rect.
TOP_PAD = 1.45
MISS_Y = 1.05
MISS_X = "#FF3B30"        # TTI the DU dropped
# The TTI grid is the frame of reference for everything else in this pane, so
# it is drawn in a colour nothing else uses. GRID (#2a2f37) put the slot
# boundaries at roughly the same value as the panel background: technically
# present, effectively invisible against the kernel bars.
TTI_Y   = "#FFD400"       # slot boundary line
TTI_W   = 2               # px; 1 disappeared under the bars
# SFN captions carry the state of their slot: a dropped TTI is red and bold,
# every other slot is blue. Both are kept under lightness 140 deliberately -
# the PDF export rewrites lighter text to grey for the white ground, so the
# obvious accent blue (#4FA3E8, lightness 156) would have printed grey and
# MISS_X (152) would have lost exactly the emphasis it was carrying.
SFN_OK   = "#2E86DE"
SFN_MISS = "#E03131"
# How much larger text is in an exported figure than on screen. These pages
# go into a paper at column width, where screen-sized type stops being legible.
PDF_FONT_BOOST = 2.2
# Minimum vertical pitch per lane, in screen px, that the exported figure is
# given. \includegraphics scales the whole page to the column, so what decides
# printed text size is the ratio of font height to figure width - and that
# ratio is capped by how much room a lane has. At the on-screen pitch of ~13 px
# the channel names could only grow 1.15x and printed at about 5 pt.
PDF_LANE_PITCH = 28
PHANTOM_FILL = "#B084F0"  # hypothetical co-tenant kernel (spare GPU)
PHANTOM_EDGE = "#DCC8FF"
FIND_FILL = "#76B900"     # search hit
FIND_EDGE = "#EAFBD0"
# The TTI window markers. Deliberately the brightest thing on the Timeline —
# they define what every other tab is reporting.
SELECT = "#2E9BFF"
SELECT_HI = "#7CC6FF"

# Lane order for the kernel Gantt — signal flow, not alphabetical: what the
# host prepares, what goes out, what comes back, what decodes it.
CHANNEL_LANES = ["FH-DL", "PDSCH", "PDCCH", "CSI-RS", "SSB",
                 "FH-UL", "PUSCH", "UCI", "PUCCH", "PRACH",
                 "Graph", "Utility", "Unknown"]

# NVTX phases worth a lane. cuPHY emits 33 distinct ranges; these are the ones
# that describe the slot rather than an internal sub-step.
PHASE_LANES = ["SLOT_DL", "SLOT_UL", "cuphySetupPdschTx", "cuphyRunPdschTx",
               "PDCCH_SETUP", "PDCCH_RUN", "cuphySetupPuschRx-P1",
               "cuphySetupPuschRx-P2", "cuphyRunPuschRx-ALL",
               "cuphySetupPucchRx", "cuphyRunPucchRx", "cuphySetupPrachRx",
               "cuphyRunPrachRx", "cuphySetupCsirsTx", "DL WAIT COMP",
               "UL CPlane", "UL CLEAN"]

pg.setConfigOptions(antialias=True, background=BG, foreground=FG)


def _q(c: str, alpha: int = 255) -> QColor:
    q = QColor(c)
    q.setAlpha(alpha)
    return q


# ─────────────────────────────────────────────────────────────────────────────
# UI scale (Ctrl +/-/0)
# ─────────────────────────────────────────────────────────────────────────────
#
# Every size in this file goes through _s(), so one factor rescales the whole
# window. Scaling only the application font would leave stylesheet px sizes,
# plot axis fonts and table row heights behind — which is exactly what makes
# font-only zoom look half-applied.
SCALE = 1.0
SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.6, 2.6, 0.1
_BASE_POINT = 10
_scaled_labels: list = []


def _s(n) -> int:
    return max(1, int(round(n * SCALE)))


_TICK_FONT = None


def _pattern_label(label: str) -> str:
    """The traffic pattern out of a run directory name.

    Run identity is <CAMPAIGN>_<CELLS>C-<PATTERN>_<TENANT>_L<lambda>_<INSTR>_
    r<NN>_<UTC>, so "A2_20C-59c_solo_L0_nsys_r01_20260829T034933Z" carries
    pattern 59c. On a figure the campaign, tenant, lambda, replica and
    timestamp are noise - what identifies the workload is the pattern, and the
    cell count is reported separately from the capture itself.
    """
    m = re.search(r"(?:^|_)\d+C-([0-9]+[a-zA-Z]?)(?:_|$)", label)
    return m.group(1).upper() if m else label


def _tick_font() -> QFont:
    """Small axis font, rebuilt whenever the UI scale changes.

    Must be a fully-formed QFont: pyqtgraph's AxisItem.generateDrawSpecs
    segfaults on a QFont constructed with an empty family string, which is
    exactly what QFont("", 8) produces.
    """
    global _TICK_FONT
    if _TICK_FONT is None:
        f = QApplication.font()
        f.setPointSize(max(5, int(round(8 * SCALE))))
        _TICK_FONT = f
    return _TICK_FONT


def _restyle_label(w):
    size, color, bold = w._tti_style
    w.setStyleSheet(f"font-size:{_s(size)}px;color:{color};"
                    f"font-weight:{'600' if bold else '400'};")


def _lbl(text="", size=11, color=FG, bold=False):
    w = QLabel(text)
    w._tti_style = (size, color, bold)
    _restyle_label(w)
    w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    _scaled_labels.append(w)
    return w


def _style_main() -> str:
    return (f"QMainWindow,QWidget{{background:{BG};color:{FG};}}"
            f"QTabWidget::pane{{border:1px solid {GRID};"
            f"border-radius:{_s(4)}px;}}"
            f"QTabBar::tab{{background:{BG};color:{MUTED};"
            f"padding:{_s(7)}px {_s(16)}px;font-size:{_s(11)}px;"
            f"border:1px solid {GRID};border-bottom:0;"
            f"border-top-left-radius:{_s(4)}px;"
            f"border-top-right-radius:{_s(4)}px;}}"
            f"QTabBar::tab:selected{{background:{BG2};color:{FG};}}"
            f"QSplitter::handle{{background:{GRID};}}")


def _style_banner() -> str:
    return (f"font-size:{_s(11)}px;color:{MUTED};background:{BG2};"
            f"padding:{_s(8)}px {_s(10)}px;border:1px solid {GRID};"
            f"border-radius:{_s(4)}px;")


def _style_list() -> str:
    return (f"QListWidget{{background:{BG2};border:1px solid {GRID};"
            f"border-radius:{_s(4)}px;font-size:{_s(11)}px;padding:{_s(2)}px;}}"
            f"QListWidget::item{{padding:{_s(5)}px {_s(6)}px;"
            f"border-radius:{_s(3)}px;}}"
            f"QListWidget::item:selected{{background:{ACCENT};color:#06121d;}}")


def _style_combo() -> str:
    # The popup list needs styling of its own: QComboBox's stylesheet does not
    # reach it, so the drop-down was rendering in the default palette - light
    # rows under a dark app, or unreadable selection colours.
    return (f"QComboBox{{background:{BG3};color:{FG};border:1px solid {EDGE};"
            f"border-radius:{_s(4)}px;padding:{_s(4)}px {_s(8)}px;"
            f"font-size:{_s(12)}px;font-weight:600;min-height:{_s(18)}px;}}"
            f"QComboBox:hover{{border-color:{ACCENT};background:{BG3};}}"
            f"QComboBox:focus{{border-color:{ACCENT};}}"
            f"QComboBox::drop-down{{border:0;width:{_s(20)}px;}}"
            f"QComboBox::down-arrow{{width:0;height:0;margin-right:{_s(7)}px;"
            f"border-left:{_s(5)}px solid transparent;"
            f"border-right:{_s(5)}px solid transparent;"
            f"border-top:{_s(6)}px solid {FG};}}"
            f"QComboBox QAbstractItemView{{background:{BG3};color:{FG};"
            f"border:1px solid {EDGE};outline:0;"
            f"font-size:{_s(12)}px;"
            f"selection-background-color:{ACCENT};selection-color:#0b0e12;}}")


def _style_spin() -> str:
    return (f"QSpinBox{{background:{BG3};color:{FG};border:1px solid {EDGE};"
            f"border-radius:{_s(4)}px;padding:{_s(4)}px {_s(6)}px;"
            f"font-size:{_s(12)}px;font-weight:600;min-height:{_s(18)}px;}}"
            f"QSpinBox:hover{{border-color:{ACCENT};}}"
            f"QSpinBox:focus{{border-color:{ACCENT};}}")


def _style_edit() -> str:
    return (f"QLineEdit{{background:{BG3};color:{FG};border:1px solid {EDGE};"
            f"border-radius:{_s(4)}px;padding:{_s(4)}px {_s(8)}px;"
            f"font-size:{_s(12)}px;font-weight:600;min-height:{_s(18)}px;}}"
            f"QLineEdit:hover{{border-color:{ACCENT};}}"
            f"QLineEdit:focus{{border-color:{ACCENT};}}")


def _style_check() -> str:
    return (f"QCheckBox{{color:{FG};font-size:{_s(12)}px;}}"
            f"QCheckBox::indicator{{width:{_s(14)}px;height:{_s(14)}px;"
            f"border:1px solid {EDGE};border-radius:{_s(3)}px;"
            f"background:{BG3};}}"
            f"QCheckBox::indicator:checked{{background:{ACCENT};"
            f"border-color:{ACCENT};}}")


def _style_button() -> str:
    return (f"QPushButton{{background:{BG3};color:{FG};border:1px solid "
            f"{EDGE};border-radius:{_s(4)}px;padding:{_s(5)}px {_s(9)}px;"
            f"font-size:{_s(12)}px;}}"
            f"QPushButton:hover{{border-color:{ACCENT};}}")


def _style_table() -> str:
    return (f"QTableWidget{{background:{BG2};color:{FG};border:1px solid "
            f"{GRID};border-radius:{_s(4)}px;font-size:{_s(11)}px;"
            f"gridline-color:{GRID};}}"
            f"QHeaderView::section{{background:{BG};color:{MUTED};border:0;"
            f"border-bottom:1px solid {GRID};padding:{_s(5)}px;"
            f"font-size:{_s(10)}px;}}")


def _style_text() -> str:
    return (f"background:{BG2};color:{FG};border:1px solid {GRID};"
            f"border-radius:{_s(4)}px;font-family:monospace;"
            f"font-size:{_s(11)}px;padding:{_s(6)}px;")


def _rescale(app, root):
    """Re-apply every scale-dependent size after SCALE changes.

    Walks the widget tree rather than keeping a registry, so a panel added
    later cannot be silently left behind at the old size. Panels that own
    stylesheets expose a restyle() for the walk to call.
    """
    global _TICK_FONT
    _TICK_FONT = None

    f = app.font()
    f.setPointSize(max(6, int(round(_BASE_POINT * SCALE))))
    app.setFont(f)

    for w in list(_scaled_labels):
        try:
            _restyle_label(w)
        except RuntimeError:          # the C++ object is already gone
            _scaled_labels.remove(w)

    for w in root.findChildren(QWidget):
        fn = getattr(w, "restyle", None)
        if callable(fn):
            try:
                fn()
            except RuntimeError:
                pass

    # Plots are most of this window's area, so "zoom the whole window" has to
    # reach inside them: tick fonts, axis label fonts and plot titles, not just
    # the Qt widgets around them.
    plots = []
    for pw in root.findChildren(pg.PlotWidget):
        plots.append(pw.plotItem)
    for gl in root.findChildren(pg.GraphicsLayoutWidget):
        for item in gl.ci.items:
            if isinstance(item, pg.PlotItem):
                plots.append(item)
    lbl_pt = max(6, int(round(9 * SCALE)))
    for item in plots:
        for side in ("left", "bottom", "top", "right"):
            ax = item.axes.get(side)
            if not ax:
                continue
            a = ax["item"]
            a.setStyle(tickFont=_tick_font())
            if a.label is not None and a.labelText:
                a.labelStyle["font-size"] = f"{lbl_pt}pt"
                a.setLabel()
        try:
            t = item.titleLabel
            if t is not None and t.text:
                t.setText(t.text, size=f"{lbl_pt}pt",
                          color=t.opts.get("color", MUTED))
        except Exception:
            pass

    # Column widths were measured in the old font.
    for t in root.findChildren(QTableWidget):
        _fit_columns(t)
    root.update()


def _fmt_wall(ns, micros=True) -> str:
    """Absolute UTC timestamp, formatted to match the cuPHY nvlog line prefix.

    The DU logs and nsys agree on this clock: the session epoch comes from
    TARGET_INFO_SESSION_START_TIME.utcEpochNs, and the nvlog's own timestamps
    are TAI with a zero offset ("Current TAI offset: 0s"). So a time shown here
    can be grepped verbatim in cuphy_<N>C_<pat>.log.
    """
    if ns is None:
        return "—"
    ns = int(ns)
    s, rem = divmod(ns, 1_000_000_000)
    t = time.gmtime(s)
    frac = f"{rem//1000:06d}" if micros else f"{rem//1_000_000:03d}"
    return (f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d} "
            f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{frac}")


def _fmt_clock(ns, micros=True) -> str:
    """Time-of-day only — for axis ticks, where the date is redundant."""
    if ns is None:
        return ""
    s, rem = divmod(int(ns), 1_000_000_000)
    t = time.gmtime(s)
    frac = f"{rem//1000:06d}" if micros else f"{rem//1_000_000:03d}"
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{frac}"


class _PinnedScale:
    """Render an export at the canonical UI scale, whatever the window is at.

    Every size in this app derives from SCALE, the interface zoom - including
    the axis fonts. That made an exported figure depend on how far the user had
    zoomed: the same window exported at 1.0 and 1.2 gave two different figures,
    and past 1.2 the axis title no longer fitted and was dropped entirely. A
    figure for a paper should not carry the state of the window it came from.

    With SCALE pinned to 1.0, _s() is the identity, so the export geometry and
    every font in it are the canonical ones.
    """

    def __init__(self, root):
        self.root = root
        self.old = None

    def __enter__(self):
        global SCALE
        if abs(SCALE - 1.0) < 1e-6:
            return self
        self.old = SCALE
        SCALE = 1.0
        self._apply()
        return self

    def __exit__(self, *exc):
        global SCALE
        if self.old is not None:
            SCALE = self.old
            self._apply()
        return False

    def _apply(self):
        try:
            _rescale(QApplication.instance(), self.root)
            QApplication.processEvents()
        except Exception:
            pass


def _write_export_geometry(pdf_path, item, src, sc, tgt, writer):
    """Emit the page geometry of a plot's data area beside the PDF.

    An overlay drawn in LaTeX - a zoom callout, an annotation, a marker at a
    known time - has to know where a DATA coordinate falls on the page. That
    depends on the axis margins inside the figure, which are invisible from
    outside and are what people end up estimating off a screenshot. Both ends
    of the axis are known exactly at render time, so they are recorded.

    Fractions are of the FULL page, because that is the bounding box
    ``\\includegraphics`` uses. The file also carries the page size so a trim
    can be corrected without re-measuring.
    """
    vb = getattr(item, "vb", None)
    if vb is None:
        return
    r = vb.sceneBoundingRect()
    (xmin, xmax), (ymin, ymax) = vb.viewRange()

    lay = writer.pageLayout()
    res = writer.resolution()
    full = lay.fullRectPixels(res)
    paint = lay.paintRectPixels(res)
    W, H = float(full.width()), float(full.height())
    if W <= 0 or H <= 0:
        return

    def fx(scene_x):
        return (paint.x() + tgt.x() + (scene_x - src.x()) * sc) / W

    def fy(scene_y):
        # TikZ measures from the bottom of the image; Qt from the top.
        return 1.0 - (paint.y() + tgt.y() + (scene_y - src.y()) * sc) / H

    xl, xr = fx(r.left()), fx(r.right())
    yb, yt = fy(r.bottom()), fy(r.top())
    # Points, for the trim correction note.
    wbp, hbp = W * 72.0 / res, H * 72.0 / res

    out = Path(pdf_path).with_suffix(".geom.tex")
    out.write_text(
        f"% TTI-Scope page geometry for {Path(pdf_path).name}\n"
        f"%\n"
        f"% The plot's DATA AREA on the page, as fractions of the full page\n"
        f"% box - which is what \\includegraphics sees. Map a data value u to\n"
        f"% a horizontal fraction with\n"
        f"%     f(u) = XL + (u - Umin)/(Umax - Umin) * (XR - XL)\n"
        f"%\n"
        f"% x axis: {xmin:.6g} .. {xmax:.6g}\n"
        f"% y axis: {ymin:.6g} .. {ymax:.6g}\n"
        f"% page  : {wbp:.2f} x {hbp:.2f} bp\n"
        f"%\n"
        f"% Included with trim=L B R T (in bp), correct with\n"
        f"%     x' = (x*{wbp:.2f} - L) / ({wbp:.2f} - L - R)\n"
        f"%     y' = (y*{hbp:.2f} - B) / ({hbp:.2f} - B - T)\n"
        f"% A bottom-only trim leaves every x fraction unchanged.\n"
        f"%\n"
        f"% \\def, not \\newcommand: two panels of one figure each \\input a\n"
        f"% file like this, and \\newcommand would abort on the second. Copy\n"
        f"% the values into panel-specific names right after \\input, e.g.\n"
        f"%   \\input{{...1a1.geom}} \\let\\WideXL\\PanelXL \\let\\WideXR\\PanelXR\n"
        f"%\n"
        f"\\def\\PanelXL{{{xl:.6f}}}   % page fraction at u = Umin\n"
        f"\\def\\PanelXR{{{xr:.6f}}}   % page fraction at u = Umax\n"
        f"\\def\\PanelUmin{{{xmin:.6g}}}\n"
        f"\\def\\PanelUmax{{{xmax:.6g}}}\n"
        f"\\def\\PanelYB{{{yb:.6f}}}   % plot floor, from the page bottom\n"
        f"\\def\\PanelYT{{{yt:.6f}}}   % plot ceiling\n",
        encoding="utf-8")
    return out


def export_plot_pdf(parent, gfx, item, default_name, title=""):
    """Write a plot to a VECTOR PDF on a white ground.

    PDF rather than PNG because these figures go into LaTeX: a vector page
    scales to any column width without resampling and the text stays
    selectable. QPdfWriter + QGraphicsScene.render() keeps it vector; an image
    exporter would rasterise everything.

    The on-screen theme is dark, and a dark figure printed in a paper is a
    black rectangle. Axes, captions and background are flipped to a light
    palette for the render and restored afterwards - in a finally, so a failed
    render cannot leave the UI white.
    """
    d = Path.home() / "Documents" / "tti-scope"
    d.mkdir(parents=True, exist_ok=True)
    path, _f = QFileDialog.getSaveFileName(
        parent, "Export plot as PDF (vector, white background)",
        str(d / default_name), "PDF document (*.pdf)")
    if not path:
        return None
    if not path.lower().endswith(".pdf"):
        path += ".pdf"

    scene = gfx.scene()
    axes = []
    if item is not None:
        try:
            axes = [item.axes[k]["item"] for k in ("left", "bottom", "top", "right")
                    if k in item.axes]
        except Exception:
            axes = []
    saved = [(a, a.pen(), a.textPen()) for a in axes]
    saved_bg = gfx.backgroundBrush()
    texts = [t for t in scene.items() if isinstance(t, pg.TextItem)]
    saved_tx = [(t, t.color) for t in texts]
    exportables = [i for i in scene.items() if hasattr(i, "setExportMode")]

    # Type sized for a screen is too small once the page is dropped into a
    # two-column paper at a fraction of its width - lane names and tick values
    # are the first things to become unreadable. Everything textual is scaled
    # up for the render and restored afterwards.
    all_axes = [i for i in scene.items() if isinstance(i, pg.AxisItem)]
    legends = [i for i in scene.items() if isinstance(i, pg.LegendItem)]
    # The plot title carries the run identity - pattern, cells, SFN span - so
    # it is the caption a reader needs most, and it was the one piece of text
    # left at screen size and in muted grey while everything around it grew.
    titles = [pi.titleLabel for pi in scene.items()
              if isinstance(pi, pg.PlotItem)
              and getattr(pi, "titleLabel", None) is not None
              and pi.titleLabel.text]
    saved_fonts = []

    def _bigger(f: QFont, k: float) -> QFont:
        g = QFont(f)
        pt = f.pointSizeF()
        if pt > 0:
            g.setPointSizeF(pt * k)
        else:
            g.setPixelSize(max(1, int(round((f.pixelSize() or 11) * k))))
        g.setBold(True)
        return g

    def _fit_axis_to_font(a):
        """Size an axis for the font it now has, from font metrics.

        setWidth(None)/setHeight(None) ask pyqtgraph to auto-size, but its
        auto-size uses `textHeight`, which is only updated when the axis last
        PAINTED. Straight after a font change that value is stale, so the axis
        kept the height it had - 29 px against a 22 pt font - and simply drew
        no tick labels and no axis title at all. Measuring here is
        deterministic and needs no repaint first.
        """
        f = a.style.get("tickFont") or QFont()
        fm = QFontMetrics(f)
        lv = getattr(a, "_tickLevels", None)
        strs = [str(t[1]) for lvl in (lv or []) for t in lvl if str(t[1])]
        pad = 8
        if a.orientation in ("left", "right"):
            if not strs:
                # auto-ticked: size for the widest number the range can show
                lo, hi = a.range
                strs = [f"{v:.0f}" for v in (lo, hi)]
            w = max(fm.horizontalAdvance(x) for x in strs)
            w += a.style["tickTextOffset"][0] + pad
            if a.label.isVisible():
                w += int(a.label.boundingRect().height()) + 4
            a.setWidth(int(w))
        else:
            h = fm.height() + a.style["tickTextOffset"][1] + pad
            if a.label.isVisible():
                h += int(a.label.boundingRect().height()) + 4
            a.setHeight(int(h))

    def _fits(a, k: float) -> float:
        """Shrink the boost to the room an axis actually has.

        A flat multiplier is right for an axis with a handful of ticks and
        wrong for the kernel lanes, where twelve channel names share one short
        column: at 1.9x they overlapped into an unreadable stack. Axes whose
        ticks were set explicitly know exactly how many rows they must fit, so
        the boost is capped to that spacing instead of being applied blind.
        """
        lv = getattr(a, "_tickLevels", None)
        if not lv or a.orientation not in ("left", "right"):
            return k                       # auto ticks thin themselves
        n = sum(len(x) for x in lv)
        if n < 2:
            return k
        room = a.height() / n
        f = a.style.get("tickFont") or QFont()
        cur = f.pointSizeF() * 96.0 / 72.0 if f.pointSizeF() > 0 else \
            (f.pixelSize() or 11)
        if cur <= 0:
            return k
        return max(1.0, min(k, (room * 0.92) / cur))
    try:
        gfx.setBackground("w")
        for a in axes:
            a.setPen(pg.mkPen("#222222")); a.setTextPen(pg.mkPen("#222222"))

        for a in all_axes:
            try:
                tf = a.style.get("tickFont") or QFont()
                lab = dict(getattr(a, "labelStyle", {}) or {})
                # An axis sizes itself once, against the font it had. Enlarging
                # the text without letting it re-measure leaves the tick values
                # and the axis name drawn on top of each other and spilling
                # over the plot - which is what the exported figures did.
                saved_fonts.append(("axisdim", a,
                                    (a.fixedWidth, a.fixedHeight), None))
                saved_fonts.append(("axis", a, tf, lab))
                a.setStyle(tickFont=_bigger(tf, _fits(a, PDF_FONT_BOOST)))
                # The axis NAME is HTML with its own style dict, not the tick
                # font, so it has to be enlarged separately or the channel
                # names stay small while their tick values grow.
                base = 9.0
                try:
                    base = float(str(lab.get("font-size", "9pt")
                                     ).replace("pt", ""))
                except Exception:
                    pass
                a.labelStyle["font-size"] = f"{base * PDF_FONT_BOOST:.0f}pt"
                a.labelStyle["font-weight"] = "bold"
                a._updateLabel()
            except Exception:
                pass
        for tl in titles:
            try:
                saved_fonts.append(("title", tl, dict(tl.opts), None))
                base = 9.0
                try:
                    base = float(str(tl.opts.get("size", "9pt")
                                     ).replace("pt", ""))
                except Exception:
                    pass
                tl.setText(tl.text, color="#222222",
                           size=f"{base * PDF_FONT_BOOST:.0f}pt", bold=True)
            except Exception:
                pass
        for a in all_axes:
            try:
                _fit_axis_to_font(a)
            except Exception:
                pass
        # Several passes. Re-measuring the axes changes the geometry, which
        # re-flows the layout, which can change the axes again; one pass left
        # the second export of a session rendering against a half-settled
        # layout with the tick labels bunched at the origin.
        for _ in range(4):
            try:
                gfx.ci.layout.activate()
            except Exception:
                pass
            QApplication.processEvents()
        for lg in legends:
            try:
                # opts, not the attribute: LegendItem.labelTextSize is a
                # METHOD, so lg.labelTextSize stringifies to a bound-method
                # repr and float() throws - which the guard then swallowed,
                # leaving the legend silently at screen size.
                _sz = str(lg.opts.get("labelTextSize", "9pt"))
                saved_fonts.append(
                    ("legend", lg,
                     (_sz, lg.opts.get("labelTextColor"),
                      lg.opts.get("brush")), None))
                # A legend grows in WIDTH with its font, and it is laid out in
                # one row: at the full boost its three entries were wider than
                # the canvas, which squeezed the rest of the layout until the
                # bottom axis had no height left and drew neither ticks nor
                # title. It does not need the full boost to be legible.
                base = float(_sz.replace("pt", "") or 9)
                lg.setLabelTextSize(f"{base * min(PDF_FONT_BOOST, 1.55):.0f}pt")
                # Legend labels are LabelItems rendering HTML, not TextItems, so
                # the light-text-to-dark pass above never reaches them: on the
                # white export ground they came out near-invisible grey.
                lg.setLabelTextColor("#222222")
                # Opaque, so the legend reads as a panel over the curves rather
                # than as text tangled in them.
                lg.setBrush(pg.mkBrush(255, 255, 255, 235))
                lg.setPen(pg.mkPen("#888888"))
                # setLabelTextSize/Color set the attribute but do not re-render
                # the label, so without this the size change is silently a
                # no-op and the legend stays at screen size.
                for _sample, _lab in lg.items:
                    _lab.setText(_lab.text, **_lab.opts)
                lg.updateSize()
            except Exception:
                pass
        for t in texts:
            try:
                f = t.textItem.font()
                saved_fonts.append(("text", t, f, None))
                t.textItem.setFont(_bigger(f, PDF_FONT_BOOST))
                t.updateTextPos()
            except Exception:
                pass
        for t in texts:
            try:
                if QColor(t.color).lightness() > 140:
                    t.setColor(pg.mkColor("#333333"))
            except Exception:
                pass
        QApplication.processEvents()
        w = QPdfWriter(path)
        w.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        try: w.setPageOrientation(QPageLayout.Orientation.Landscape)
        except Exception: pass
        # Resolution has to match the SCREEN, not be as high as possible.
        # Geometry is in scene pixels and gets scaled by the source->target fit
        # factor; fonts are in POINTS and get resolved against the device DPI
        # as well. At 600 dpi a 9 pt label came out 6.25x larger relative to
        # the bars than it is on screen - axis names sprawling across the page
        # over a correctly-drawn but comparatively tiny chart. Resolving points
        # against the same DPI the view uses makes both scale by one factor, so
        # the page is the view. The output stays vector either way; resolution
        # sets the coordinate system, not the fidelity.
        _scr = QApplication.primaryScreen()
        _dpi = int(round(_scr.logicalDotsPerInch())) if _scr else 96
        w.setResolution(max(72, min(200, _dpi)))
        w.setPageMargins(QMarginsF(8, 8, 8, 8))
        painter = QPainter(w)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # pyqtgraph renders scatter symbols from a pixmap atlas unless the item
        # is told it is being exported - which QGraphicsScene.render() does not
        # do by itself, only pyqtgraph's own exporters do. Without this the
        # TTI-MISS crosses and search carets are embedded as ~18x18 bitmaps and
        # go visibly soft at the 600 dpi this page is written at.
        for i in exportables:
            try:
                i.setExportMode(True, {"painter": painter, "antialias": True})
            except Exception:
                pass
        # WYSIWYG: render the region the view is DISPLAYING, not the scene.
        #
        # QGraphicsScene.sceneRect() is not the visible area. Left unset - as
        # it is here - it returns the union of every item's bounds since the
        # scene was created, and it grows monotonically: it never shrinks when
        # items are removed. Every TTI boundary is a pyqtgraph InfiniteLine,
        # which reports a practically unbounded rect, so after a few windows
        # the scene rect measured 3,510,180 x 5,053 against a 1,044 x 679
        # chart. Fitting that to the page scaled the figure to 1/3362 of its
        # size - a single dot in the corner of an otherwise blank sheet, with
        # only the axis text landing anywhere visible. Resizing the canvas
        # first made it worse, and slow: the render walked a scene millions of
        # pixels wide at 600 dpi.
        src = gfx.mapToScene(gfx.viewport().rect()).boundingRect()
        if src.isEmpty():                       # not laid out yet
            src = scene.itemsBoundingRect()
        if src.isEmpty():
            src = scene.sceneRect()
        pr = painter.viewport()
        sc = min(pr.width()/max(1.0, src.width()), pr.height()/max(1.0, src.height()))
        tw, th = src.width()*sc, src.height()*sc
        tgt = QRectF((pr.width()-tw)/2.0, (pr.height()-th)/2.0, tw, th)
        scene.render(painter, tgt, src)
        # Where the plot's data area landed on the page, written out before the
        # painter closes. Measuring this by eye off a rendered page is what
        # makes a LaTeX overlay miss: the numbers are exact here and nowhere
        # else.
        _write_export_geometry(path, item, src, sc, tgt, w)
        painter.end()
    finally:
        for kind, obj, f, extra in saved_fonts:
            try:
                if kind == "axis":
                    obj.setStyle(tickFont=f)
                    obj.labelStyle = extra
                    obj._updateLabel()
                elif kind == "axisdim":
                    # Only the dimension the orientation owns. A bottom axis
                    # gets its WIDTH from the layout; calling setWidth on it
                    # pins it to the auto value - which collapsed it from 874
                    # to 71 px and left every later export in this session
                    # rendering its ticks bunched at the origin.
                    w, h = f
                    if obj.orientation in ("left", "right"):
                        obj.setWidth(w)
                    else:
                        obj.setHeight(h)
                elif kind == "title":
                    obj.setText(obj.text, **f)
                elif kind == "legend":
                    size, colr, brsh = f
                    obj.setLabelTextSize(size)
                    if colr is not None:
                        obj.setLabelTextColor(colr)
                    if brsh is not None:
                        obj.setBrush(brsh)
                    for _sample, _lab in obj.items:
                        _lab.setText(_lab.text, **_lab.opts)
                    obj.updateSize()
                else:
                    obj.textItem.setFont(f)
                    obj.updateTextPos()
            except Exception:
                pass
        for i in exportables:
            try:
                i.setExportMode(False)
            except Exception:
                pass
        gfx.setBackground(saved_bg)
        for a, pen, tpen in saved:
            a.setPen(pen); a.setTextPen(tpen)
        for t, c in saved_tx:
            try: t.setColor(c)
            except Exception: pass
        QApplication.processEvents()
    geom = Path(path).with_suffix(".geom.tex")
    QMessageBox.information(parent, APP,
        f"Exported\n{path}\n\nVector PDF, white background, A4 fitted."
        + (f"\n\nPage geometry for LaTeX overlays:\n{geom.name}"
           if geom.exists() else "")
        + (f"\n\n{title}" if title else ""))
    return path


class ChartGrip(QFrame):
    """Drag handle that changes the SIZE OF THE CANVAS, not the view on it.

    The distinction matters. Panning a ViewBox changes which slice of the data
    is shown while the drawing surface stays exactly as big as its pane, so a
    tall lane stack stays cramped no matter how much you drag. These grips
    resize the drawing surface itself and let it overflow the pane, which the
    scroll area then lets you move around in.

    axis is "x" (right edge), "y" (bottom edge) or "xy" (corner).
    """
    dragged = pyqtSignal(int, int)
    reset = pyqtSignal()

    CURSOR = {"x": Qt.CursorShape.SizeHorCursor,
              "y": Qt.CursorShape.SizeVerCursor,
              "xy": Qt.CursorShape.SizeFDiagCursor}

    def __init__(self, axis="y"):
        super().__init__()
        self.axis = axis
        self.setCursor(self.CURSOR[axis])
        what = {"x": "wider or narrower", "y": "taller or shorter",
                "xy": "in both directions"}[axis]
        self.setToolTip(f"Drag to make the chart canvas {what}.\n"
                        "Double-click to fit it back to the pane.")
        self._p = None
        self.restyle()

    def restyle(self):
        # 13 px inset by 70 px each side left a ~7 px grabbable sliver in the
        # middle of the edge - the handle existed but was nearly impossible to
        # hit, which reads as "resizing does not work". Wider bar, thinner
        # inset, and a colour that says it is interactive.
        t = _s(22)
        if self.axis == "y":
            self.setFixedHeight(t)
            self.setMinimumWidth(0)
            m = f"margin:{_s(5)}px {_s(12)}px;"
        elif self.axis == "x":
            self.setFixedWidth(t)
            self.setMinimumHeight(0)
            m = f"margin:{_s(12)}px {_s(5)}px;"
        else:
            self.setFixedSize(t, t)
            m = f"margin:{_s(4)}px;"
        self.setStyleSheet(
            f"QFrame{{background:{MUTED};border-radius:{_s(4)}px;{m}}}"
            f"QFrame:hover{{background:{ACCENT};}}"
            f"QFrame:pressed{{background:{TTI_Y};}}")

    def mousePressEvent(self, ev):
        self._p = ev.globalPosition()

    def mouseMoveEvent(self, ev):
        if self._p is None:
            return
        q = ev.globalPosition()
        dx = int(round(q.x() - self._p.x())) if "x" in self.axis else 0
        dy = int(round(q.y() - self._p.y())) if "y" in self.axis else 0
        if dx or dy:
            self.dragged.emit(dx, dy)
            self._p = q

    def mouseReleaseEvent(self, ev):
        self._p = None

    def mouseDoubleClickEvent(self, ev):
        self.reset.emit()


class WallClockAxis(pg.AxisItem):
    """Top ruler showing absolute UTC for a plot whose x data is us-from-
    window-start.

    Both axes are wanted at once: relative microseconds are what you measure a
    kernel against, absolute UTC is what you take to the nvlog, the RU capture
    or the L2 trace. Keeping them on opposite edges of the same linked x range
    avoids making the user choose.
    """

    def __init__(self, **kw):
        super().__init__(orientation="top", **kw)
        self._base_ns = None

    def set_base(self, wall_ns):
        self._base_ns = wall_ns
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self._base_ns is None:
            return ["" for _ in values]
        # spacing is in us; below ~1 ms per tick the millisecond field alone
        # collapses every label to the same string.
        micros = spacing < 1000.0
        return [_fmt_clock(self._base_ns + int(v * 1000), micros)
                for v in values]


class SlotWallClockAxis(pg.AxisItem):
    """Top ruler for a plot whose x data is a SLOT INDEX.

    Slot index is the right x unit for the timeline — it is what the region
    selector, the statistics window and every other tab are addressed in — but
    an index is not a time anyone can take to a log. This maps it back.
    """

    def __init__(self, **kw):
        super().__init__(orientation="top", **kw)
        self._epoch = None
        self._idx = None
        self._t0 = None

    def set_map(self, epoch_ns, idx, t0_ns):
        self._epoch, self._idx, self._t0 = epoch_ns, idx, t0_ns
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self._epoch is None or self._t0 is None or not len(self._t0):
            return ["" for _ in values]
        # Slots are a uniform grid, so index -> time is a lookup, and beyond
        # the ends it extrapolates at the slot period rather than clamping
        # (which would print the same time for every off-scale tick).
        period = ((self._t0[-1] - self._t0[0]) / max(1, len(self._t0) - 1)
                  if len(self._t0) > 1 else 500_000)
        out = []
        for v in values:
            i = int(round(v))
            if 0 <= i < len(self._t0):
                t = int(self._t0[i])
            else:
                t = int(self._t0[0] + i * period)
            micros = spacing * period < 1_000_000
            out.append(_fmt_clock(self._epoch + t, micros))
        return out


def _fmt_ns(ns) -> str:
    if ns is None:
        return "—"
    ns = float(ns)
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns/1e3:.2f} us"
    return f"{ns/1e6:.3f} ms"


# ─────────────────────────────────────────────────────────────────────────────
# Build worker
# ─────────────────────────────────────────────────────────────────────────────

class BuildThread(QThread):
    """Store construction off the UI thread.

    Building a 4 GB capture takes ~60s (and up to several minutes when the
    .nsys-rep still has to be exported), which is far too long to block on.
    """
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, lib: SessionLibrary, entry: SessionEntry, force=False):
        super().__init__()
        self.lib, self.entry, self.force = lib, entry, force

    def run(self):
        try:
            st = self.lib.open(self.entry, log=self.progress.emit,
                               force=self.force)
            self.done.emit(st)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Sessions panel
# ─────────────────────────────────────────────────────────────────────────────

class CpuProfileThread(QThread):
    """Decode a perf.data off the UI thread.

    `perf script` on a 997 Hz capture is 16-100 MB of text and takes seconds.
    Running it inline froze the window every time a session was opened, which
    looked exactly like the app hanging on a big file.
    """
    done = pyqtSignal(object, object)     # (CpuProfile, intra dict)
    failed = pyqtSignal(str)

    def __init__(self, cap, max_slots):
        super().__init__()
        self.cap, self.max_slots = cap, max_slots

    def run(self):
        try:
            prof = build_profile(self.cap, log=lambda m: None)
            intra = None
            if self.cap.tti_capable:
                intra = intra_tti_profile(self.cap, n_bins=25,
                                          max_slots=self.max_slots,
                                          log=lambda m: None)
            self.done.emit(prof, intra)
        except Exception:
            self.failed.emit(traceback.format_exc())


class SessionPanel(QWidget):
    opened = pyqtSignal(object)          # SessionEntry

    def __init__(self, lib: SessionLibrary):
        super().__init__()
        self.lib = lib
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        v.addWidget(_lbl("SESSIONS", 10, MUTED, True))

        self.list = QListWidget()
        self._buttons = []
        self.list.setStyleSheet(_style_list())
        self.list.itemDoubleClicked.connect(self._open_selected)
        v.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for text, fn in (("Add folder…", self.add_folder),
                         ("Remove", self.remove_selected),
                         ("Open", self._open_selected),
                         ("Rebuild", lambda: self._open_selected(force=True))):
            b = QPushButton(text)
            b.setStyleSheet(_style_button())
            b.clicked.connect(fn)
            self._buttons.append(b)
            row.addWidget(b)
        v.addLayout(row)

        self.detail = _lbl("", 10, MUTED)
        self.detail.setWordWrap(True)
        v.addWidget(self.detail)
        self.list.currentRowChanged.connect(self._show_detail)

    def restyle(self):
        self.list.setStyleSheet(_style_list())
        for b in self._buttons:
            b.setStyleSheet(_style_button())

    def refresh(self):
        self.list.clear()
        for e in self.lib.entries:
            it = QListWidgetItem(f"{e.label}    ·  {e.state}")
            it.setData(Qt.ItemDataRole.UserRole, e)
            if not e.built:
                it.setForeground(_q(MUTED))
            self.list.addItem(it)

    def _show_detail(self, row):
        e = self.current()
        if not e:
            self.detail.setText("")
            return
        txt = f"{e.path.name}\n{e.sources}\n{e.src_bytes/1e9:.1f} GB of sources"
        if e.note:
            txt += f"\n\n{e.note}"
        self.detail.setText(txt)

    def current(self):
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _open_selected(self, *_a, force=False):
        e = self.current()
        if e:
            e._force = force
            self.opened.emit(e)

    def remove_selected(self, *_a):
        """Remove the selected sessions from the library index.

        The capture directories on disk are NOT touched - this only forgets
        them, and Add folder brings them back. Multi-select is enabled, and
        Delete/Backspace does the same thing.
        """
        rows = sorted({i.row() for i in self.list.selectedIndexes()})
        if not rows:
            QMessageBox.information(self, APP, "Select one or more sessions to remove.")
            return
        picked = [self.lib.entries[r] for r in rows if 0 <= r < len(self.lib.entries)]
        if not picked:
            return
        names = "\n  ".join(e.label for e in picked[:12])
        more = "" if len(picked) <= 12 else f"\n  ... and {len(picked) - 12} more"
        if QMessageBox.question(
                self, APP,
                f"Remove {len(picked)} session(s) from the library?\n\n  {names}{more}"
                "\n\nThe capture folders on disk are left untouched.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        n = self.lib.remove(picked)
        self.lib.save()
        self.refresh()
        self.detail.setText(f"removed {n} session(s) from the library")

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.list.hasFocus():
            self.remove_selected()
            return
        super().keyPressEvent(ev)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Add capture folder")
        if not d:
            return
        n = len(self.lib.entries)
        self.lib.scan(d)
        self.lib.save()
        self.refresh()
        if len(self.lib.entries) == n:
            QMessageBox.information(
                self, APP,
                "No captures found there.\n\nA capture folder holds at least "
                "one of:\n  cuphy_<N>C_<pat>.sqlite  (nsys export)\n"
                "  kernel_printf_<N>C_<pat>.log  (AERIAL_KTRACE)\n"
                "  cuphy_<N>C_<pat>.log  (cuPHY nvlog)")


# ─────────────────────────────────────────────────────────────────────────────
# Global timeline
# ─────────────────────────────────────────────────────────────────────────────

class GlobalTimeline(QWidget):
    """Whole-capture GPU utilisation, one point per TTI.

    Reads slot_stats — 43,492 precomputed rows for the 20-cell capture — so
    this renders instantly regardless of how many kernel launches are behind
    it. The draggable region is the window every other tab reports on.
    """
    window_changed = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        head = QHBoxLayout()
        head.addWidget(_lbl("GPU utilisation per TTI", 12, FG, True))
        head.addStretch(1)
        self.readout = _lbl("", 11, MUTED)
        head.addWidget(self.readout)
        v.addLayout(head)

        self.ax_wall = SlotWallClockAxis()
        self.plot = pg.PlotWidget(axisItems={"top": self.ax_wall})
        self.plot.setLabel("left", "% of slot")
        self.plot.setLabel("bottom", "TTI (slot index)")
        self.plot.showAxis("top")
        self.plot.setLabel("top", "UTC wall clock", color=MUTED, size="8pt")
        self.plot.showGrid(x=True, y=True, alpha=0.16)
        self.plot.addLegend(offset=(-10, 10), labelTextSize="9pt")
        self.plot.setMouseEnabled(x=True, y=False)
        v.addWidget(self.plot, 3)

        # Bars, not curves. At one bar per TTI the quantity is a per-slot
        # measurement, not a continuous signal, and a line implies
        # interpolation between slots that does not exist.
        self.b_busy = pg.BarGraphItem(x0=[], x1=[], y0=0, height=[],
                                      brush=_q("#1D9E75", 200),
                                      pen=pg.mkPen(None))
        self.b_dem = pg.BarGraphItem(x0=[], x1=[], y0=0, height=[],
                                     brush=_q("#D4537E", 170),
                                     pen=pg.mkPen(None))
        self.plot.addItem(self.b_busy)
        self.plot.addItem(self.b_dem)
        # Legend entries only — BarGraphItem carries no legend sample.
        self.plot.plot([], [], pen=pg.mkPen("#1D9E75", width=6),
                       name="GPU busy (union of kernels)")
        self.plot.plot([], [], pen=pg.mkPen("#D4537E", width=6),
                       name="Warp demand (100% = machine full)")
        self.cap_line = pg.InfiniteLine(pos=100, angle=0,
                                        pen=pg.mkPen(MUTED, style=Qt.PenStyle.DashLine))
        self.plot.addItem(self.cap_line)

        self.region = pg.LinearRegionItem(
            brush=_q(SELECT, 34),
            pen=pg.mkPen(SELECT, width=3),
            hoverPen=pg.mkPen(SELECT_HI, width=4))
        for ln in self.region.lines:
            ln.setPen(pg.mkPen(SELECT, width=3))
            ln.setHoverPen(pg.mkPen(SELECT_HI, width=4))
        self.region.setZValue(10)
        self.region.sigRegionChangeFinished.connect(self._emit)
        self.plot.addItem(self.region)

        self.count = pg.PlotWidget()
        self.count.setLabel("left", "launches")
        self.count.setMaximumHeight(_s(110))
        self.count.showGrid(x=True, y=True, alpha=0.16)
        self.count.setMouseEnabled(x=True, y=False)
        self.count.setXLink(self.plot)
        self.b_nsys = pg.BarGraphItem(x0=[], x1=[], y0=0, height=[],
                                      brush=_q("#4FA3E8", 190),
                                      pen=pg.mkPen(None))
        self.b_ktr = pg.BarGraphItem(x0=[], x1=[], y0=0, height=[],
                                     brush=_q("#E07A5F", 190),
                                     pen=pg.mkPen(None))
        self.count.addItem(self.b_nsys)
        self.count.addItem(self.b_ktr)
        # The per-TTI launch-count plot is not shown. It is still built so
        # _redraw_bars and any external reference keep working; it is simply
        # never added to the layout, leaving "% of slot" as the only plot.
        self.SHOW_LAUNCH_COUNT = False
        if self.SHOW_LAUNCH_COUNT:
            v.addWidget(self.count, 1)

        self.store = None
        self._series = None
        self.plot.sigRangeChanged.connect(lambda *_a: self._redraw_bars())

    # -- bars -----------------------------------------------------------
    #
    # 43,492 slots is far more bars than pixels. Rather than hand pyqtgraph
    # every slot and let it draw sub-pixel rectangles, only the visible range
    # is drawn, and when that range holds more slots than the widget has room
    # for, adjacent slots are aggregated into buckets. Aggregation takes the
    # MAXIMUM, not the mean: a timeline is for finding the slot that broke, and
    # averaging hides exactly that.
    MAX_BARS = 1200

    def _redraw_bars(self):
        if self._series is None:
            return
        idx, busy, dem, nk, ktr = self._series
        n = len(idx)
        (x0, x1), _y = self.plot.viewRange()
        lo = max(0, int(np.floor(x0)))
        hi = min(n - 1, int(np.ceil(x1)))
        if hi <= lo:
            return
        vis = hi - lo + 1
        step = max(1, int(np.ceil(vis / self.MAX_BARS)))
        sl = slice(lo, hi + 1)
        if step == 1:
            xs = idx[sl].astype(np.float64)
            b, d, a, k = busy[sl], dem[sl], nk[sl], ktr[sl]
        else:
            m = (vis // step) * step
            def agg(v):
                return v[sl][:m].reshape(-1, step).max(axis=1)
            xs = idx[sl][:m].reshape(-1, step)[:, 0].astype(np.float64)
            b, d, a, k = agg(busy), agg(dem), agg(nk), agg(ktr)
        w = float(step)
        self.b_busy.setOpts(x0=xs, x1=xs + w, height=b, y0=0)
        self.b_dem.setOpts(x0=xs + w * 0.28, x1=xs + w * 0.72, height=d, y0=0)
        self.b_nsys.setOpts(x0=xs, x1=xs + w, height=a, y0=0)
        self.b_ktr.setOpts(x0=xs + w * 0.28, x1=xs + w * 0.72, height=k, y0=0)
        self._agg_step = step

    def set_store(self, st: SessionStore):
        self.store = st
        n = st.n_slots
        if not n:
            return
        rows = st.slot_stats(0, n - 1)
        idx = np.array([r[0] for r in rows], dtype=np.int64)
        dur = np.maximum(1, np.array([r[2] - r[1] for r in rows], dtype=np.float64))
        span = np.array([r[7] for r in rows], dtype=np.float64)
        warp = np.array([r[8] for r in rows], dtype=np.float64)
        nk = np.array([r[5] for r in rows], dtype=np.float64)
        ktr = np.array([r[9] for r in rows], dtype=np.float64)
        self._series = (idx, 100.0 * span / dur,
                        100.0 * warp / (st.max_warps * dur), nk, ktr)

        t0 = np.array([r[1] for r in rows], dtype=np.int64)
        self.ax_wall.set_map(st.utc_epoch_ns, idx, t0)

        # Open on the middle of the capture: the head and tail of every sweep
        # are ramp-up and tear-down, not steady-state traffic.
        busy_i = np.nonzero(nk > 0)[0]
        mid = int(idx[busy_i[len(busy_i) // 2]]) if len(busy_i) else n // 2
        # ~100 TTI (50 ms) is the widest span at which the per-slot structure
        # is still individually readable. The user can zoom out from here.
        self.plot.setXRange(max(0, mid - 50), min(n, mid + 50), padding=0)
        self.region.setRegion((mid - 2, mid + 2))
        self._redraw_bars()
        self._emit()

    def set_window(self, lo, hi):
        self.region.setRegion((lo, hi))

    def _emit(self):
        if not self.store:
            return
        a, b = self.region.getRegion()
        lo = max(0, int(round(a)))
        hi = min(self.store.n_slots - 1, max(lo, int(round(b))))
        s = self.store.slot(lo)
        w = self.store.wall_ns(s[1]) if s else None
        self.readout.setText(
            (f"{_fmt_wall(w)} UTC   ·   " if w else "") +
            f"{self.store.slot_label(lo)} → {self.store.slot_label(hi)}   "
            f"({hi-lo+1} TTI, {(hi-lo+1)*self.store.slot_dur_ns/1e6:.2f} ms)")
        self.window_changed.emit(lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# TTI detail
# ─────────────────────────────────────────────────────────────────────────────

class TTIView(QWidget):
    """One window in three linked lanes: what cuPHY said it was doing (NVTX),
    what the GPU actually ran (kernels), and what the host pipeline was doing
    at the same moment ({TI} task records)."""

    # Asks the window owner to move to [lo,hi]. The view does not set its own
    # window: the timeline still owns the selection, so a jump made here and a
    # jump made anywhere else go through one path and cannot disagree.
    jump_requested = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        v.addWidget(self._build_window_pane())

        bar = QHBoxLayout()
        bar.addWidget(_lbl("Lanes", 12, LABEL, bold=True))
        self.lane_mode = QComboBox()
        self.lane_mode.addItems(["by channel", "by CUDA stream"])
        self.lane_mode.setStyleSheet(_style_combo())
        self.lane_mode.currentIndexChanged.connect(lambda _i: self.redraw())
        self._combo = self.lane_mode
        bar.addWidget(self.lane_mode)
        self.show_ktrace = QCheckBox("KTRACE (device-launched)")
        self.show_ktrace.setChecked(True)
        self.show_ktrace.setStyleSheet(
            f"QCheckBox{{color:{FG};font-size:{_s(12)}px;}}"
            f"QCheckBox::indicator{{width:{_s(14)}px;height:{_s(14)}px;"
            f"border:1px solid {EDGE};border-radius:{_s(3)}px;"
            f"background:{BG3};}}"
            f"QCheckBox::indicator:checked{{background:{ACCENT};"
            f"border-color:{ACCENT};}}")
        self._chk = self.show_ktrace
        self.show_ktrace.stateChanged.connect(lambda _s: self.redraw())
        bar.addWidget(self.show_ktrace)

        # -- phantom stream lane (beta) -------------------------------------
        # Off by default: it is a MODEL laid over measured lanes, and it should
        # never appear in a figure unless the reader asked for it.
        self.show_phantom = QCheckBox("Phantom stream \u03b2")
        self.show_phantom.setChecked(False)
        self.show_phantom.setStyleSheet(self.show_ktrace.styleSheet())
        self.show_phantom.setToolTip(
            "Draw a hypothetical co-tenant lane: at each instant, a kernel "
            "shaped to\nroofline minus what Aerial holds. Each bar ends where "
            "Aerial reclaims the\nspace, because a resident CUDA block cannot "
            "be resized or evicted.\n\nThis is a feasibility ORACLE, not a "
            "schedule. Aerial's demand respects each cuPHY MPS context's SM\n"
            "cap; DCGM occupancy is shown as a check. Off by default.")
        self.show_phantom.stateChanged.connect(lambda _s: self.redraw())
        bar.addWidget(self.show_phantom)

        self.phantom_mode = QComboBox()
        self.phantom_mode.addItems(["oracle (exact)",
                                    "greedy fill x6 (heuristic)"])
        self.phantom_mode.setStyleSheet(_style_combo())
        self.phantom_mode.setToolTip(
            "oracle - the OPTIMAL single co-tenant schedule: non-overlapping\n"
            "         kernels maximising captured warp-time, with a relaunch\n"
            "         cost per kernel, solved exactly by dynamic programming.\n"
            "         Each kernel's size is the largest that fits the spare\n"
            "         for its WHOLE life. Default.\n"
            "greedy - up to 6 phantoms launched at the first instant they\n"
            "         fit. A heuristic: no bar is a bound on anything.")
        self.phantom_mode.currentIndexChanged.connect(lambda _i: self.redraw())
        bar.addWidget(self.phantom_mode)

        # The co-tenant kernel the oracle places. S and R are what it needs;
        # B blank or 'auto' lets the oracle pick the block size per kernel.
        self.cotenant_edit = QLineEdit()
        self.cotenant_edit.setStyleSheet(_style_edit())
        self.cotenant_edit.setMaximumWidth(_s(130))
        self.cotenant_edit.setPlaceholderText("B,S,R  auto,0,16")
        self.cotenant_edit.setToolTip(
            "Co-tenant kernel shape for the oracle: threads/block, shared\n"
            "bytes/block, registers/thread - what YOUR kernel needs.\n"
            "B 'auto' searches 32..1024 per phantom. Relaunch cost and D_min\n"
            "come from TTISCOPE_LAUNCH_GAP_US / TTISCOPE_DMIN_US.")
        self.cotenant_edit.editingFinished.connect(self._cotenant_changed)
        bar.addWidget(self.cotenant_edit)

        # The DCGM ratio is a validation of the demand model, not part of it.
        # Applying it is a choice, made visibly, and off by default.
        self.chk_scale = QCheckBox("scale to DCGM")
        self.chk_scale.setChecked(False)
        self.chk_scale.setStyleSheet(self.show_ktrace.styleSheet())
        self.chk_scale.setToolTip(
            "Multiply Aerial's modelled demand by DCGM sm_occupancy / model\n"
            "occupancy over the same interval. The model is MPS-cap aware and\n"
            "lands within ~1.3x of DCGM on its own; the residue is wave tails\n"
            "launch geometry cannot see. Scaling fixes the warp level and is\n"
            "applied to all four resources alike.")
        self.chk_scale.stateChanged.connect(lambda _s: self._scale_changed())
        bar.addWidget(self.chk_scale)

        self.btn_spare = QPushButton("Spare GPU \u25b8")
        self.btn_spare.setStyleSheet(_style_button())
        self.btn_spare.setToolTip(
            "Quantify the spare GPU in this window in the BenchmarkingApp's "
            "seven\ndimensions [B,S,R,D,n,P,M], with the capacity-duration "
            "frontier.")
        self.btn_spare.clicked.connect(self.show_spare)
        bar.addWidget(self.btn_spare)

        # -- find a kernel by name and jump to its next occurrence ----------
        bar.addSpacing(_s(14))
        bar.addWidget(_lbl("Find kernel", 12, LABEL, bold=True))
        self.find_box = QComboBox()
        self.find_box.setEditable(True)
        self.find_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.find_box.setMinimumWidth(_s(170))
        self.find_box.setStyleSheet(_style_combo())
        self.find_box.lineEdit().setPlaceholderText("kernel name…")
        bar.addWidget(self.find_box)
        self.find_box.lineEdit().returnPressed.connect(self.find_next_kernel)
        self.btn_find = QPushButton("Next ▸")
        self.btn_find.setStyleSheet(_style_button())
        self.btn_find.clicked.connect(self.find_next_kernel)
        bar.addWidget(self.btn_find)
        self.find_hit = _lbl("", 12, LABEL)
        bar.addWidget(self.find_hit)

        bar.addStretch(1)

        # Second row. One row of eleven controls only fitted on a wide window;
        # in a scrolling bar the ones past the edge were invisible, which is
        # how the hand tool ended up unreachable.
        bar2 = QHBoxLayout()
        bar2.setContentsMargins(0, 0, 0, 0)
        self.btn_hand = QPushButton("✋ Hand — drag chart")
        self.btn_hand.setCheckable(True)
        self.btn_hand.setStyleSheet(_style_button())
        self.btn_hand.setToolTip(
            "Hand tool — grab the chart and drag it in BOTH axes; wheel zooms.\n"
            "Off, the view is locked to the lane stack and drags in time only.")
        self.btn_hand.toggled.connect(self.set_pan_mode)
        bar2.addWidget(self.btn_hand)

        self.btn_details = QPushButton("Details")
        self.btn_details.setCheckable(True)
        self.btn_details.setChecked(True)
        self.btn_details.setStyleSheet(_style_button())
        self.btn_details.setToolTip("Show or hide the kernel detail panel")
        self.btn_details.clicked.connect(self.toggle_info)
        bar2.addWidget(self.btn_details)

        self.btn_layout = QPushButton("Stack")
        self.btn_layout.setStyleSheet(_style_button())
        self.btn_layout.setToolTip("Move the detail panel below the chart, so "
                                   "the splitter drags vertically instead of "
                                   "horizontally")
        self.btn_layout.clicked.connect(self.toggle_layout)
        bar2.addWidget(self.btn_layout)
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setStyleSheet(_style_button())
        self.btn_fit.setToolTip("Rescale so every item is inside the view, and "
                                "refit the chart to the pane")
        self.btn_fit.clicked.connect(self.fit_view)
        bar2.addWidget(self.btn_fit)

        bar2.addSpacing(_s(10))
        bar2.addWidget(_lbl("Canvas", 11, MUTED))
        self.btn_canvas_out = QPushButton("\u2212")
        self.btn_canvas_out.setStyleSheet(_style_button())
        self.btn_canvas_out.setToolTip("Shrink the chart canvas by 20%")
        self.btn_canvas_out.clicked.connect(lambda: self.zoom_canvas(1 / 1.25))
        bar2.addWidget(self.btn_canvas_out)
        self.btn_canvas_in = QPushButton("+")
        self.btn_canvas_in.setStyleSheet(_style_button())
        self.btn_canvas_in.setToolTip(
            "Grow the chart canvas by 25% in both axes. Past the pane it "
            "scrolls,\nand the hand tool drags it around.")
        self.btn_canvas_in.clicked.connect(lambda: self.zoom_canvas(1.25))
        bar2.addWidget(self.btn_canvas_in)

        # Height alone. With a dozen lanes the useful move is almost always
        # "make the lanes taller at this exact time window", which the
        # both-axes zoom cannot express: it widens the canvas too and throws
        # away the span you had framed.
        bar2.addSpacing(_s(6))
        bar2.addWidget(_lbl("Height", 11, MUTED))
        self.btn_hgt_out = QPushButton("\u2195\u2212")
        self.btn_hgt_out.setStyleSheet(_style_button())
        self.btn_hgt_out.setToolTip(
            "Shorten the chart by 20%. Height only - the time window and the "
            "canvas width\nare left exactly as they are.  Ctrl+Shift+Up")
        self.btn_hgt_out.clicked.connect(lambda: self.resize_canvas_h(1 / 1.25))
        bar2.addWidget(self.btn_hgt_out)
        self.btn_hgt_in = QPushButton("\u2195+")
        self.btn_hgt_in.setStyleSheet(_style_button())
        self.btn_hgt_in.setToolTip(
            "Make the chart 25% taller so each lane gets more room. Height "
            "only - the\ntime window is unchanged; past the pane it scrolls."
            "  Ctrl+Shift+Down")
        self.btn_hgt_in.clicked.connect(lambda: self.resize_canvas_h(1.25))
        bar2.addWidget(self.btn_hgt_in)
        self.canvas_lbl = _lbl("", 11, MUTED)
        bar2.addWidget(self.canvas_lbl)
        bar2.addSpacing(_s(10))

        self.btn_pdf = QPushButton("PDF")
        self.btn_pdf.setStyleSheet(_style_button())
        self.btn_pdf.setToolTip("Export these lanes as a vector PDF (white, for LaTeX)")
        self.btn_pdf.clicked.connect(self.export_pdf)
        bar2.addWidget(self.btn_pdf)

        bar2.addSpacing(_s(10))
        self.hdr = _lbl("", 11, MUTED)
        # Ignored horizontally: the header is a status line, and its natural
        # width should never be a constraint on the window's.
        self.hdr.setSizePolicy(QSizePolicy.Policy.Ignored,
                               QSizePolicy.Policy.Preferred)
        bar2.addWidget(self.hdr, 1)

        barv = QVBoxLayout()
        barv.setContentsMargins(0, 0, 0, 0)
        barv.setSpacing(_s(3))
        barv.addLayout(bar)
        barv.addLayout(bar2)
        barw = QWidget()
        barw.setLayout(barv)
        self.bar_scroll = QScrollArea()
        self.bar_scroll.setWidget(barw)
        self.bar_scroll.setWidgetResizable(True)
        self.bar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.bar_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.bar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.bar_scroll.setFixedHeight(_s(66))
        v.addWidget(self.bar_scroll)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.gl = pg.GraphicsLayoutWidget()
        # The plot sits in a scroll area so the grip can make it TALLER than
        # the pane. With a dozen lanes the only way to give each one readable
        # height is to let the chart overflow and scroll.
        self.chart_scroll = QScrollArea()
        self.chart_scroll.setWidget(self.gl)
        # widgetResizable pins the canvas to the viewport, which is why the
        # canvas could never grow in x. It is turned back on only by "Fit".
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.chart_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chart_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # The hand tool drags the viewport over an oversized canvas, so the
        # events have to be caught before the QGraphicsView consumes them.
        self.gl.viewport().installEventFilter(self)
        self.gl.installEventFilter(self)

        chart_pane = QWidget()
        cv = QVBoxLayout(chart_pane)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)
        top.addWidget(self.chart_scroll, 1)
        self.grip_x = ChartGrip("x")
        top.addWidget(self.grip_x)
        cv.addLayout(top, 1)
        bot = QHBoxLayout()
        bot.setContentsMargins(0, 0, 0, 0)
        bot.setSpacing(0)
        self.grip_y = ChartGrip("y")
        bot.addWidget(self.grip_y, 1)
        self.grip_xy = ChartGrip("xy")
        bot.addWidget(self.grip_xy)
        cv.addLayout(bot)
        for g in (self.grip_x, self.grip_y, self.grip_xy):
            g.dragged.connect(self._grip_drag)
            g.reset.connect(self._grip_reset)
        self.grip = self.grip_y          # kept: restyle() and tests use it
        split.addWidget(chart_pane)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.info.setStyleSheet(_style_text())
        # The detail panel is dismissable. On a laptop screen it costs a third
        # of the width, and most of the time the lanes are what you want; its
        # 400px floor also stopped the whole window from shrinking below it.
        self.info_box = QWidget()
        iv = QVBoxLayout(self.info_box)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(_s(2))
        ih = QHBoxLayout()
        ih.setContentsMargins(0, 0, 0, 0)
        self.info_title = _lbl("Kernel detail", 11, MUTED)
        ih.addWidget(self.info_title)
        ih.addStretch(1)
        self.btn_info_close = QPushButton("✕")
        self.btn_info_close.setFixedSize(_s(22), _s(20))
        self.btn_info_close.setStyleSheet(_style_button())
        self.btn_info_close.setToolTip(
            "Close this panel. It reopens when you click a kernel.")
        self.btn_info_close.clicked.connect(self.hide_info)
        ih.addWidget(self.btn_info_close)
        iv.addLayout(ih)
        iv.addWidget(self.info, 1)
        self.info_box.setMinimumWidth(_s(220))
        split.addWidget(self.info_box)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([_s(1100), _s(420)])
        # A 1px handle is technically draggable and practically invisible.
        split.setHandleWidth(_s(8))
        split.setChildrenCollapsible(False)
        split.setStyleSheet(f"QSplitter::handle{{background:{GRID};}}"
                            f"QSplitter::handle:hover{{background:{ACCENT};}}")
        self.split = split
        v.addWidget(split, 1)

        # The TTI tab shows ONLY the GPU-kernel lanes (by channel / by CUDA
        # stream). The cuPHY-phase and host-task plots are still constructed so
        # every reference below keeps working, but they are NOT added to the
        # layout and their draw paths return immediately - see SHOW_PHASE_TASK.
        self.ax_wall = WallClockAxis()
        self.p_kern = self.gl.addPlot(row=0, col=0,
                                      axisItems={"top": self.ax_wall})
        self.p_kern.showAxis("top")
        self.ax_wall.setHeight(_s(30))
        self.p_phase = pg.PlotItem()          # detached, never shown
        self.p_task = pg.PlotItem()           # detached, never shown
        for p, title in ((self.p_phase, "cuPHY phases (NVTX)"),
                         (self.p_kern, "GPU kernels"),
                         (self.p_task, "Host task pipeline (nvlog {TI})")):
            p.setTitle(title, color=MUTED, size="9pt")
            p.showGrid(x=True, y=False, alpha=0.14)
            p.setMouseEnabled(y=False)
            p.getAxis("left").setWidth(_s(170))
            p.getAxis("left").setStyle(tickFont=_tick_font())
        # (no setXLink: p_phase / p_task are detached)
        # Rows 1 and 2 would hold the phase and task plots. With those detached,
        # any stretch left on them gives the kernel plot only part of the
        # canvas: dead space under the lanes on screen, a blank lower half in
        # every exported figure, and channel names squeezed too tight to be
        # enlarged for print. So the kernel plot takes all of it.
        self.gl.ci.layout.setRowStretchFactor(0, 1)
        self.gl.ci.layout.setRowStretchFactor(1, 0)
        self.gl.ci.layout.setRowStretchFactor(2, 0)
        self.p_kern.setLabel("bottom", "µs within window")

        self._show_canvas_size()
        self.store = None
        self.lo = self.hi = 0
        self._draw_lo = self._draw_hi = 0
        self.clipped = 0
        self._bars = []
        self.p_kern.scene().sigMouseClicked.connect(self._on_click)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._show_canvas_size()

    def restyle(self):
        self.lane_mode.setStyleSheet(_style_combo())
        self.show_ktrace.setStyleSheet(f"color:{FG};font-size:{_s(11)}px;")
        self.info.setStyleSheet(_style_text())
        self.info_box.setMinimumWidth(_s(220))
        self.btn_info_close.setFixedSize(_s(22), _s(20))
        for pl in (self.p_phase, self.p_kern, self.p_task):
            pl.getAxis("left").setWidth(_s(170))
        self.ax_wall.setHeight(_s(30))
        self.bar_scroll.setFixedHeight(_s(66))
        for g in (self.grip_x, self.grip_y, self.grip_xy):
            g.restyle()
        self.find_box.setMinimumWidth(_s(170))
        self.redraw()

    def set_store(self, st):
        self.store = st
        self._populate_find(st)
        # The legend describes the tracers this capture actually used. It used
        # to describe both unconditionally, so an nsys-only run was annotated
        # with KTRACE semantics it had never exercised.
        srcs = st.capture_sources() if st else []
        has_kt = "KTRACE" in srcs
        self.show_ktrace.setVisible(has_kt)
        L = ["Click a kernel bar for its full record.", "",
             "Captured by:  " + (" + ".join(srcs) if srcs else "nothing"), ""]
        if "nsys" in srcs:
            L.append("Solid bars   nsys (CUPTI), measured start and end.")
        if has_kt:
            L += ["Dashed bars  AERIAL_KTRACE, device-launched. Start is",
                  "             cross-calibrated; duration is the per-kernel",
                  "             median, not this launch's own.",
                  "Diamonds     KTRACE launch with no instrumented end —",
                  "             only the launch instant is known."]
        L += ["Amber edge   overlaps another kernel on this lane; click the",
              "             stack to list every kernel under the cursor.",
              "Red ✕        a TTI the DU missed, from the cuphy log."]
        if "nsys" in srcs and not has_kt:
            L += ["",
                  "No KTRACE in this capture. Every kernel below reached it",
                  "through nsys — the PUSCH pipeline included, which the",
                  "stock config hides from CUPTI by launching its graph from",
                  "device code. Seeing it here means this run was taken with",
                  "pusch_workCancelMode: 0 and pusch_deviceGraphLaunchEn: 0."]
        self.info.setPlainText("\n".join(L))
        self.redraw()

    # A Gantt is a detail view. Drawing 1,000 TTI took 2.3 s and scaled
    # linearly - 5,000 slots froze the window for eleven seconds - and the
    # result is unreadable anyway: at 1,000 slots each one is sub-pixel. Draw a
    # readable prefix and say so, rather than blocking to render something
    # nobody can look at.
    MAX_DRAW_SLOTS = 48

    # Search cursor: where the last "Next" landed, so repeated presses walk
    # forward through occurrences instead of re-finding the same one.
    _find_cursor = -1
    _find_name = ""

    # Set by TTICompare when a secondary viewer is present; ignored otherwise.
    # Deliberately a plain callable, not a pyqtSignal - GlobalTimeline already
    # owns a signal of that name and two meanings for one identifier is how
    # this gets miswired later.
    on_window_changed = None

    def set_window(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.redraw()
        if self.on_window_changed:
            try:
                self.on_window_changed(lo, hi)
            except Exception:
                pass

    # -- drawing ------------------------------------------------------------
    def redraw(self):
        st = self.store
        for p in (self.p_phase, self.p_kern, self.p_task):
            p.clear()
        self._bars = []
        self._phantoms = []
        self._phantom_note = ""
        if not st or not st.n_slots:
            return
        lo, hi = self.lo, self.hi
        self.clipped = 0
        if hi - lo + 1 > self.MAX_DRAW_SLOTS:
            self.clipped = hi - lo + 1
            hi = lo + self.MAX_DRAW_SLOTS - 1
        s_lo, s_hi = st.slot(lo), st.slot(hi)
        if not s_lo or not s_hi:
            return
        t0, t1 = s_lo[1], s_hi[2]
        us = lambda t: (t - t0) / 1000.0
        w0 = st.wall_ns(t0)
        self.ax_wall.set_base(w0)
        self.p_phase.setLabel(
            "top", "UTC wall clock" if w0 else
            "no wall-clock anchor in this capture (nsys export missing)",
            color=MUTED, size="8pt")

        # slot boundaries, drawn on every lane so a bar can be read against the
        # TTI it belongs to
        # A caption on every boundary is a wall of near-identical numbers, and
        # thinning it by "whatever fits" still left a dozen of them. The slot
        # grid is regular and labelled at both ends, so three anchors are
        # enough to count from - plus the slots that actually matter, the ones
        # the DU dropped.
        _idx = st.tti_miss_slot_idx() if hasattr(st, "tti_miss_slot_idx") else {}
        caption_at = {lo, (lo + hi) // 2, hi}
        caption_at |= {i for i in _idx if lo <= i <= hi}
        for i in range(lo, hi + 1):
            r = st.slot(i)
            if not r:
                continue
            for p in (self.p_phase, self.p_task):
                p.addItem(pg.InfiniteLine(
                    pos=us(r[1]), angle=90,
                    pen=pg.mkPen(TTI_Y, width=TTI_W, style=Qt.PenStyle.DashLine)))
            # The caption is attached to the boundary line rather than placed
            # at a fixed y above the top lane. An InfLineLabel keeps itself
            # inside the view box and flips side near an edge; a TextItem at
            # ymax + k sits wherever that lands, which is off the top of the
            # chart as soon as the lane count or the zoom changes.
            lab = st.slot_label(i) if i in caption_at else None
            dropped = i in _idx
            # The caption is the SFN.slot the DU itself used, read from the
            # nvlog - not a tile index. No fill: a filled block of colour on
            # the kernel lane reads as another bar, one more thing that was
            # launched, which is exactly what this pane is otherwise full of.
            ln = pg.InfiniteLine(
                pos=us(r[1]), angle=90,
                pen=pg.mkPen(TTI_Y, width=TTI_W, style=Qt.PenStyle.DashLine),
                label=lab,
                labelOpts=({"position": 0.985,
                            "color": SFN_MISS if dropped else SFN_OK,
                            "anchors": [(0, 0), (1, 0)]}
                           if lab else None))
            self.p_kern.addItem(ln)
            if lab and dropped:
                # Bold as well as red, so a dropped slot still reads as
                # different in kind once the figure is printed in greyscale.
                try:
                    f = ln.label.textItem.font()
                    f.setBold(True)
                    ln.label.textItem.setFont(f)
                    ln.label.updateTextPos()
                except Exception:
                    pass

        self._draw_lo, self._draw_hi = lo, hi
        self._draw_phases(st, t0, t1, us)
        n_ev, ymax = self._draw_kernels(st, t0, t1, us)
        ymax = self._draw_phantom(st, t0, t1, us, ymax)
        self._n_miss = self._mark_misses(st, t0, t1, us, ymax if isinstance(ymax, (int, float)) else 0)
        self._draw_tasks(st, us)

        self._mark_overlaps()
        # The SFN.slot captions sit ABOVE the top lane, so the default Y range
        # spans only the lanes and clipped them off the top. Reserve headroom
        # explicitly, plus a little under the bottom lane.
        self.p_kern.setYRange(-0.7, ymax + TOP_PAD, padding=0)
        self.p_kern.setXRange(0, us(t1), padding=0.01)
        cov = st.clock_iqr_ns
        w1 = st.wall_ns(t1)
        when = (f"{_fmt_wall(w0)} → {_fmt_clock(w1)} UTC   ·   "
                if w0 else "")
        clip = ("" if not self.clipped else
                f"   ·   showing the first {self.MAX_DRAW_SLOTS} of "
                f"{self.clipped:,} selected TTI (detail view)")
        # The cross-calibration residual describes how well KTRACE timestamps
        # were fitted onto the nsys timeline. With no KTRACE in the capture
        # there is no fit, and printing "±0 us" implied a measurement.
        align = (f"   ·   KTRACE alignment ±{(cov or 0)/1000:.0f} us"
                 if getattr(st, "n_ktrace", 0) else "")
        miss = (f"   ·   {self._n_miss} TTI-MISS"
                if getattr(self, "_n_miss", 0) else "")
        phan = getattr(self, "_phantom_note", "") \
            if self.show_phantom.isChecked() else ""
        self.hdr.setText(
            f"{when}{n_ev} launches   ·   window {us(t1):.0f} us"
            f"{align}{miss}{clip}{phan}")

    SHOW_PHASE_TASK = False        # phase + host-task plots not shown

    def _draw_phases(self, st, t0, t1, us):
        if not self.SHOW_PHASE_TASK:
            return
        rows = [r for r in st.phases_in_window(t0, t1) if r[1] > t0]
        present = [n for n in PHASE_LANES
                   if any(r[3] == n for r in rows)]
        if not present:
            self.p_phase.getAxis("left").setTicks([[]])
            return
        y = {n: len(present) - 1 - i for i, n in enumerate(present)}
        x0, x1, yy, br = [], [], [], []
        for a, b, _name, base in rows:
            if base not in y:
                continue
            x0.append(us(max(a, t0)))
            x1.append(us(min(b, t1)))
            yy.append(y[base])
            col = "#4FA3E8" if base.startswith(("SLOT_DL", "cuphySetupPdsch",
                                                "cuphyRunPdsch", "PDCCH")) \
                else "#D4537E" if "Pusch" in base or base == "SLOT_UL" \
                else "#59C3A5"
            br.append(_q(col, 150))
        if not x0:
            return
        self.p_phase.addItem(pg.BarGraphItem(
            x0=x0, x1=x1, y=yy, height=0.6, brushes=br,
            pens=[pg.mkPen(None)] * len(x0)))
        self.p_phase.getAxis("left").setTicks(
            [[(v, k) for k, v in y.items()]])
        self.p_phase.setYRange(-0.6, len(present) - 0.4, padding=0)

    def _draw_kernels(self, st, t0, t1, us):
        by_stream = self.lane_mode.currentIndex() == 1
        rows = st.events_in_window(t0, t1)
        if not self.show_ktrace.isChecked():
            rows = [r for r in rows if r[3] == 0]

        if by_stream:
            keys = sorted({r[5] for r in rows if r[5] is not None})
            lanes = [f"stream {k}" for k in keys] + ["device-launched"]
            lane_of = lambda r: (lanes.index(f"stream {r[5]}")
                                 if r[5] is not None else len(lanes) - 1)
        else:
            chans = {st.kernels[r[2]][1] for r in rows}
            lanes = [c for c in CHANNEL_LANES if c in chans]
            lane_of = lambda r: lanes.index(st.kernels[r[2]][1])
        if not lanes:
            return 0, 0
        ymax = len(lanes) - 1

        x0, x1, yy, br, pn = [], [], [], [], []
        fx0, fx1, fyy = [], [], []          # search hits, drawn on top
        pts_x, pts_y = [], []
        for r in rows:
            a, b, kid, src = r[0], r[1], r[2], r[3]
            name, chan, _role, _vis, _kind = st.kernels[kid]
            try:
                lane = ymax - lane_of(r)
            except ValueError:
                continue
            if b is None:
                # KTRACE start with no measured end — the kernel was not
                # instrumented with KERNEL_TIME_END_PRINT, so only the launch
                # instant is known. Drawn as a mark, never as a bar with an
                # invented width.
                pts_x.append(us(a))
                pts_y.append(lane)
                self._bars.append((us(a), us(a), lane, r))
                continue
            xa, xb = us(max(a, t0)), us(min(b, t1))
            if xb <= xa:
                xb = xa + 0.05
            col = channel_color(chan)
            if getattr(self, "_find_ids", None) and kid in self._find_ids:
                # Held back for a separate layer. Inside the shared
                # BarGraphItem the hit sat at the same z as its neighbours, so
                # on a busy lane the thing you searched for was still buried
                # under whatever happened to be drawn after it.
                #
                # The geometry lists are appended AFTER this branch, not
                # before: a hit contributes no bar to the shared item, and
                # appending its x0/x1/y there while its brush and pen went to
                # the other layer left the shared item with more rects than
                # pens - which BarGraphItem indexes positionally.
                fx0.append(xa)
                fx1.append(xb)
                fyy.append(lane)
                self._bars.append((xa, xb, lane, r))
                continue
            x0.append(xa)
            x1.append(xb)
            yy.append(lane)
            if src == 1:
                br.append(_q(col, 90))
                pn.append(pg.mkPen(col, width=1, style=Qt.PenStyle.DashLine))
            else:
                br.append(_q(col, 210))
                pn.append(pg.mkPen(None))
            self._bars.append((xa, xb, lane, r))
        if x0:
            self.p_kern.addItem(pg.BarGraphItem(x0=x0, x1=x1, y=yy, height=0.66,
                                                brushes=br, pens=pn))
        if fx0:
            hit = pg.BarGraphItem(x0=fx0, x1=fx1, y=fyy, height=0.86,
                                  brush=pg.mkBrush(FIND_FILL),
                                  pen=pg.mkPen(FIND_EDGE, width=2))
            hit.setZValue(40)
            self.p_kern.addItem(hit)
            # A caret above each hit. Over a 48-slot window a single launch is
            # a hairline a couple of pixels wide, so recolouring the bar alone
            # is not enough to find it - the marker is what you actually see.
            mk = pg.ScatterPlotItem(
                x=[(a + b) / 2.0 for a, b in zip(fx0, fx1)],
                y=[y + 0.60 for y in fyy], symbol="t", size=_s(11),
                brush=pg.mkBrush(FIND_FILL), pen=pg.mkPen(FIND_EDGE, width=1))
            mk.setZValue(41)
            self.p_kern.addItem(mk)
        self._n_found = len(fx0)
        if pts_x:
            self.p_kern.addItem(pg.ScatterPlotItem(
                x=pts_x, y=pts_y, symbol="d", size=7,
                brush=_q("#E07A5F", 220), pen=pg.mkPen("#E07A5F")))
        self._lane_names = list(lanes)
        self.p_kern.getAxis("left").setTicks(
            [[(ymax - i, n) for i, n in enumerate(lanes)]])
        self.p_kern.setYRange(-0.7, ymax + TOP_PAD, padding=0)
        return len(rows), ymax

    # -- phantom stream lane (beta) -----------------------------------------
    def _draw_phantom(self, st, t0, t1, us, ymax):
        """A hypothetical co-tenant lane above the real ones.

        Drawn as a STACK whose full height is the whole roofline, so the ink
        the phantoms occupy is read directly as the fraction of the GPU they
        would hold. The dashed line above them is the spare envelope itself --
        the ceiling they are filling -- so the gap between the two is visible
        as unusable fragmentation rather than being hidden by a summary.

        Every bar ends where Aerial reclaims the resource named in its tooltip.
        """
        if not getattr(self, "show_phantom", None) or \
                not self.show_phantom.isChecked() or not st:
            return ymax
        try:
            ctx = spare.context(st)
            ctx.apply_scale = self.chk_scale.isChecked()
            bn = spare.auto_bin_ns(t0, t1)
            live = 1 if self.phantom_mode.currentIndex() == 0 else 6
            key = (id(st), t0, t1, bn, live, ctx.apply_scale,
                   ctx.spec.describe(), ctx.n_sm_cotenant)
            hit = getattr(self, "_ph_cache", {}).get(key)
            if hit is None:
                hit = spare.window_phantoms(st, t0, t1, ctx, bin_ns=bn,
                                            max_live=live)
                self._ph_cache = {key: hit}
            ph, (sb, sw, ss, sr), prof = hit
        except Exception as exc:                     # never break the tab
            self._phantom_note = f"   \u00b7   phantom lane unavailable: {exc}"
            return ymax

        lane = ymax + 1
        W = float(ctx.rf.max_warps_per_sm or 64)
        H = 0.92                                     # lane height == roofline

        # Stack base for each phantom: the warps held by phantoms launched
        # before it that are still live. Retirement is LIFO, so launch order
        # and stack order agree and the bars never cross.
        order = sorted(range(len(ph)), key=lambda i: ph[i].i0)
        act, base = [], {}
        for i in order:
            q = ph[i]
            act = [a for a in act if a[0] > q.i0]
            base[i] = sum(a[1] for a in act)
            act.append((q.i1, q.vec.warps_per_sm))

        x0, x1, y0, y1, br, pn = [], [], [], [], [], []
        self._phantoms = []
        for i in order:
            q = ph[i]
            xa, xb = us(max(q.t0, t0)), us(min(q.t1, t1))
            if xb <= xa:
                xb = xa + 0.05
            b0 = lane - H / 2.0 + (base[i] / W) * H
            b1 = b0 + (q.vec.warps_per_sm / W) * H
            x0.append(xa)
            x1.append(xb)
            y0.append(b0)
            y1.append(b1)
            # Alpha tracks duration: the long survivors are the ones that
            # matter for a co-tenant and should not be lost among slivers.
            a = 90 + int(min(1.0, q.dur_us / 200.0) * 120)
            br.append(_q(PHANTOM_FILL, a))
            pn.append(pg.mkPen(_q(PHANTOM_EDGE, 150), width=1))
            self._phantoms.append((xa, xb, b0, b1, q))
        if x0:
            it = pg.BarGraphItem(x0=x0, x1=x1, y0=y0, y1=y1,
                                 brushes=br, pens=pn)
            it.setZValue(20)
            self.p_kern.addItem(it)

        # the spare envelope the stack is filling
        if sw:
            xs = [us(t0 + i * bn) for i in range(len(sw))]
            ys = [lane - H / 2.0 + (min(v, W) / W) * H for v in sw]
            env = pg.PlotDataItem(x=xs, y=ys,
                                  pen=pg.mkPen(_q(PHANTOM_EDGE, 170), width=1,
                                               style=Qt.PenStyle.DashLine))
            env.setZValue(21)
            self.p_kern.addItem(env)

        names = list(getattr(self, "_lane_names", []))
        ticks = [(ymax - i, n) for i, n in enumerate(names)]
        ticks.append((lane, "\u25b8 phantom (spare)"))
        self.p_kern.getAxis("left").setTicks([ticks])
        self.p_kern.setYRange(-0.7, lane + TOP_PAD, padding=0)

        d = sorted(q.dur_us for q in ph) or [0]
        occ = sorted(q.vec.occ_pct for q in ph) or [0]
        eff = ""
        if getattr(prof, "oracle", None) and prof.oracle[1] > 0:
            eff = (f", captures {100*prof.oracle[0]/prof.oracle[1]:.0f}% of "
                   f"spare warp-time")
        self._phantom_note = (
            f"   \u00b7   phantom {len(ph)} kernels "
            f"({'ORACLE' if live == 1 else 'greedy heuristic'}{eff}), "
            f"D p50 {d[len(d)//2]:.0f}us "
            f"p90 {d[int(.9*(len(d)-1))]:.0f}us, occ p50 {occ[len(occ)//2]:.0f}%"
            f" [{ctx.note()}]")
        return lane

    def _cotenant_changed(self):
        if not self.store:
            return
        try:
            ctx = spare.context(self.store)
            ctx.spec = spare.CotenantSpec.parse(
                self.cotenant_edit.text(), spare.CotenantSpec.from_env())
            self.cotenant_edit.setStyleSheet(_style_edit())
        except ValueError:
            self.cotenant_edit.setStyleSheet(
                _style_edit() + f"QLineEdit{{border-color:{MISS_X};}}")
            return
        self.redraw()

    def _scale_changed(self):
        if self.store:
            try:
                spare.context(self.store).apply_scale = \
                    self.chk_scale.isChecked()
            except Exception:
                pass
        self.redraw()

    def show_spare(self):
        """Quantify spare GPU for the drawn window in the harvester's own
        seven dimensions, and put it in the detail panel."""
        st = self.store
        if not st or not st.n_slots:
            return
        lo = getattr(self, "_draw_lo", self.lo)
        hi = getattr(self, "_draw_hi", self.hi)
        self.info_title.setText("Spare GPU")
        self.show_info()
        self.info.setPlainText("computing\u2026")
        QApplication.processEvents()
        try:
            ctx = spare.context(st)
            ctx.apply_scale = self.chk_scale.isChecked()
            rep = spare.analyse(st, lo, hi, ctx=ctx, bin_ns=1000, max_tti=200)
        except Exception as exc:
            self.info.setPlainText(f"spare analysis failed:\n{exc}\n\n"
                                   f"{traceback.format_exc()}")
            return
        if rep is None:
            self.info.setPlainText("no slots in this window")
            return
        L = [f"SPARE GPU  \u2014  TTI {lo}..{hi}", ""]
        L += rep.summary_lines()
        one = rep.n_tti <= 1
        L += ["", "CAPACITY-DURATION FRONTIER",
              "  how big a co-tenant kernel can be, as a function of how long",
              "  it runs."]
        if one:
            L += ["",
                  "  ONE TTI SELECTED - the p5/p50/p95 columns below are all "
                  "the same",
                  "  number, because a single slot has no distribution. This "
                  "is the",
                  "  spare in THIS slot, not a bound a co-tenant could rely "
                  "on. Widen",
                  "  the window to get a percentile worth trusting."]
        else:
            L += [f"  Computed per TTI over {rep.n_tti} TTI, then a percentile "
                  f"ACROSS them:",
                  "  a co-tenant has to survive the worst TTI it meets, not "
                  "the mean."]
        L += ["",
              "  DERIVATION, per TTI",
              "    1. sweep the slot between launch boundaries (1 us bins)",
              "    2. grant each live launch, in launch order, the blocks it",
              "       can hold: min(grid, k x N_ctx) where k is the occupancy",
              "       calculator under the launch's own carveout and N_ctx",
              "       its MPS context's SM cap - then clip to what is left of",
              "       that context's N_ctx SMs and of the device, in blocks,",
              "       warps, shared memory (+1 KiB/block) and registers",
              ("    3. multiply by the DCGM ratio "
               f"{rep.ctx.scale:.3f} (toggled ON)" if rep.ctx and
               rep.ctx.apply_scale else
               "    3. no DCGM scaling (validation only, see Check above)"),
              "    4. spare = roofline - demand, clamped at zero; shared",
              f"       memory against the live carveout ({rep.smem_mode})",
              "    5. for each D, take the SLIDING MINIMUM over a D-wide "
              "window",
              "       and keep the best position: the largest capacity free",
              "       CONTINUOUSLY for D us somewhere in the slot",
              "    6. percentile across TTI, taken per dimension",
              "    7. K* = min_d floor(capacity_d / h_d(x)) for the co-tenant",
              f"       shape ({rep.ctx.spec.describe() if rep.ctx else ''});",
              "       with B searched, the B claiming the most warps. S and R",
              "       stay at the co-tenant's need; (opt N) is the S that",
              "       would fit under the optimistic carveout",
              ""]
        hdr = (f"  {'D us':>6} {'free w/SM':>10}   shape  [B,S,R,D,n,P,M]"
               if one else
               f"  {'D us':>6} {'p5':>7} {'p50':>7} {'p95':>7}   "
               f"shape at p5  [B,S,R,D,n,P,M]")
        L.append(hdr)
        for D, p5, p50, p95, v in rep.frontier:
            head = (f"  {D:>6} {p5:>10.1f}   " if one else
                    f"  {D:>6} {p5:>7.1f} {p50:>7.1f} {p95:>7.1f}   ")
            if not v.feasible:
                L.append(head + f"INFEASIBLE \u2014 binding: {v.reason}")
                continue
            so = (f" (opt {v.S_opt})" if v.S_opt is not None
                  and v.S_opt != v.S else "")
            L.append(head + f"B={v.B} S={v.S}{so} R={v.R} D={D} n={v.n} "
                            f"P={v.P:.0f} M={v.M}")
            pad = " " * (len(head) - 2)
            L.append(f"  {pad}k={v.k} G={v.G} "
                     f"{v.warps_per_sm:.0f} warps/SM occ {v.occ_pct:.1f}% "
                     f"[{v.limiter}]")
        if rep.phantoms:
            d = sorted(q.dur_us for q in rep.phantoms)
            eff = (100.0 * rep.oracle_captured / rep.oracle_avail
                   if rep.oracle_avail else 0.0)
            L += ["", "ORACLE PHANTOM SCHEDULE (this window)",
                  f"  {len(rep.phantoms)} kernels, optimal for captured "
                  f"warp-time: {eff:.1f}% of the",
                  "  window's spare warp-time is reachable by one co-tenant "
                  "kernel at a time",
                  f"  D p10 {d[int(.1*(len(d)-1))]:.0f} p50 {d[len(d)//2]:.0f} "
                  f"p90 {d[int(.9*(len(d)-1))]:.0f} max {max(d):.0f} us"]
            from collections import Counter
            for k, n in Counter(q.retired_by for q in rep.phantoms).most_common():
                L.append(f"    ends: {k:<16} {n}")
        L += ["", "PROVENANCE",
              f"  N_SM, frame buffer, GPU name   {rep.rf.src}",
              f"  M (device memory)              {rep.sustained.src_M}"
              f"  \u2014 DCGM fb_total - fb_used",
              f"  level (occupancy)              {rep.prof.anchor_src}",
              "  shape within the TTI           MODELLED from nsys launch",
              "                                 geometry under MPS SM caps and",
              "                                 launch-order allocation; wave",
              "                                 tails are not visible, so it",
              "                                 reads somewhat high vs DCGM.",
              "",
              "  Reachable only if the co-tenant shares the GPU spatially",
              "  (same MPS server, or a green context). Separate processes",
              "  without MPS time-slice, and none of this is usable.",
              "",
              "  The profile is a MEAN FIELD over "
              f"{rep.rf.num_sm} SMs. It does not support",
              "  a claim about any individual SM."]
        self.info.setPlainText("\n".join(L))

    def _draw_tasks(self, st, us):
        if not self.SHOW_PHASE_TASK:
            return
        rows = []
        for i in range(self._draw_lo, self._draw_hi + 1):
            rows += st.tasks_in_slot(i)
        if not rows:
            self.p_task.getAxis("left").setTicks([[]])
            return
        names = sorted({r[2] for r in rows})
        y = {n: len(names) - 1 - i for i, n in enumerate(names)}
        x0, x1, yy, br = [], [], [], []
        for a, b, nm, *_rest in rows:
            x0.append(us(a))
            x1.append(us(b))
            yy.append(y[nm])
            br.append(_q("#7F77DD" if nm.startswith("DL") else "#BA7517", 160))
        self.p_task.addItem(pg.BarGraphItem(
            x0=x0, x1=x1, y=yy, height=0.6, brushes=br,
            pens=[pg.mkPen(None)] * len(x0)))
        self.p_task.getAxis("left").setTicks([[(v, k) for k, v in y.items()]])
        self.p_task.setYRange(-0.6, len(names) - 0.4, padding=0)

    # -- interaction --------------------------------------------------------
    def _on_click(self, ev):
        if not self.store:
            return
        pt = self.p_kern.vb.mapSceneToView(ev.scenePos())
        x, y = pt.x(), pt.y()

        # The phantom lane sits above every kernel lane and its bars carry a
        # y RANGE rather than a lane centre, so it is tested first and on its
        # own terms. A phantom is a stacked rect: hitting it means landing
        # inside that rect, not within half a lane of a centre line.
        ph_hits = []
        for xa, xb, b0, b1, q in getattr(self, "_phantoms", []):
            if not (b0 - 0.02 <= y <= b1 + 0.02):
                continue
            d = 0.0 if xa <= x <= xb else min(abs(x - xa), abs(x - xb))
            if d <= 8.0:
                ph_hits.append((d, xa, xb, q))
        if ph_hits:
            ph_hits.sort(key=lambda h: (h[0], h[1]))
            self.show_info()
            self.info_title.setText("Phantom kernel")
            head = ""
            if len(ph_hits) > 1:
                head = (f"{len(ph_hits)} phantom kernels under the cursor - "
                        f"stacked here. The first is detailed below.\n\n")
                for n, (_d, xa, xb, q) in enumerate(ph_hits, 1):
                    head += (f"  {n}. {q.vec.warps_per_sm:5.1f} warps/SM  "
                             f"{xa:9.2f} -> {xb:9.2f} us  "
                             f"D={q.dur_us:.0f}us  retired by "
                             f"{q.retired_by}\n")
                head += "\n" + "-" * 74 + "\n\n"
            self.info.setPlainText(head + self._describe_phantom(ph_hits[0][3]))
            return

        if not self._bars:
            return
        hits = []
        for i, (xa, xb, lane, r) in enumerate(self._bars):
            if abs(lane - y) > 0.5:
                continue
            d = 0.0 if xa <= x <= xb else min(abs(x - xa), abs(x - xb))
            if d <= 8.0:
                hits.append((d, xa, xb, i, r))
        if not hits:
            return
        hits.sort(key=lambda h: (h[0], h[1]))
        # Clicking a kernel is a request to see its record, so the panel comes
        # back whether or not it was dismissed.
        self.show_info()
        self.info_title.setText("Kernel detail")
        head = ""
        if len(hits) > 1:
            head = (f"{len(hits)} kernels under the cursor - overlapping on "
                    f"this lane, outlined amber. The first is detailed below.\n\n")
            for n, (d, xa, xb, i, r) in enumerate(hits, 1):
                nm = self.store.kernel_names.get(r[2], "?")
                head += f"  {n}. {nm[:56]:<56} {xa:9.2f} -> {xb:9.2f} us\n"
            head += "\n" + "-" * 74 + "\n\n"
        self.info.setPlainText(head + self._describe(hits[0][4]))

    def _describe_phantom(self, q) -> str:
        """The record for a hypothetical co-tenant kernel.

        Laid out like _describe() so a phantom reads against the real kernels
        beside it. Every field that is not a measurement says so.
        """
        st = self.store
        v = q.vec
        try:
            ctx = spare.context(st)
            rf = ctx.rf
        except Exception:
            return "phantom context unavailable"
        oracle = v.d_safe_us is not None
        W = rf.max_warps_per_sm
        wall = st.wall_ns(q.t0)
        title = ("PHANTOM  \u2014  oracle co-tenant kernel" if oracle else
                 "PHANTOM  \u2014  greedy fill (HEURISTIC, not an oracle)")
        L = [title, "-" * len(title),
             "channel   \u25b8 phantom (spare)",
             "role      co-tenant kernel in the OPTIMAL single-kernel schedule"
             if oracle else
             "role      co-tenant candidate from a greedy stacked fill",
             "captured  NOT CAPTURED \u2014 this kernel never ran. It is part "
             "of the",
             "          schedule a perfectly informed co-tenant would have run "
             "here.",
             ""]
        L += [f"source    MODEL \u2014 {ctx.caps_note()}",
              f"          {ctx.note()}",
              f"co-tenant {ctx.spec.describe()}",
              f"duration  {_fmt_ns(int(q.dur_us * 1000))}   chosen by the "
              f"oracle, not measured" if oracle else
              f"duration  {_fmt_ns(int(q.dur_us * 1000))}   not measured"]
        if oracle:
            L.append(f"D*        {_fmt_ns(int(v.d_safe_us * 1000))}   the "
                     f"longest THIS kernel (k={v.k}) could")
            L.append("          have run from its start before Aerial "
                     "reclaims a resource")
        why = q.retired_by
        if why == "relaunch":
            L += ["ends      BY CHOICE. It still fitted, but ending here and "
                  "relaunching",
                  "          at a different size captures more warp-time "
                  "after paying",
                  f"          the {ctx.spec.launch_gap_us:g} us relaunch cost. "
                  "D < D* says exactly that."]
        elif why == "end of window":
            L += ["ends      at the edge of the drawn window; the schedule "
                  "continues",
                  "          beyond it. Widen the window to see the rest."]
        elif why == "duration cap":
            L += ["ends      at an imposed duration ceiling, not a resource "
                  "limit."]
        else:
            L += [f"ends      because Aerial reclaims {why}: staying one more "
                  "segment",
                  "          would cost a block, and a resident CUDA block "
                  "cannot be",
                  "          resized or evicted."]
        s0 = st.slot_of_ns(q.t0)
        s1 = st.slot_of_ns(max(q.t0, q.t1 - 1))
        if s1 > s0:
            r0 = st.slot(s0)
            off = (q.t0 - r0[1]) / 1000.0 if r0 else 0.0
            L += [f"          spans {s1 - s0 + 1} TTIs: starts {off:.0f} us "
                  f"into {st.slot_label(s0)} and",
                  f"          ends inside {st.slot_label(s1)} - Aerial had no "
                  "demand that",
                  "          would evict it on the far side."]
        L += ["stream    would need one of its own; this lane is not a real "
              "stream"]
        L += ["", f"t0        {q.t0:,} ns on nsys timeline",
              f"t1        {q.t1:,} ns"]
        if wall:
            L += [f"wall      {_fmt_wall(wall)} UTC",
                  f"          {wall:,} ns since epoch",
                  f"slot      {st.slot_label(st.slot_of_ns(q.t0))}"]
        if not v.feasible:
            L += ["", f"INFEASIBLE \u2014 binding resource: {v.reason}"]
            return "\n".join(L)

        wpb = max(1, -(-v.B // rf.warp_size))
        L += ["", f"grid      {v.G} blocks   "
                  f"({v.k} blocks/SM x {v.n_sm} SM)",
              f"block     {v.B} threads   ({wpb} warps/block)",
              f"warps     {v.G * wpb}   "
              f"({v.warps_per_sm:.0f} of {W} resident per SM "
              f"= {v.occ_pct:.1f}%)",
              f"registers {v.R}/thread"
              + (f"   could be {v.R + v.R_room} at this k"
                 if oracle and v.R_room else ""),
              f"shared    {v.S} B/block (+{rf.reserved_smem_per_block} B "
              f"runtime reserve)"
              + (f"   could be {v.S + v.S_room} B at this k"
                 if oracle and v.S_room else "")]
        L += ["", f"occupancy {v.occ_pct:.1f}%   k limited by {v.limiter}",
              "          WOULD hold, not achieved \u2014 the occupancy "
              "calculator",
              "          applied to a shape that was never launched"]
        if q.margin:
            L += ["", "spare left beside it at the tightest instant "
                      "(min over its life):"]
            unit = {"warps": "warps/SM", "blocks": "blk/SM",
                    "shared memory": "B/SM", "registers": "regs/SM"}
            for dim in ("blocks", "warps", "shared memory", "registers"):
                if dim in q.margin:
                    L.append(f"          {dim:<14} {q.margin[dim]:>12,.1f} "
                             f"{unit[dim]}")
            L += ["          one more block of this shape does not fit in at "
                  "least one",
                  "          of these - that is what makes k maximal."]

        L += ["", "SPARE VECTOR  [B, S, R, D, n, P, M]",
              f"  B  {v.B:<10} threads per block",
              f"  S  {v.S:<10} B shared memory per block",
              f"  R  {v.R:<10} registers per thread",
              f"  D  {v.D:<10.0f} us   this kernel's duration in the schedule",
              f"  n  {v.n:<10} wave",
              f"  P  {v.P:<10.0f} us   the TTI",
              f"  M  {v.M:<10} MiB  [{v.src_M}]"]
        if v.src_M == spare.MEASURED:
            L.append("     \u2514\u2500 DCGM fb_total - fb_used, measured "
                     "outright")
        L += [f"  derived: k={v.k}  G={v.G}  duty={v.duty:.3f}  "
              f"f={v.f_hz/1000:.1f} kHz"]

        L += ["", "HOW THIS SHAPE WAS DERIVED",
              "  envelope  C(t) = roofline - Aerial demand, per SM, in blocks,",
              "            warps, shared memory (+1 KiB/block) and registers.",
              "            Demand: every live launch granted blocks in launch",
              "            order under its MPS context's SM cap and the "
              "device's",
              "            capacity; queued blocks hold nothing. "
              + (f"Scaled x{ctx.scale:.3f}"
                 " to DCGM." if ctx.apply_scale else "Not scaled to DCGM."),
              "  cost      h(x) = (1, w, align(S + 1 KiB), w * align(32 R, "
              "256))",
              "            per block; k blocks/SM hold k * h(x).",
              "  feasible  k * h(x) <= C(t) for every t in [t0, t1) - a "
              "resident",
              "            block cannot be evicted or resized.",
              "  size      k = K*(x; t0, D) = min_d floor(min C_d / h_d),",
              f"            here {v.k}, limited by {v.limiter}.",
              ]
        if oracle:
            L += ["  schedule  the lane is the OPTIMAL set of non-overlapping",
                  "            kernels maximising captured warp-time",
                  "              sum_i  w_i * k_i * (D_i - lambda)",
                  f"            with lambda = {ctx.spec.launch_gap_us:g} us "
                  f"relaunch cost and D >= "
                  f"{ctx.spec.d_min_us:g} us, solved exactly by",
                  "            dynamic programming over the envelope's "
                  "change points",
                  "            (weighted interval scheduling). B is chosen "
                  "per kernel",
                  "            from 32..1024 unless the co-tenant shape fixes "
                  "it."]
        else:
            L += ["  schedule  greedy: each kernel launched at the first "
                  "instant it",
                  "            fits, up to 6 at once. A HEURISTIC with no "
                  "optimality",
                  "            claim; use the oracle mode for any bound."]
        L += [f"  G         = k x N_cotenant = {v.k} x {v.n_sm} = {v.G}",
              "",
              "hand this to the harvester as:",
              f"  {v.ctl_line()}"]

        L += ["", "WHAT THIS IS NOT",
              "  An oracle has perfect foreknowledge of Aerial's schedule and",
              "  pays only the stated relaunch cost. Scheduler latency, cache",
              "  and memory-bandwidth interference are NOT modelled and are",
              "  real; the harvester's measured ceiling includes them, so",
              "  compare the two rather than trusting this alone. It is only",
              "  reachable if the co-tenant shares the GPU spatially (same",
              "  MPS server or a green context).",
              "",
              f"  Capacity is a MEAN FIELD over {rf.num_sm} SMs. It does not",
              "  support a claim about any individual SM."]
        return "\n".join(L)

    def _describe(self, r) -> str:
        st = self.store
        t0, t1, kid, src, dkind, stream, blocks, tpb, warps, regs, smem = r
        name, chan, role, vis, kind = st.kernels[kid]
        wall = st.wall_ns(t0)
        L = [f"{name}", "-" * max(12, len(name)), f"channel   {chan}",
             f"role      {role}",
             f"captured  {st.source_label(kid) or 'no events in this capture'}",
             ""]
        if kind == "device_fn":
            L += ["NOT A KERNEL LAUNCH",
                  "  A __device__ function whose body carries a KTRACE",
                  "  macro. It runs inside its parent kernel, so the grid",
                  "  and block below are the PARENT's, and this record is",
                  "  excluded from launch counts and occupancy.", ""]
        if src == 0:
            L += ["source    nsys (CUPTI, host-launched)",
                  f"duration  {_fmt_ns(t1-t0 if t1 else None)}   measured",
                  f"stream    {stream}"]
        else:
            L += ["source    AERIAL_KTRACE (device printf)",
                  "          nsys cannot see this kernel: its graph is",
                  "          launched from device code, so there is no",
                  "          host-side launch event to record.",
                  f"start     ±{(st.clock_iqr_ns or 0)/1000:.0f} us "
                  f"(clock cross-calibration IQR)"]
            if t1:
                L.append(f"duration  {_fmt_ns(t1-t0)}   ESTIMATED "
                         "(per-kernel median)")
            else:
                L.append("duration  not instrumented "
                         "(no KERNEL_TIME_END_PRINT)")
        L += ["", f"t0        {t0:,} ns on nsys timeline"]
        if wall:
            L += [f"wall      {_fmt_wall(wall)} UTC",
                  f"          {wall:,} ns since epoch",
                  f"slot      {st.slot_label(st.slot_of_ns(t0))}"]
        L += ["", f"grid      {blocks} blocks",
              f"block     {tpb} threads",
              f"warps     {warps}  ({100.0*min(warps or 0, st.max_warps)/st.max_warps:.1f}% "
              f"of {st.max_warps} resident)"]
        if regs is not None:
            L.append(f"registers {regs}/thread")
        if smem is not None:
            L.append(f"shared    {smem} B")
        krow = [k for k in st.kernel_table() if k[0] == name]
        if krow:
            (_n, _c, _r, _v, nl, dn, dmin, dmed, dp99, dmax,
             occ, lim, res_src, kregs, ksmem, _kind, _parent) = krow[0]
            if occ is not None:
                L += ["", f"theoretical occupancy  {occ:.1f}%   "
                          f"limited by {lim}"]
                if res_src == "cuobjdump":
                    L.append("          LOWER BOUND — registers/shared come "
                             "from the static")
                    L.append("          cuobjdump table (max across template "
                             "instantiations),")
                    L.append("          not from this launch.")
            L += ["", f"across capture: {nl:,} launches"]
            if dn:
                L += [f"  duration  min {_fmt_ns(dmin)}  med {_fmt_ns(dmed)}"
                      f"  p99 {_fmt_ns(dp99)}  max {_fmt_ns(dmax)}",
                      f"            over {dn:,} samples"]
        return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# Channels
# ─────────────────────────────────────────────────────────────────────────────

    # Canvas floor/ceiling. The ceiling exists because a QGraphicsView renders
    # its whole canvas: a runaway drag to 100k px would allocate and repaint it.
    CANVAS_MIN = (320, 150)
    CANVAS_MAX = (40000, 40000)

    def _canvas_size(self):
        """Current canvas size. Layout-managed or pinned, it is the graphics
        widget's own size either way."""
        return max(1, self.gl.width()), max(1, self.gl.height())

    def _set_canvas(self, w, h):
        """Pin the drawing surface to an explicit size.

        widgetResizable has to go off first: while it is on, the scroll area
        forces the canvas back to the viewport on every resize, which is why
        the canvas could never grow horizontally at all.
        """
        w = max(_s(self.CANVAS_MIN[0]), min(int(w), _s(self.CANVAS_MAX[0])))
        h = max(_s(self.CANVAS_MIN[1]), min(int(h), _s(self.CANVAS_MAX[1])))
        self.chart_scroll.setWidgetResizable(False)
        self.gl.setFixedSize(w, h)
        self._chart_fixed = True
        self._show_canvas_size()

    def _grip_drag(self, dx: int, dy: int):
        w, h = self._canvas_size()
        self._set_canvas(w + dx, h + dy)

    def _grip_reset(self):
        """Give the canvas back to the layout, filling the pane as before."""
        self.gl.setMinimumSize(0, 0)
        self.gl.setMaximumSize(16777215, 16777215)      # QWIDGETSIZE_MAX
        self.chart_scroll.setWidgetResizable(True)
        self._chart_fixed = False
        self._show_canvas_size()

    def zoom_canvas(self, factor: float):
        """Grow or shrink the drawing surface in both axes at once."""
        w, h = self._canvas_size()
        self._set_canvas(w * factor, h * factor)

    def resize_canvas_h(self, factor: float):
        """Change chart height only, leaving width - and so the visible time
        window - untouched."""
        w, h = self._canvas_size()
        self._set_canvas(w, h * factor)

    def _show_canvas_size(self):
        if not hasattr(self, "canvas_lbl"):
            return
        w, h = self._canvas_size()
        # A layout-managed canvas fits its pane by definition. Comparing sizes
        # instead reported "(scrolls)" for one paint after a reset, while the
        # scroll area had not yet reflowed the viewport.
        if not self._chart_fixed:
            over = False
        else:
            vp = self.chart_scroll.viewport()
            over = (w > vp.width() + 2 or h > vp.height() + 2)
        self.canvas_lbl.setText(
            f"{w}\u00d7{h}" + ("  (scrolls \u2014 use \u270b)" if over
                                else "  (fits)"))

    def eventFilter(self, obj, ev):
        """Hand tool: drag the viewport across an oversized canvas.

        Only meaningful once the canvas is bigger than its pane. When it is
        not, the events are passed through so the ViewBox pans the data range
        as before - otherwise the hand tool would look broken at default size.
        """
        if not getattr(self, "btn_hand", None) or not self.btn_hand.isChecked():
            return False
        hb = self.chart_scroll.horizontalScrollBar()
        vb = self.chart_scroll.verticalScrollBar()
        if hb.maximum() == 0 and vb.maximum() == 0:
            return False
        t = ev.type()
        if (t == QEvent.Type.MouseButtonPress
                and ev.button() == Qt.MouseButton.LeftButton):
            self._pan_from = ev.globalPosition()
            self._pan_at = (hb.value(), vb.value())
            self.gl.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        if t == QEvent.Type.MouseMove and getattr(self, "_pan_from", None):
            d = ev.globalPosition() - self._pan_from
            hb.setValue(int(self._pan_at[0] - d.x()))
            vb.setValue(int(self._pan_at[1] - d.y()))
            return True
        if t == QEvent.Type.MouseButtonRelease and getattr(self, "_pan_from", None):
            self._pan_from = None
            self.gl.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return True
        return False

    # -- window selection ---------------------------------------------------
    def _build_window_pane(self) -> QWidget:
        """Address a window directly, by SFN, wall clock, or slot index.

        Dragging a region on a global bar chart is fine for browsing and no
        use at all when you already know where you want to be - and after a
        capture is reduced to a table of TTI-MISS SFNs, knowing where you want
        to be is the normal case. All three coordinates the rest of the tool
        prints are accepted here so a number can be copied straight back in.
        """
        box = QFrame()
        box.setStyleSheet(
            f"QFrame{{background:{BG2};border:1px solid {EDGE};"
            f"border-radius:{_s(5)}px;}}")
        h = QHBoxLayout(box)
        h.setContentsMargins(_s(9), _s(6), _s(9), _s(6))
        h.setSpacing(_s(7))

        h.addWidget(_lbl("Window", 12, ACCENT, bold=True))
        h.addWidget(_lbl("go to", 12, LABEL))

        self.jump_mode = QComboBox()
        self.jump_mode.addItems(["SFN.slot", "TTI slot #", "UTC time",
                                 "prev TTI-MISS", "next TTI-MISS"])
        self.jump_mode.setStyleSheet(_style_combo())
        self.jump_mode.setToolTip(
            "Which coordinate the box holds:\n"
            "  SFN.slot        the DU's own frame number, e.g. 553.0\n"
            "  TTI slot #      index into this capture, 0-based\n"
            "  UTC time        03:51:03.888  or  03:51:03\n"
            "  prev/next TTI-MISS   step to a dropped slot and CENTRE the\n"
            "                  window on it, so the slots either side of the\n"
            "                  drop are in view. No box entry needed.")
        self.jump_mode.currentIndexChanged.connect(self._jump_mode_changed)
        h.addWidget(self.jump_mode)

        self.jump_edit = QLineEdit()
        self.jump_edit.setStyleSheet(_style_edit())
        self.jump_edit.setMinimumWidth(_s(150))
        self.jump_edit.setPlaceholderText("553.0")
        self.jump_edit.returnPressed.connect(lambda: self._jump(0))
        h.addWidget(self.jump_edit)

        h.addWidget(_lbl("span", 12, LABEL))
        self.jump_span = QSpinBox()
        self.jump_span.setRange(1, 20000)
        self.jump_span.setValue(40)
        self.jump_span.setStyleSheet(_style_spin())
        self.jump_span.setToolTip("How many TTI the window covers")
        h.addWidget(self.jump_span)
        h.addWidget(_lbl("TTI", 12, LABEL))

        self.btn_jump = QPushButton("Go")
        self.btn_jump.setStyleSheet(_style_button())
        self.btn_jump.clicked.connect(lambda: self._jump(0))
        h.addWidget(self.btn_jump)

        # SFN wraps every 10.24 s, so one SFN.slot names several real slots in
        # any longer capture. These step between those occurrences instead of
        # pretending the first one is the only one.
        self.btn_prev = QPushButton("\u25c2")
        self.btn_prev.setStyleSheet(_style_button())
        self.btn_prev.setToolTip("Previous occurrence of this SFN (it wraps "
                                 "every 1024 frames = 10.24 s)")
        self.btn_prev.clicked.connect(lambda: self._jump(-1))
        h.addWidget(self.btn_prev)
        self.btn_next = QPushButton("\u25b8")
        self.btn_next.setStyleSheet(_style_button())
        self.btn_next.setToolTip("Next occurrence of this SFN")
        self.btn_next.clicked.connect(lambda: self._jump(+1))
        h.addWidget(self.btn_next)

        self.jump_msg = _lbl("", 12, LABEL)
        h.addWidget(self.jump_msg, 1)

        self._sfn_hits, self._sfn_i = [], 0
        self._sfn_key = None
        self._last_miss = None
        self._jump_mode_changed()
        return box

    def _jump_mode_changed(self, _i=None):
        self._sfn_hits = []
        m = self.jump_mode.currentText()
        # A mode change is a fresh intent; drop the miss cursor so the next
        # press starts from what is on screen.
        self._last_miss = None
        _miss = m.endswith("TTI-MISS")
        self.jump_edit.setPlaceholderText(
            {"SFN.slot": "553.0", "TTI slot #": "15136",
             "UTC time": "03:51:03.888"}.get(
                m, "press Go \u2014 no value needed"))
        # The box has nothing to say in the miss modes: the target comes from
        # the nvlog's own markers, not from anything typed.
        self.jump_edit.setEnabled(not _miss)
        for b in (self.btn_prev, self.btn_next):
            b.setEnabled(m == "SFN.slot" or _miss)
        self.jump_msg.setText("")

    def _jump_say(self, text, ok=True):
        """Result line. Failures are stated, never silent - a jump that did
        nothing looks identical to a jump that landed somewhere unexpected."""
        self.jump_msg.setText(text)
        self.jump_msg.setStyleSheet(
            f"color:{LABEL if ok else MISS_X};font-size:{_s(12)}px;")

    def _jump(self, step: int):
        st = self.store
        if st is None:
            self._jump_say("no session open", False)
            return
        n = st.n_slots
        if not n:
            return
        mode = self.jump_mode.currentText()
        span = int(self.jump_span.value())

        if mode.endswith("TTI-MISS"):
            lo = self._step_to_miss(mode, step, span)
            if lo is None:
                return
            self.jump_requested.emit(lo, min(n - 1, lo + span - 1))
            return

        raw = self.jump_edit.text().strip()
        if not raw:
            self._jump_say("enter a value", False)
            return

        if mode == "TTI slot #":
            try:
                lo = int(float(raw))
            except ValueError:
                self._jump_say(f"{raw!r} is not a slot number", False)
                return
            if not 0 <= lo < n:
                self._jump_say(f"slot {lo} outside 0..{n-1}", False)
                return
            note = f"slot {lo}  ({st.slot_label(lo)})"

        elif mode == "UTC time":
            lo = self._slot_from_wall(raw)
            if lo is None:
                return
            note = f"{st.slot_label(lo)}  ·  slot {lo}"

        else:                                    # SFN.slot
            txt = raw.upper().replace("SFN", "").strip()
            try:
                if "." in txt:
                    a, b = txt.split(".", 1)
                    sfn, slot = int(a), int(b)
                else:
                    sfn, slot = int(txt), None
            except ValueError:
                self._jump_say(f"{raw!r} is not an SFN - try 553.0", False)
                return
            key = (sfn, slot)
            if getattr(self, "_sfn_key", None) != key or not self._sfn_hits:
                self._sfn_hits = st.slots_by_sfn(sfn, slot)
                self._sfn_key = key
                self._sfn_i = 0
                step = 0
            if not self._sfn_hits:
                self._jump_say(
                    f"SFN {sfn}" + (f".{slot}" if slot is not None else "") +
                    " is not in the traced window", False)
                return
            self._sfn_i = (self._sfn_i + step) % len(self._sfn_hits)
            lo = self._sfn_hits[self._sfn_i]
            note = (f"{st.slot_label(lo)}  ·  slot {lo}  ·  "
                    f"occurrence {self._sfn_i+1} of {len(self._sfn_hits)}")

        hi = min(n - 1, lo + span - 1)
        r = st.slot(lo)
        w = st.wall_ns(r[1]) if r else None
        if w:
            note += f"  ·  {_fmt_wall(w)} UTC"
        self._jump_say(note)
        self.jump_requested.emit(lo, hi)

    def _step_to_miss(self, mode, step, span):
        """Centre the window on the previous / next dropped TTI.

        Centred, not started, on purpose: a dropped slot is only meaningful
        against the slots around it - what the DU was doing as it fell behind,
        and whether it recovered - and a window that BEGINS at the drop shows
        the recovery but hides the cause.
        """
        st = self.store
        misses = sorted(st.tti_miss_slot_idx()) if st else []
        if not misses:
            self._jump_say(
                "this capture has no TTI-MISS markers \u2014 either the DU "
                "dropped nothing, or it was built without the instrumentation",
                False)
            return None
        # Step from the drop we last landed on, while it is still on screen.
        # Using the window centre alone does not advance: near either end of
        # the capture the window clamps, the centre stops tracking the target,
        # and every press re-finds the same drop. Falling back to the centre
        # keeps it sensible after the window is moved by any other means.
        last = getattr(self, "_last_miss", None)
        here = (last if last is not None and self.lo <= last <= self.hi
                else (self.lo + self.hi) // 2)
        fwd = (step > 0) if step else mode.startswith("next")
        if fwd:
            nxt = next((m for m in misses if m > here), None)
            if nxt is None:
                self._jump_say(
                    f"no TTI-MISS after this window \u2014 "
                    f"{len(misses):,} in the capture, last at slot "
                    f"{misses[-1]:,} ({st.slot_label(misses[-1])})", False)
                return None
        else:
            nxt = next((m for m in reversed(misses) if m < here), None)
            if nxt is None:
                self._jump_say(
                    f"no TTI-MISS before this window \u2014 "
                    f"{len(misses):,} in the capture, first at slot "
                    f"{misses[0]:,} ({st.slot_label(misses[0])})", False)
                return None
        self._last_miss = nxt
        # Centre it, then clamp to the capture. Near an end the drop cannot sit
        # dead centre; it is still in view, and the readout gives its slot.
        lo = max(0, min(nxt - span // 2, max(0, st.n_slots - span)))
        rank = misses.index(nxt) + 1
        r = st.slot(nxt)
        w = st.wall_ns(r[1]) if r else None
        kind = (st.tti_miss_slot_idx().get(nxt) or ("", ))[0]
        self._jump_say(
            f"TTI-MISS {rank} of {len(misses):,}  \u00b7  "
            f"{st.slot_label(nxt)}  \u00b7  slot {nxt:,}"
            + (f"  \u00b7  {kind}" if kind else "")
            + (f"  \u00b7  {_fmt_wall(w)} UTC" if w else "")
            + f"  \u00b7  centred in a {span}-TTI window")
        return lo

    def _slot_from_wall(self, raw: str):
        """Parse a UTC time and return the slot holding it.

        Accepts what the rest of the UI prints - "03:51:03.888" as shown on
        the readout and in the nvlog prefix - as well as a full ISO stamp and
        a raw epoch-ns integer, so any of them can be pasted back in.
        """
        st = self.store
        epoch = st.utc_epoch_ns
        if epoch is None:
            self._jump_say("this capture has no UTC epoch", False)
            return None
        import datetime as _dt
        txt = raw.strip().replace("Z", "").replace("UTC", "").strip()
        ns = None
        if txt.isdigit() and len(txt) >= 16:              # epoch ns
            ns = int(txt)
        else:
            base = _dt.datetime.fromtimestamp(
                epoch / 1e9, tz=_dt.timezone.utc)
            for fmt, whole in (("%Y-%m-%dT%H:%M:%S.%f", True),
                               ("%Y-%m-%d %H:%M:%S.%f", True),
                               ("%Y-%m-%dT%H:%M:%S", True),
                               ("%H:%M:%S.%f", False),
                               ("%H:%M:%S", False)):
                try:
                    t = _dt.datetime.strptime(txt, fmt)
                except ValueError:
                    continue
                if not whole:
                    # Time of day only: take the date from the capture itself.
                    t = t.replace(year=base.year, month=base.month,
                                  day=base.day)
                ns = int(t.replace(tzinfo=_dt.timezone.utc).timestamp() * 1e9)
                break
        if ns is None:
            self._jump_say(f"{raw!r} is not a time - try 03:51:03.888", False)
            return None
        rel = ns - epoch
        first = st.slot(0)
        last = st.slot(st.n_slots - 1)
        if first and last and not (first[1] <= rel <= last[2]):
            lo_w, hi_w = st.wall_ns(first[1]), st.wall_ns(last[2])
            self._jump_say(
                f"outside the traced window "
                f"({_fmt_wall(lo_w, False)} .. {_fmt_wall(hi_w, False)})",
                False)
            return None
        return st.slot_of_ns(rel)

    def set_pan_mode(self, on: bool):
        """Hand tool.

        Off, the view is locked to the lane stack and drags in time only —
        right for reading a timeline, wrong for a tall stack of lanes on a
        short window, where the lanes you want are simply off-screen and
        nothing you drag brings them back. On, the chart moves freely in both
        axes and the cursor says so.
        """
        vb = self.p_kern.vb
        self.p_kern.setMouseEnabled(x=True, y=bool(on))
        cur = (Qt.CursorShape.OpenHandCursor if on
               else Qt.CursorShape.ArrowCursor)
        for w in (vb, self.gl, self.gl.viewport()):
            try:
                w.setCursor(cur)
            except Exception:
                pass
        self.btn_hand.setText("✋ Hand ON — dragging" if on
                              else "✋ Hand — drag chart")
        if not on:
            # Leaving the hand tool restores the fitted lane stack, so a view
            # dragged off into empty space is never left stranded there.
            self.redraw()

    # Canvas state. _chart_fixed is False while the canvas is layout-managed
    # (matching its pane); it goes True once a grip or the Canvas buttons pin
    # an explicit size.
    _chart_fixed = False
    _pan_from = None

    # Whether the user wants the panel open. Deliberately not isVisible():
    # that is false whenever any ancestor is hidden, so on a background tab
    # the panel reported itself closed while it was merely off-screen, and the
    # toggle then always took the "open" branch.
    _info_open = True

    def hide_info(self):
        """Dismiss the detail panel and give its width back to the lanes."""
        if self._info_open:
            self._info_sizes = self.split.sizes()
        self._info_open = False
        self.info_box.hide()
        self.btn_details.blockSignals(True)
        self.btn_details.setChecked(False)
        self.btn_details.blockSignals(False)

    def show_info(self):
        """Bring the detail panel back at the width it had when dismissed."""
        reopening = not self._info_open
        self._info_open = True
        self.info_box.show()
        if reopening and getattr(self, "_info_sizes", None):
            self.split.setSizes(self._info_sizes)
        self.btn_details.blockSignals(True)
        self.btn_details.setChecked(True)
        self.btn_details.blockSignals(False)

    def toggle_info(self):
        self.hide_info() if self._info_open else self.show_info()

    def toggle_layout(self):
        """Swap the detail panel between right-hand side and below.

        A horizontal splitter only drags horizontally; stacking gives the
        vertical drag, which is what you want when the lane count is high and
        the chart needs height rather than width.
        """
        if self.split.orientation() == Qt.Orientation.Horizontal:
            self.split.setOrientation(Qt.Orientation.Vertical)
            self.split.setSizes([_s(620), _s(240)])
            self.btn_layout.setText("Side")
        else:
            self.split.setOrientation(Qt.Orientation.Horizontal)
            self.split.setSizes([_s(1100), _s(420)])
            self.btn_layout.setText("Stack")

    def fit_view(self):
        """Rescale so everything drawn is inside, and undo any grip resize."""
        self._grip_reset()
        vb = self.p_kern.vb
        vb.enableAutoRange(vb.XYAxes, True)
        QApplication.processEvents()
        vb.enableAutoRange(vb.XYAxes, False)
        # autoRange ignores TextItems, so re-assert the caption headroom.
        ymin, ymax = vb.viewRange()[1]
        vb.setYRange(ymin - 0.2, ymax + TOP_PAD, padding=0)

    # Concurrent kernels share a lane and overlap in x, so the topmost bar
    # hides the ones underneath - you could neither see that anything was
    # under it nor click it.
    OVERLAP_PEN = "#E8A33D"

    def _mark_overlaps(self):
        """Outline every bar sharing a lane and time span with another.

        Sort-and-sweep per lane rather than all-pairs: a 48-slot window at 20
        cells holds tens of thousands of bars and an all-pairs check stalls
        the redraw.
        """
        self._overlap = set()
        by_lane = {}
        for i, (xa, xb, lane, r) in enumerate(self._bars):
            by_lane.setdefault(lane, []).append((xa, xb, i))
        for items in by_lane.values():
            items.sort()
            far, far_i = -1e18, None
            for xa, xb, i in items:
                if xa < far:
                    self._overlap.add(i)
                    if far_i is not None:
                        self._overlap.add(far_i)
                if xb > far:
                    far, far_i = xb, i
        if not self._overlap:
            return
        xs, ys, ws = [], [], []
        for i in self._overlap:
            xa, xb, lane, _r = self._bars[i]
            xs.append((xa + xb) / 2.0); ys.append(lane); ws.append(max(xb - xa, 0.05))
        bg = pg.BarGraphItem(x=xs, y=ys, width=ws, height=0.62,
                             brush=pg.mkBrush(None),
                             pen=pg.mkPen(self.OVERLAP_PEN, width=1.4))
        bg.setZValue(20)
        self.p_kern.addItem(bg)

    # -- kernel search ------------------------------------------------------
    def _populate_find(self, st):
        """Fill the dropdown with the kernel names this capture contains.

        Sorted by name so the list is scannable; the box is editable so a long
        name can be typed or pasted rather than hunted for.
        """
        self.find_box.blockSignals(True)
        cur = self.find_box.currentText()
        self.find_box.clear()
        if st:
            names = sorted({v[0] for v in st.kernels.values()})
            self.find_box.addItems(names)
        i = self.find_box.findText(cur)
        self.find_box.setCurrentIndex(i if i >= 0 else -1)
        if i < 0:
            self.find_box.setEditText(cur)
        self.find_box.blockSignals(False)

    def find_next_kernel(self):
        """Move the window to the next slot containing the named kernel.

        Searches forward from the slot after the current one and wraps once, so
        pressing Next repeatedly walks every occurrence and then starts over
        rather than stopping silently at the end of the capture.
        """
        st = self.store
        name = self.find_box.currentText().strip()
        if not st:
            return
        if not name:
            # An empty box clears the highlight rather than doing nothing, so
            # there is a way back to an unmarked chart.
            self._find_ids = set()
            self._find_name = ""
            self.find_hit.setText("")
            self.redraw()
            return
        ids = [i for i, v in st.kernels.items() if v[0] == name]
        if not ids:
            self.find_hit.setText("no such kernel")
            self.find_hit.setStyleSheet(f"color:{MISS_X};font-size:{_s(11)}px;")
            return
        if name != self._find_name:
            self._find_name, self._find_cursor = name, self.lo - 1
        start = max(self._find_cursor + 1, 0)
        hit = self._find_slot_with(st, ids, start, st.n_slots)
        wrapped = False
        if hit is None:
            hit = self._find_slot_with(st, ids, 0, start)
            wrapped = hit is not None
        if hit is None:
            self.find_hit.setText("not found in this capture")
            return
        self._find_cursor = hit
        self._find_ids = set(ids)
        span = max(1, self.hi - self.lo)
        self.set_window(hit, min(hit + span, st.n_slots - 1))
        sl = st.slot(hit)
        sfn = f"SFN {sl[3]}.{sl[4]}" if sl and sl[3] is not None else f"slot {hit}"
        n_here = getattr(self, "_n_found", 0)
        self.find_hit.setText(
            f"▸ {sfn}   {n_here} in view"
            + ("   (wrapped)" if wrapped else ""))
        self.find_hit.setStyleSheet(
            f"color:{FIND_FILL};font-size:{_s(11)}px;font-weight:600;")
        if self.window():
            self.window().statusBar().showMessage(
                f"{name}: found at {sfn} (slot index {hit})", 6000)

    def _find_slot_with(self, st, ids, a, b):
        """First slot in [a,b) whose window contains one of `ids`.

        Scanned in blocks rather than slot-by-slot: one windowed query per
        block instead of per slot, which keeps a wrap-around search over a
        300k-slot capture interactive.
        """
        BLK = 256
        idset = set(ids)
        i = a
        while i < b:
            j = min(i + BLK, b)
            s_a, s_b = st.slot(i), st.slot(j - 1)
            if not s_a or not s_b:
                i = j
                continue
            rows = st.events_in_window(s_a[1], s_b[2])
            hits = [r for r in rows if r[2] in idset]
            if hits:
                t = min(r[0] for r in hits)
                for k in range(i, j):
                    sk = st.slot(k)
                    if sk and sk[1] <= t <= sk[2]:
                        return k
                return i
            i = j
        return None

    # -- TTI-miss marking ---------------------------------------------------
    def _mark_misses(self, st, t0, t1, us, ymax):
        """Shade slots the DU dropped, from the nvlog TTI-MISS markers.

        A dropped TTI has no kernels by definition, so without this the lane is
        simply empty there and an empty stretch is indistinguishable from an
        idle one. The band makes the difference explicit.
        """
        if not st or not hasattr(st, "tti_miss_in"):
            return 0
        rows = st.tti_miss_in(t0, t1)
        if not rows:
            return 0
        mx, my = [], []
        for (a, b, kind, sfn, slot, dt_ns, site) in rows:
            col = "#B3261E" if kind == "cleanup" else "#C97B1E"
            reg = pg.LinearRegionItem(
                values=(us(max(a, t0)), us(min(b, t1))),
                brush=pg.mkBrush(col + "24"),
                pen=pg.mkPen(col, width=1, style=Qt.PenStyle.DashLine),
                movable=False)
            reg.setZValue(-40)
            self.p_kern.addItem(reg)
            # No text block. It repeated the SFN that is already printed on
            # the boundary just above it, in a filled panel that read as one
            # more launched bar - and the slot's own caption now says "dropped"
            # by being red and bold. The cross, the band and the header count
            # carry the rest.
            txt = None
            # Same reasoning as the SFN captions: anchored to a line so it
            # cannot be drawn off the top of the chart.
            self.p_kern.addItem(pg.InfiniteLine(
                pos=us(max(a, t0)), angle=90,
                pen=pg.mkPen(col, width=1, style=Qt.PenStyle.DashLine),
                label=txt,
                labelOpts=({"position": 0.90, "color": col,
                            "anchors": [(0, 0), (1, 0)],
                            "fill": pg.mkBrush(20, 22, 26, 210)}
                           if txt else None)))
            # Sit the cross ON the slot boundary, which is where that slot's
            # SFN caption is drawn - so the mark and the number it refers to
            # read as one annotation. At mid-slot the two drifted apart as soon
            # as the window widened and the eye had to pair them by guesswork.
            mx.append(us(max(a, t0)))
            my.append(ymax + MISS_Y)
        if mx:
            # A cross at the top of the slot. The shaded band alone is easy to
            # miss when the window is wide and the miss is one slot of forty,
            # and a dropped TTI has no bars of its own to draw attention.
            x = pg.ScatterPlotItem(x=mx, y=my, symbol="x", size=_s(18),
                                   pen=pg.mkPen(MISS_X, width=3), brush=None)
            x.setZValue(50)
            self.p_kern.addItem(x)
        return len(rows)

    # -- PNG export ---------------------------------------------------------
    def export_pdf(self):
        """Vector PDF of the kernel lanes, white ground, for LaTeX."""
        if not self.store:
            return
        # The wall-clock axis is a reading aid for the live view: it says where
        # in the capture you are, and lets a time be grepped in the nvlog. On a
        # figure it is noise - the caption already carries the SFN span, and an
        # absolute UTC stamp means nothing to a reader who does not have the
        # run. Hidden for the render only, restored afterwards.
        top = self.p_kern.getAxis("top")
        was_shown = top.isVisible()
        self.p_kern.hideAxis("top")

        # Give the lane stack enough height that the channel names can take the
        # full font boost. This changes the figure's ASPECT, not its content:
        # the time window, the lanes and every bar are exactly what is on
        # screen, drawn taller.
        left = self.p_kern.getAxis("left")
        lanes = sum(len(x) for x in (left._tickLevels or [])) or 1
        need = lanes * _s(PDF_LANE_PITCH)
        w0, h0 = self.gl.width(), self.gl.height()
        fixed0 = getattr(self, "_chart_fixed", False)
        grow = max(0, int(need - left.height()))
        if grow:
            self._set_canvas(w0, h0 + grow)
            QApplication.processEvents()
        try:
            export_plot_pdf(self, self.gl, self.p_kern, "tti_kernel_lanes.pdf",
                            self.hdr.text())
        finally:
            if grow:
                if fixed0:
                    self._set_canvas(w0, h0)
                else:
                    self._grip_reset()
            if was_shown:
                self.p_kern.showAxis("top")
class TTICompare(QWidget):
    """The TTI tab: the primary viewer, and a second one below it.

    The secondary viewer owns an independent SessionStore, so two captures -
    a clean run and a starved one, say - can be read against each other in one
    window. It is off by default and costs nothing until enabled: no store is
    opened, and the widget is not constructed until first use.

    Deliberately scoped to this tab. The Timeline, Statistics, Channels and
    Kernels tabs continue to follow the single session chosen in the Sessions
    panel, and nothing here touches them.
    """

    def __init__(self, primary: "TTIView"):
        super().__init__()
        self.primary = primary
        self.secondary = None
        self._sec_store = None

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)
        self.chk_cmp = QCheckBox("Compare with a second capture")
        self.chk_cmp.setStyleSheet(f"color:{FG};font-size:{_s(11)}px;")
        self.chk_cmp.toggled.connect(self._toggle)
        bar.addWidget(self.chk_cmp)
        self.btn_load = QPushButton("Load second capture…")
        self.btn_load.setStyleSheet(_style_button())
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self.load_secondary)
        bar.addWidget(self.btn_load)
        self.chk_lock = QCheckBox("Lock windows together")
        self.chk_lock.setStyleSheet(f"color:{FG};font-size:{_s(11)}px;")
        self.chk_lock.setChecked(True)
        self.chk_lock.setEnabled(False)
        bar.addWidget(self.chk_lock)
        self.sec_lbl = _lbl("", 11, MUTED)
        bar.addWidget(self.sec_lbl)
        bar.addStretch(1)
        v.addLayout(bar)

        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.addWidget(primary)
        v.addWidget(self.split, 1)

    # -- secondary lifecycle ------------------------------------------------
    def _toggle(self, on):
        self.btn_load.setEnabled(on)
        self.chk_lock.setEnabled(on)
        if on and self.secondary is None:
            self.secondary = TTIView()
            self.secondary.hdr.setText("secondary")
            # The secondary holds its own store, so its Window pane addresses
            # its own slots directly rather than going through the primary's
            # timeline - which indexes a different capture entirely.
            self.secondary.jump_requested.connect(
                lambda lo, hi: self.secondary.set_window(lo, hi))
            self.split.addWidget(self.secondary)
            self.split.setSizes([_s(430), _s(430)])
            # Window changes propagate primary -> secondary only. Two-way
            # binding fights itself when the captures differ in length.
            self.primary.on_window_changed = self._mirror
        if self.secondary is not None:
            self.secondary.setVisible(on)
        if not on:
            self.sec_lbl.setText("")

    def _mirror(self, lo, hi):
        if (self.secondary is not None and self.secondary.isVisible()
                and self.chk_lock.isChecked() and self._sec_store):
            n = self._sec_store.n_slots
            self.secondary.set_window(min(lo, n - 1), min(hi, n - 1))

    def load_secondary(self):
        """Open a second built session for the lower viewer.

        Takes a built .ttistore rather than a raw capture directory: building
        is minutes of work and belongs in the Sessions panel, not behind a
        button that looks instant.
        """
        d = Path.home()
        path, _f = QFileDialog.getOpenFileName(
            self, "Open a built session for the secondary viewer",
            str(d), "TTI-Scope store (*.ttistore *.sqlite);;All files (*)")
        if not path:
            return
        try:
            from ttiscope_ingest import SessionStore
            st = SessionStore(Path(path))
        except Exception as e:
            QMessageBox.warning(self, APP,
                                f"Could not open that store:\n{e}")
            return
        self._sec_store = st
        self.secondary.set_store(st)
        self.secondary.set_window(0, min(7, max(0, st.n_slots - 1)))
        nm = Path(path).name
        miss = len(st.tti_miss_all()) if hasattr(st, "tti_miss_all") else 0
        self.sec_lbl.setText(f"{nm}  ·  {st.n_slots:,} slots"
                             + (f"  ·  {miss:,} TTI-miss" if miss else ""))


class _SortItem(QTableWidgetItem):
    """Cell that sorts by its VALUE, not its rendering.

    Qt's default compares DisplayRole as text, which puts "9,912" above
    "69,092" and makes every count and duration column meaningless once the
    user clicks a header. Kernels with no duration sort to the bottom rather
    than the top, since "unmeasured" is not "fastest".
    """
    def __lt__(self, other):
        a = self.data(Qt.ItemDataRole.UserRole)
        b = other.data(Qt.ItemDataRole.UserRole)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


def _num(text: str, value) -> QTableWidgetItem:
    it = _SortItem()
    it.setData(Qt.ItemDataRole.DisplayRole, text)
    it.setData(Qt.ItemDataRole.UserRole, -1.0 if value is None else float(value))
    it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
    return it


def _table(headers) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setSortingEnabled(True)
    t.setStyleSheet(_style_table())
    t.restyle = lambda t=t: t.setStyleSheet(_style_table())
    t.verticalHeader().setVisible(False)
    t.setAlternatingRowColors(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.horizontalHeader().setSectionResizeMode(
        QHeaderView.ResizeMode.Interactive)
    t.horizontalHeader().setStretchLastSection(False)
    return t


def _fit_columns(t: QTableWidget, cap: int = 460):
    """Size every column to its content, capped.

    ResizeToContents alone lets one long cell — a kernel role, a metric
    definition — push the table wider than the window; stretchLastSection made
    the opposite mistake and ran the final column to the right edge whatever
    was in it.
    """
    t.resizeColumnsToContents()
    for c in range(t.columnCount()):
        t.setColumnWidth(c, min(t.columnWidth(c) + _s(10), _s(cap)))
    return t


class ChannelPanel(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.hdr = _lbl("", 11, MUTED)
        v.addWidget(self.hdr)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "% of warp-time in window")
        self.plot.showGrid(x=True, y=False, alpha=0.16)
        self.plot.setMouseEnabled(y=False)
        self.plot.getAxis("left").setWidth(_s(80))
        self.plot.setMaximumHeight(_s(280))
        self.restyle = lambda: (self.plot.getAxis("left").setWidth(_s(80)),
                                self.plot.setMaximumHeight(_s(280)))
        v.addWidget(self.plot)
        self.tbl = _table(["kernel", "channel", "source", "launches",
                           "busy (us)", "warp-time %"])
        v.addWidget(self.tbl, 1)
        self.store = None

    def set_store(self, st):
        self.store = st

    def set_window(self, lo, hi):
        st = self.store
        self.plot.clear()
        self.tbl.setRowCount(0)
        if not st:
            return
        rows = st.channel_histogram(lo, hi)
        tot = sum(r[3] or 0 for r in rows) or 1
        rows = sorted(rows, key=lambda r: r[3] or 0)
        self.plot.addItem(pg.BarGraphItem(
            x0=0, width=[100.0 * (r[3] or 0) / tot for r in rows],
            y=list(range(len(rows))), height=0.62,
            brushes=[_q(channel_color(r[0]), 210) for r in rows],
            pens=[pg.mkPen(None)] * len(rows)))
        self.plot.getAxis("left").setTicks(
            [[(i, r[0]) for i, r in enumerate(rows)]])
        self.plot.setYRange(-0.6, max(0, len(rows) - 0.4), padding=0)

        kr = st.kernel_histogram(lo, hi)
        ktot = sum(r[4] or 0 for r in kr) or 1
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(kr))
        for i, (name, chan, n, busy, warp, src) in enumerate(kr):
            it = QTableWidgetItem(name)
            self.tbl.setItem(i, 0, it)
            it = QTableWidgetItem(chan)
            it.setForeground(_q(channel_color(chan)))
            self.tbl.setItem(i, 1, it)
            it = QTableWidgetItem("KTRACE" if src else "nsys")
            if src:
                it.setForeground(_q("#E07A5F"))
            self.tbl.setItem(i, 2, it)
            self.tbl.setItem(i, 3, _num(f"{n:,}", n))
            self.tbl.setItem(i, 4, _num(f"{(busy or 0)/1000:,.1f}", busy or 0))
            pct = 100.0 * (warp or 0) / ktot
            self.tbl.setItem(i, 5, _num(f"{pct:.2f}", pct))
        self.tbl.setSortingEnabled(True)
        self.tbl.sortItems(5, Qt.SortOrder.DescendingOrder)
        _fit_columns(self.tbl)
        self.hdr.setText(
            f"{st.slot_label(lo)} → {st.slot_label(hi)}   ·   "
            f"{len(kr)} kernel/source rows   ·   warp-time shares sum to 100%")



# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

class LLMPanel(QWidget):
    """An LLM inference trace, in the same terms as the TTI lanes.

    Independent of the Sessions library on purpose. An Aerial capture and a
    vLLM capture are two different runs of two different stacks; pretending
    they are one session would invite windows to be compared that were never
    concurrent. This tab opens its own nsys SQLite and says so.
    """

    PHASE_COLOR = {"prefill": "#E8544F", "decode": "#4FA3E8"}

    def __init__(self):
        super().__init__()
        self.trace = None
        self.lo = self.hi = 0                    # step indices
        self._lanes = []
        self._for_print = False
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        bar = QHBoxLayout()
        self.btn_open = QPushButton("Load nsys SQLite\u2026")
        self.btn_open.setStyleSheet(_style_button())
        self.btn_open.setToolTip(
            "An nsys report exported to SQLite:\n"
            "    nsys export --type sqlite run.nsys-rep")
        self.btn_open.clicked.connect(self.load)
        bar.addWidget(self.btn_open)

        bar.addSpacing(_s(12))
        bar.addWidget(_lbl("Step", 12, LABEL, bold=True))
        self.spin_step = QSpinBox()
        self.spin_step.setRange(0, 0)
        self.spin_step.setStyleSheet(_style_spin())
        self.spin_step.valueChanged.connect(lambda _v: self.redraw())
        bar.addWidget(self.spin_step)
        bar.addWidget(_lbl("span", 12, LABEL))
        self.spin_span = QSpinBox()
        self.spin_span.setRange(1, 5000)
        self.spin_span.setValue(12)
        self.spin_span.setStyleSheet(_style_spin())
        self.spin_span.valueChanged.connect(lambda _v: self.redraw())
        bar.addWidget(self.spin_span)
        bar.addWidget(_lbl("steps", 12, LABEL))

        self.btn_first_pre = QPushButton("\u25c2 prefill")
        self.btn_first_pre.setStyleSheet(_style_button())
        self.btn_first_pre.setToolTip("Jump to the next prefill step")
        self.btn_first_pre.clicked.connect(self.next_prefill)
        bar.addWidget(self.btn_first_pre)

        bar.addStretch(1)
        self.btn_pdf = QPushButton("Export PDF")
        self.btn_pdf.setStyleSheet(_style_button())
        self.btn_pdf.clicked.connect(self.export_pdf)
        bar.addWidget(self.btn_pdf)
        v.addLayout(bar)

        self.hdr = _lbl("No LLM trace loaded.", 11, MUTED)
        self.hdr.setWordWrap(True)
        v.addWidget(self.hdr)

        self.gl = pg.GraphicsLayoutWidget()
        self.p = self.gl.addPlot(row=0, col=0)
        self.p.setTitle("LLM kernels", color=MUTED, size="9pt")
        self.p.showGrid(x=True, y=False, alpha=0.14)
        self.p.setMouseEnabled(y=False)
        self.p.getAxis("left").setWidth(_s(170))
        self.p.getAxis("left").setStyle(tickFont=_tick_font())
        self.p.setLabel("bottom", "milliseconds within window")
        v.addWidget(self.gl, 1)

        # "busy %" sums overlapping kernels, so it can exceed 100 - that is
        # concurrency, not utilisation. Named so it cannot be read as the
        # fraction of the window the GPU was occupied.
        self.tbl = _table(["family", "launches", "busy ms",
                           "busy % (sums overlap)"])
        self.tbl.setMaximumHeight(_s(190))
        v.addWidget(self.tbl)

        self.note = _lbl("", 10, MUTED)
        self.note.setWordWrap(True)
        v.addWidget(self.note)

    def restyle(self):
        for b in (self.btn_open, self.btn_pdf, self.btn_first_pre):
            b.setStyleSheet(_style_button())
        for sp in (self.spin_step, self.spin_span):
            sp.setStyleSheet(_style_spin())
        self.p.getAxis("left").setStyle(tickFont=_tick_font())
        self.redraw()

    # -- loading -------------------------------------------------------------
    def load(self):
        path, _f = QFileDialog.getOpenFileName(
            self, "Open an nsys trace of an LLM run", str(Path.home()),
            "nsys trace (*.nsys-rep *.sqlite *.db);;All files (*)")
        if not path:
            return
        p = Path(path)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # A .nsys-rep is what a capture archives, and the only thing that
            # travels off the rig - the sqlite is derived, large, and stays
            # local. Export it here rather than making that a manual step.
            if p.suffix == ".nsys-rep":
                sq = p.with_suffix(".sqlite")
                if not sq.exists():
                    self.hdr.setText(
                        f"Exporting {p.name} to SQLite (one-time, a few "
                        f"minutes)\u2026")
                    QApplication.processEvents()
                    sq = export_nsys_rep(p, log=lambda m: None)
                    if not sq or not Path(sq).exists():
                        raise ValueError(
                            "nsys export failed. Is `nsys` on PATH?\n"
                            "Export by hand with:\n"
                            f"    nsys export --type sqlite -o {p.stem}.sqlite "
                            f"{p.name}")
                p = Path(sq)
            tr = LLMTrace(p)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, APP, f"Could not read that trace:\n\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        if self.trace:
            self.trace.close()
        self.trace = tr
        n = max(0, len(tr.steps) - 1)
        self.spin_step.setRange(0, n)
        # Open on the first prefill if there is one: it is the step the figure
        # is usually about, and it is one step among hundreds.
        first = next((s.idx for s in tr.steps if s.phase == "prefill"), 0)
        self.spin_step.setValue(first)
        self.redraw()

    def next_prefill(self):
        if not self.trace:
            return
        cur = self.spin_step.value()
        nxt = next((s.idx for s in self.trace.steps
                    if s.phase == "prefill" and s.idx > cur), None)
        if nxt is None:
            nxt = next((s.idx for s in self.trace.steps
                        if s.phase == "prefill"), None)
        if nxt is not None:
            self.spin_step.setValue(nxt)

    # -- drawing -------------------------------------------------------------
    def redraw(self):
        self.p.clear()
        tr = self.trace
        if not tr or not tr.steps:
            return
        lo = max(0, min(self.spin_step.value(), len(tr.steps) - 1))
        hi = min(len(tr.steps) - 1, lo + self.spin_span.value() - 1)
        self.lo, self.hi = lo, hi
        t0, t1 = tr.steps[lo].t0, tr.steps[hi].t1
        ms = lambda t: (t - t0) / 1e6

        rows = tr.kernels_in(t0, t1)
        self._rows = rows
        # Only the families this window actually contains get a lane. A fixed
        # fifteen-lane axis on a decode step is thirteen empty rows.
        present = [f for f in FAMILY_ORDER if any(r[2] == f for r in rows)]
        self._lanes = present
        idx = {f: i for i, f in enumerate(present)}
        self.p.getAxis("left").setTicks(
            [[(i, f) for f, i in idx.items()]])
        self.p.setYRange(-0.6, max(0.4, len(present) - 0.4), padding=0)

        # phase bands behind everything
        for phase, a, b, k in tr.phase_runs():
            if b < t0 or a > t1:
                continue
            reg = pg.LinearRegionItem(
                values=(ms(max(a, t0)), ms(min(b, t1))),
                brush=_q(self.PHASE_COLOR.get(phase, MUTED), 26),
                pen=pg.mkPen(None), movable=False)
            reg.setZValue(-60)
            self.p.addItem(reg)

        # step boundaries; a phase change is drawn heavier and labelled
        prev = None
        for s in tr.steps[lo:hi + 1]:
            change = prev is not None and s.phase != prev
            if not change and self._for_print:
                prev = s.phase          # every-step rules are clutter in print
                continue
            pen = (pg.mkPen(self.PHASE_COLOR.get(s.phase, MUTED),
                            width=2, style=Qt.PenStyle.DashLine)
                   if change else
                   pg.mkPen(GRID, width=1, style=Qt.PenStyle.DotLine))
            self.p.addItem(pg.InfiniteLine(
                pos=ms(s.t0), angle=90, pen=pen,
                label=(f"{s.phase}" if change or s.idx == lo else None),
                labelOpts=({"position": 0.97,
                            "color": self.PHASE_COLOR.get(s.phase, MUTED),
                            "anchors": [(0, 0), (1, 0)]}
                           if change or s.idx == lo else None)))
            prev = s.phase

        # kernels
        for fam in present:
            xs = [(ms(a), max(ms(b) - ms(a), (t1 - t0) / 1e6 * 0.0006))
                  for a, b, f in rows if f == fam]
            if not xs:
                continue
            y = idx[fam]
            self.p.addItem(pg.BarGraphItem(
                x0=[a for a, _w in xs], width=[w for _a, w in xs],
                y0=y - 0.34, height=0.68,
                brush=_q(FAMILY_COLOR.get(fam, MUTED), 220),
                pen=pg.mkPen(None)))

        self.p.setXRange(0, ms(t1), padding=0.01)
        self._fill_table(rows, t0, t1)
        self._fill_header(tr, lo, hi, t0, t1, len(rows))

    def _fill_table(self, rows, t0, t1):
        # Aggregated from the SAME rows the plot drew. Querying again risked
        # the table describing a different set than the one on screen whenever
        # the draw limit bit.
        agg = {}
        for a, b, fam in rows:
            n, busy = agg.get(fam, (0, 0))
            agg[fam] = (n + 1, busy + max(0, min(b, t1) - max(a, t0)))
        tot = sorted(((f, n, d) for f, (n, d) in agg.items()),
                     key=lambda r: -r[2])
        span = max(1, t1 - t0)
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(tot))
        for i, (fam, n, busy) in enumerate(tot):
            it = QTableWidgetItem(fam)
            it.setForeground(_q(FAMILY_COLOR.get(fam, MUTED)))
            self.tbl.setItem(i, 0, it)
            self.tbl.setItem(i, 1, _num(f"{n:,}", n))
            self.tbl.setItem(i, 2, _num(f"{busy/1e6:.3f}", busy))
            pct = 100.0 * busy / span
            self.tbl.setItem(i, 3, _num(f"{pct:.1f}", pct))
        self.tbl.setSortingEnabled(True)
        _fit_columns(self.tbl)

    def _fill_header(self, tr, lo, hi, t0, t1, n_drawn):
        s = tr.summary()
        pre_ms = s["prefill_med_ns"] / 1e6
        dec_ms = s["decode_med_ns"] / 1e6
        ratio = (f"{pre_ms/dec_ms:.0f}x" if dec_ms else "n/a")
        self.hdr.setText(
            f"{tr.path.name}   \u00b7   {s['n_kernels']:,} kernels   "
            f"\u00b7   {s['n_steps']:,} steps over {s['span_ns']/1e9:.2f} s   "
            f"\u00b7   {s['n_prefill']:,} prefill / {s['n_decode']:,} decode   "
            f"\u00b7   median step: prefill {pre_ms:.2f} ms, decode "
            f"{dec_ms:.2f} ms ({ratio})")
        note = ["Phase provenance: " + tr.provenance + "."]
        note += tr.warnings
        note.append(
            f"Window: steps {lo}\u2013{hi}, {(t1-t0)/1e6:.2f} ms, "
            f"{n_drawn:,} kernels drawn.")
        self.note.setText("  ".join(note))

    # -- export --------------------------------------------------------------
    def export_pdf(self):
        if not self.trace:
            return
        self._for_print = True
        self.redraw()
        QApplication.processEvents()
        try:
            export_plot_pdf(self, self.gl, self.p, "llm_kernel_timeline.pdf",
                            self.hdr.text())
        finally:
            self._for_print = False
            self.redraw()


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _pct(sorted_vals, q: float):
    """Nearest-rank percentile on an already-sorted list.

    Nearest-rank rather than interpolated: every value here is a real measured
    slot, and P99 of 1,000 slots should name one of them, not a number that
    never occurred.
    """
    if not sorted_vals:
        return None
    i = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(i, len(sorted_vals) - 1))]


def _summary(vals):
    """(n, min, max, mean, median, p95, p99) — None-safe."""
    v = sorted(x for x in vals if x is not None)
    if not v:
        return (0, None, None, None, None, None, None)
    return (len(v), v[0], v[-1], sum(v) / len(v), _pct(v, .50),
            _pct(v, .95), _pct(v, .99))


class CpuIntraTtiPlot(pg.PlotWidget):
    """Where inside the 500 us TTI the CPU is busy, broken down by thread role.

    The GPU plot on the right answers this for the device; this is its
    counterpart for the host, folded from the sweep's own `perf record`
    samples onto slot phase.

    How the fold is possible at all: this kernel refuses CLOCK_REALTIME,
    CLOCK_TAI and CLOCK_BOOTTIME for perf events, so the sweep records
    CLOCK_MONOTONIC and writes paired monotonic/realtime/TAI readings before,
    during and after the capture. Measured spread within a reading group is
    112-176 ns and drift across a 45 s capture was 0 ns - three orders of
    magnitude finer than the 500 us slot it has to resolve. The slot grid comes
    from the same run's nvlog (L2A.TICK_TIMES `tick=`), so both sides are on
    one clock.

    y is CORES BUSY, not a percentage: samples / (rate x bin x slots). That is
    comparable across bins, across captures and across sample rates, and it
    reads directly - "3.5 cores busy at 80 us into the slot".
    """

    def __init__(self):
        super().__init__()
        self.setLabel("bottom", "position within the TTI slot (us)")
        self.setLabel("left", "CPU cores busy")
        self.showGrid(x=False, y=False)
        self.setMouseEnabled(x=False, y=False)
        self.profile = None
        self._items = []
        self._title = ""
        self.scene().sigMouseClicked.connect(lambda _e: self.export_pdf())

    def restyle(self):
        for ax in ("left", "bottom"):
            self.getAxis(ax).setStyle(tickFont=_tick_font())

    def set_profile(self, prof, title=""):
        self.profile, self._title = prof, title
        self.clear()
        self._items = []
        if not prof:
            return
        edges = np.array(prof["edges_us"], dtype=float)
        centres = (edges[:-1] + edges[1:]) / 2.0
        xs = np.concatenate(([edges[0]], centres, [edges[-1]]))

        def pad(v):
            v = np.asarray(v, dtype=float)
            return np.concatenate(([v[0]], v, [v[-1]]))

        # Largest contributors at the bottom of the stack, so the band a reader
        # cares about is the one anchored to the axis and easiest to judge.
        roles = sorted(prof["roles"], key=lambda r: -sum(prof["roles"][r]))
        cum = np.zeros(len(xs))
        for r in roles:
            lower = pg.PlotCurveItem(xs, cum.copy(), pen=pg.mkPen(None))
            cum = cum + pad(prof["roles"][r])
            upper = pg.PlotCurveItem(xs, cum.copy(), pen=pg.mkPen(None))
            col = ROLE_COLOR.get(r, ROLE_COLOR["other"])
            fill = pg.FillBetweenItem(lower, upper, brush=_q(col, 170))
            self.addItem(fill)
            self._items.append((fill, r, col))

        total = pg.PlotCurveItem(xs, cum, pen=pg.mkPen("#E6E8EA", width=2.5))
        self.addItem(total)
        peak = float(max(cum)) or 1.0
        self.setXRange(edges[0], edges[-1], padding=0.01)
        self.setYRange(0, peak * 1.42, padding=0)

        # A legend the exported figure carries with it, laid out over two rows.
        # Eight role names on one row collide in a docked panel; only the roles
        # that carry real time are worth naming, and the rest are visible in
        # the stack anyway.
        share = {r: sum(prof["roles"][r]) for r in roles}
        tot = sum(share.values()) or 1.0
        named = [r for r in roles if share[r] / tot >= 0.02][:6]
        per_row = 3
        span = edges[-1] - edges[0]
        for i, r in enumerate(named):
            col = ROLE_COLOR.get(r, ROLE_COLOR["other"])
            t = pg.TextItem(f"{r}  {100*share[r]/tot:.0f}%", color=col,
                            anchor=(0, 0.5))
            t.setPos(edges[0] + (i % per_row) * span / per_row,
                     peak * (1.34 if i < per_row else 1.18))
            self.addItem(t)

    def export_pdf(self):
        """Vector PDF on white, for LaTeX."""
        export_plot_pdf(self, self, self.plotItem, "tti_intra_slot_cpu.pdf")
class CpuUtilPlot(pg.PlotWidget):
    """Per-core CPU utilisation - the fallback when a capture has no clock
    anchor and therefore cannot be folded into TTI slots."""

    MIN_PCT = 0.4
    MAX_ROWS = 16

    def __init__(self):
        super().__init__()
        self.setLabel("bottom", "core utilisation over the traffic window (%)")
        self.showGrid(x=True, y=False, alpha=0.16)
        self.setMouseEnabled(y=False)
        self.getAxis("left").setWidth(_s(150))
        self.profile = None
        self.n_hidden = 0

    def restyle(self):
        self.getAxis("left").setWidth(_s(150))
        for ax in ("left", "bottom"):
            self.getAxis(ax).setStyle(tickFont=_tick_font())

    def set_profile(self, prof):
        self.profile = prof
        self.clear()
        if not prof or not prof.cores:
            return
        cores = [c for c in prof.cores if c["pct"] >= self.MIN_PCT]
        cores.sort(key=lambda c: -c["pct"])
        self.n_hidden = max(0, len(cores) - self.MAX_ROWS)
        cores = cores[:self.MAX_ROWS][::-1]
        if not cores:
            return
        ys = list(range(len(cores)))
        self.addItem(pg.BarGraphItem(
            x0=0, width=[c["pct"] for c in cores], y=ys, height=0.66,
            brushes=[_q(ROLE_COLOR.get(c["role"], ROLE_COLOR["other"]), 210)
                     for c in cores],
            pens=[pg.mkPen(None)] * len(cores)))
        ticks = []
        for i, c in enumerate(cores):
            top = c["top"][0]["comm"] if c["top"] else ""
            ticks.append((i, f"cpu{c['cpu']}  {top[:18]}"))
        self.getAxis("left").setTicks([ticks])
        self.setYRange(-0.7, len(cores) - 0.3, padding=0)
        self.setXRange(0, max(100.0, max(c["pct"] for c in cores) * 1.05),
                       padding=0)


class IntraSlotPlot(pg.GraphicsLayoutWidget):
    """Where inside the 500 us TTI the work lands.

    Every other view answers "which slot was busy". This one folds a thousand
    slots on top of each other and answers "which part OF a slot is busy" — the
    question that decides whether headroom exists within the deadline or only
    between deadlines.

    Deliberately spare, because it is a figure before it is a UI panel:

      main row   theoretical SM occupancy as one thick line over a light-green
                 fill (left axis, %), and energy as one bold line (right axis,
                 uJ). No grid, no uncertainty bands, no annotations — those
                 read as data and there is already enough of it.
      band       concurrent kernels per window as a slim box-and-whisker
                 low in the same frame, each printing its own numbers, so no
                 third axis is needed to read it.
    """

    # Every row of the Statistics table above, folded into the slot the same
    # way. (label, unit, profile key, fixed axis maximum or None)
    METRICS = [
        ("Theoretical SM occupancy", "%",  "occ",         100.0),
        ("GPU utilisation",          "%",  "util_mean",   100.0),
        ("GPU idle time",            "µs", "idle_mean",   None),
        ("Warp demand",              "%",  "demand_mean", None),
        ("Kernel launches",          "",   "launch_mean", None),
        ("Energy",                   "µJ", "e_p95",       None),
    ]

    OCC = "#3F9D7C"
    OCC_FILL = "#8FD8BC"
    ENERGY = "#D4553A"
    # The concurrency band was grey on a dark ground, which read as furniture
    # rather than as data. Blue separates it from the green occupancy curve and
    # the red energy curve, and holds up on the white PDF ground too.
    BOX = "#2E9BFF"
    BOX_W = 2.4               # spine and box outline
    BOX_MED_W = 4.2           # median bar, the value being read

    def __init__(self):
        super().__init__()
        self.ci.layout.setSpacing(2)

        self.p = self.addPlot(row=0, col=0)
        self.p.setLabel("left", "theoretical SM occupancy (%)", color=self.OCC)
        # A faint y grid only. The x positions are already marked by the bin
        # edges and the boxes; a full grid behind a filled area is noise.
        self.p.showGrid(x=False, y=True, alpha=0.12)
        for _side in ("left", "bottom", "right"):
            _ax = self.p.getAxis(_side)
            _ax.setStyle(tickLength=-5, tickTextOffset=6)
            _ax.setPen(pg.mkPen(MUTED, width=1))
        self.p.setMouseEnabled(x=False, y=False)
        self.p.setYRange(0, 100, padding=0)
        self.p.getAxis("bottom").setStyle(showValues=False)
        self.p.hideButtons()

        # Energy shares the frame on its own scale; a second ViewBox rather
        # than a normalised overlay, so the microjoule numbers stay readable.
        self.vb_e = pg.ViewBox()
        self.p.showAxis("right")
        self.p.scene().addItem(self.vb_e)
        self.p.getAxis("right").linkToView(self.vb_e)
        self.vb_e.setXLink(self.p)
        self.vb_e.setMouseEnabled(x=False, y=False)
        self.p.setLabel("right", "energy per 100 µs window (µJ)",
                        color=self.ENERGY)
        self.p.vb.sigResized.connect(self._sync)

        # Below the axes, in its own layout row, laid out horizontally. Inside
        # the plot there is nowhere for it to go: the filled curve occupies the
        # upper band and the boxes the lower one, so an in-plot legend covers
        # data whichever corner it is pinned to.
        self.legend = pg.LegendItem(horSpacing=22, colCount=3,
                                    labelTextSize="9pt", labelTextColor=FG,
                                    brush=pg.mkBrush(0, 0, 0, 0),
                                    pen=pg.mkPen(None))
        self.addItem(self.legend, row=1, col=0)
        self.ci.layout.setRowStretchFactor(0, 1)
        self.ci.layout.setRowStretchFactor(1, 0)
        self.curve_occ = pg.PlotCurveItem(
            pen=pg.mkPen(self.OCC, width=3.5),
            brush=_q(self.OCC_FILL, 90), fillLevel=0)
        self.p.addItem(self.curve_occ)
        self.curve_e = pg.PlotCurveItem(pen=pg.mkPen(self.ENERGY, width=3.5))
        self.vb_e.addItem(self.curve_e)

        self.p.setLabel("bottom", "position within the TTI slot (µs)")
        self.p.getAxis("bottom").setStyle(showValues=True)

        # ── concurrency, as a band INSIDE the main frame ─────────────────
        # No third axis: the box-and-whisker occupies a fixed band low in the
        # plot and every window prints its own numbers, so the glyph is read
        # against its labels rather than against a scale.
        self.box = pg.BarGraphItem(x0=[], x1=[], y0=[], y1=[],
                                   brush=_q(self.BOX, 150),
                                   pen=pg.mkPen(self.BOX, width=self.BOX_W))
        self.p.addItem(self.box)
        self.whisk = pg.PlotCurveItem(
            pen=pg.mkPen(self.BOX, width=self.BOX_W), connect="pairs")
        self.p.addItem(self.whisk)
        self.medline = pg.PlotCurveItem(
            pen=pg.mkPen(self.BOX, width=self.BOX_MED_W), connect="pairs")
        self.p.addItem(self.medline)
        self._notes = []

        # BarGraphItem carries no legend sample, so the band is represented by
        # a zero-length curve drawn with its pen. Same trick the TTI lanes use.
        self._leg_band = pg.PlotCurveItem(
            pen=pg.mkPen(self.BOX, width=self.BOX_MED_W))
        self._profile = None
        self._title = ""
        self._metric = 0
        # Overlays are the user's choice now. Tying them to "is the metric the
        # default one" meant the only way to see warp demand against energy was
        # not to ask for warp demand.
        self._show_energy = True
        self._show_band = True
        self._sync()
        self.scene().sigMouseClicked.connect(lambda _e: self.export_pdf())

    def _sync(self):
        self.vb_e.setGeometry(self.p.vb.sceneBoundingRect())
        self.vb_e.linkedViewChanged(self.p.vb, self.vb_e.XAxis)

    def restyle(self):
        for ax in ("left", "bottom", "right"):
            a = self.p.axes.get(ax)
            if a:
                a["item"].setStyle(tickFont=_tick_font())

    def set_overlays(self, energy: bool, band: bool):
        self._show_energy = bool(energy)
        self._show_band = bool(band)
        if self._profile:
            self.set_profile(self._profile, self._title)

    def set_metric(self, i: int):
        """Choose which row of the Statistics table this plot folds."""
        self._metric = max(0, min(len(self.METRICS) - 1, int(i)))
        if self._profile:
            self.set_profile(self._profile, self._title)

    def set_profile(self, prof, title=""):
        self._profile, self._title = prof, title
        if title:
            self.p.setTitle(title, color=MUTED, size="9pt")
        if not prof:
            return

        edges = np.array(prof["edges_us"], dtype=float)
        centres = (edges[:-1] + edges[1:]) / 2.0

        # A line through the window centres rather than a step function. The
        # value IS a per-window mean, but drawn as steps the eye reads the
        # riser between windows as an instantaneous jump in the hardware, which
        # is not what a 100 us average says. Anchored at both slot edges so the
        # fill covers the whole TTI.
        xs = np.concatenate(([edges[0]], centres, [edges[-1]]))
        # Rebuilt every draw: which series are present depends on the metric
        # and the two overlay switches, and a stale legend is worse than none.
        try:
            self.legend.clear()
        except Exception:
            pass
        name, unit, key, fixed = self.METRICS[self._metric]
        main = np.array(prof.get(key) or [0.0] * len(centres), dtype=float)
        ys = np.concatenate(([main[0]], main, [main[-1]]))
        self.curve_occ.setData(xs, ys)
        self.p.setLabel("left", f"{name} ({unit})" if unit else name,
                        color=self.OCC)

        # Energy is the right-hand reference curve; drawing it there as well as
        # on the left, in two different scales, would be the same series twice.
        # Energy is never drawn twice: selecting it as the metric puts it on
        # the left axis, so the right-hand reference curve is redundant.
        e95 = np.array(prof["e_p95"], dtype=float)
        if not self._show_energy or key == "e_p95":
            self.curve_e.setData([], [])
            self.p.getAxis("right").setLabel("")
            self.p.hideAxis("right")
        else:
            self.p.showAxis("right")
            self.curve_e.setData(xs, np.concatenate(([e95[0]], e95, [e95[-1]])))
            self.p.setLabel("right", "energy per 100 µs window (µJ)",
                            color=self.ENERGY)

        self.legend.addItem(self.curve_occ, f"{name} ({unit})" if unit else name)
        if self.curve_e.xData is not None and len(self.curve_e.xData):
            self.legend.addItem(self.curve_e, "energy / 100 \u00b5s")

        # The axis has to follow the metric: a percentage tops out at 100, a
        # launch count or a microjoule figure does not, and pinning those to
        # 100 flattened them against the ceiling.
        ymax = fixed if fixed else (float(max(main)) * 1.2 if len(main) and
                                    max(main) > 0 else 1.0)
        # Explicit ticks on the bin edges. Auto-ticking put the last one at
        # 450 and dropped 500 - the end of the TTI, and the single most
        # important number on this axis, since the whole figure is about where
        # work lands inside a 500 us budget.
        _bot = self.p.getAxis("bottom")
        _step = 100.0 if edges[-1] >= 400 else max(1.0, edges[-1] / 5.0)
        _major, _t = [], 0.0
        while _t <= edges[-1] + 1e-6:
            _major.append((_t, f"{_t:.0f}"))
            _t += _step
        if abs(_major[-1][0] - edges[-1]) > 1e-6:
            _major.append((edges[-1], f"{edges[-1]:.0f}"))
        _bot.setTicks([_major,
                       [(x, "") for x in np.arange(0, edges[-1] + 1, _step / 2)
                        if all(abs(x - m[0]) > 1e-6 for m in _major)]])
        # A little padding at both ends. With none, the 0 and 500 labels are
        # centred exactly on the axis ends, overrun them, and pyqtgraph drops
        # both - leaving an axis that runs 100..400 and never names the slot
        # boundary the whole figure is about.
        self.p.setXRange(edges[0], edges[-1], padding=0.035)
        # Limits BEFORE range. The previous metric's ceiling is still in force
        # until it is replaced, and setYRange is clamped to it - which left
        # warp demand (peak 128%) drawn against an axis that still ended at
        # 34.8 from the idle-time metric, with the curve off the top.
        self.p.vb.setLimits(yMin=0, yMax=ymax)
        self.p.setYRange(0, ymax, padding=0)
        emax = max(e95) or 1.0
        self.vb_e.setYRange(0, emax * 1.15, padding=0)
        self._sync()

        # ── box-and-whisker: min / median / P95 / max per window ─────────
        for it in self._notes:
            self.p.removeItem(it)
        self._notes = []
        if not self._show_band:
            for it in (self.box, self.whisk, self.medline):
                it.setVisible(False)
            return
        for it in (self.box, self.whisk, self.medline):
            it.setVisible(True)

        c_min = np.array(prof["conc_min"], dtype=float)
        c_med = np.array(prof["conc_med"], dtype=float)
        c_p95 = np.array(prof["conc_p95"], dtype=float)
        c_max = np.array(prof["conc_max"], dtype=float)
        top = float(max(c_max)) if len(c_max) else 1.0
        top = top if top > 0 else 1.0

        # A band low in the frame, clear of the main curve. Expressed as a
        # FRACTION of the current axis, because that axis is no longer always
        # 0..100 - on a launch-count or energy metric the old fixed 14..38
        # would have sat off the top or on the floor.
        Y0, Y1 = 0.14 * ymax, 0.38 * ymax
        def m(v):
            return Y0 + (Y1 - Y0) * (np.asarray(v, dtype=float) / top)

        bw = (edges[1] - edges[0]) * 0.20          # slim
        cap = bw * 0.5
        self.box.setOpts(x0=centres - bw / 2, x1=centres + bw / 2,
                         y0=m(c_med), y1=np.maximum(m(c_p95), m(c_med) + 0.2))

        wx, wy = [], []
        for i, c in enumerate(centres):
            wx += [c, c]                                   # min-to-max spine
            wy += [m(c_min[i]), m(c_max[i])]
            wx += [c - cap / 2, c + cap / 2]               # min cap
            wy += [m(c_min[i]), m(c_min[i])]
            wx += [c - cap / 2, c + cap / 2]               # max cap
            wy += [m(c_max[i]), m(c_max[i])]
        self.whisk.setData(np.array(wx), np.array(wy))

        mx, my = [], []
        for i, c in enumerate(centres):
            mx += [c - bw / 2, c + bw / 2]
            my += [m(c_med[i]), m(c_med[i])]
        self.medline.setData(np.array(mx), np.array(my))

        # The median only. "1.7 (0.0-4.1)" under every box was four numbers per
        # 100 us of axis; enlarged for print they ran into each other and into
        # the neighbouring box. The spread is still drawn - it is the whisker -
        # and the legend says what the glyph means.
        for i, c in enumerate(centres):
            t = pg.TextItem(f"{c_med[i]:.1f}", color=self.BOX, anchor=(0.5, 0))
            t.setPos(c, Y0 - 0.055 * ymax)
            self.p.addItem(t)
            self._notes.append(t)
        self.legend.addItem(self._leg_band, "concurrent kernels")
        # No floating caption. It sat inside the data area, overlapped the
        # filled curve, and said in prose what the axis and the printed numbers
        # already say. What the band IS belongs in the figure caption.

    def export_pdf(self):
        """Vector PDF on white, for LaTeX.

        Rendered at a fixed figure geometry rather than at whatever size the
        Statistics pane happens to be. Two reasons: the enlarged print type
        needs more room than the on-screen pane has - at pane size the right
        axis and the legend were clipped at the canvas edge - and a figure
        whose proportions depend on the window is not reproducible.
        """
        if getattr(self, "_profile", None) is None:
            return
        with _PinnedScale(self.window()):
            self._export_pdf_pinned()

    def _export_pdf_pinned(self):
        # Unscaled on purpose: the canvas must not track the UI zoom, or the
        # figure's proportions change with the window. Wide enough that the
        # legend row never competes with the plot for width.
        self.setFixedSize(1420, 600)
        # invalidate(), not just activate(). On a second export the widget is
        # already at this size, so setFixedSize is a no-op, Qt emits no resize,
        # and without an explicit invalidate the graphics layout keeps the
        # geometry it had - which rendered the axes bunched into the corner.
        for _ in range(4):
            self.ci.layout.invalidate()
            self.ci.layout.activate()
            QApplication.processEvents()
        try:
            export_plot_pdf(self, self, self.p, "tti_intra_slot_profile.pdf",
                            self._title)
        finally:
            # Hand the widget back to its layout rather than to a remembered
            # size: the splitter owns it, and re-imposing the old numbers left
            # it pinned at the export geometry.
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)      # QWIDGETSIZE_MAX
            self.ci.layout.invalidate()
            QApplication.processEvents()


class StatsPanel(QWidget):
    """Per-slot distributions over a window of consecutive TTIs.

    Deliberately has its OWN window width, independent of the Timeline
    selection. The TTI tab wants ~5 slots to be readable; a distribution wants
    a thousand. Both are driven from the same centre, so what you see here is
    the neighbourhood of what you selected, at a statistically useful size.
    """

    METRICS = [
        ("GPU utilisation", "%",
         "union of kernel intervals / slot length"),
        ("GPU idle time", "us",
         "slot length - busy union; the complement of the row above"),
        ("Energy", "uJ",
         "ESTIMATED: P_idle x slot + sum of marginal kernel power x busy"),
        ("Theoretical SM occupancy", "%",
         "busy-time-weighted mean of per-launch occupancy in the slot"),
        ("Warp demand", "%",
         "warp-ns / (max resident warps x slot); >100% = oversubscribed"),
        ("Kernel launches", "",
         "per slot, both sources"),
    ]

    def __init__(self):
        super().__init__()
        self.store = None
        self.centre = 0
        self.sel = (0, 0)
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(6)

        bar = QHBoxLayout()
        bar.addWidget(_lbl("Window", 11, MUTED))
        self.spin = QSpinBox()
        self.spin.setRange(2, 200_000)
        self.spin.setValue(1000)
        self.spin.setSingleStep(100)
        self.spin.setSuffix("  TTI")
        self.spin.setStyleSheet(_style_spin())
        self.spin.valueChanged.connect(lambda _v: self.recompute())
        bar.addWidget(self.spin)
        bar.addWidget(_lbl("minimum; a wider TTI window selection wins",
                           10, MUTED))
        bar.addStretch(1)
        self.hdr = _lbl("", 11, MUTED)
        bar.addWidget(self.hdr)
        v.addLayout(bar)

        self.tbl = _table(["metric", "unit", "N", "min", "max", "mean",
                           "median", "P95", "P99", "definition"])
        self.tbl.setSortingEnabled(False)
        self.tbl.setMaximumHeight(_s(190))
        v.addWidget(self.tbl)

        self.plots = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(2)
        self.cpu_hdr = _lbl("", 10, MUTED)
        self.cpu_hdr.setWordWrap(True)
        lv.addWidget(self.cpu_hdr)
        # Two CPU views, stacked: the intra-TTI fold when the capture has a
        # clock anchor, the per-core bars when it does not. A capture from the
        # older sweeps genuinely cannot be folded, so it falls back rather than
        # showing an empty frame.
        self.cpu_intra = CpuIntraTtiPlot()
        self.cpu = CpuUtilPlot()
        self.cpu_stack = QStackedWidget()
        self.cpu_stack.addWidget(self.cpu_intra)
        self.cpu_stack.addWidget(self.cpu)
        lv.addWidget(self.cpu_stack, 1)
        cb = QHBoxLayout()
        cb.addStretch(1)
        self.btn_cpu_png = QPushButton("Export CPU PNG")
        self.btn_cpu_png.setStyleSheet(_style_button())
        self.btn_cpu_png.clicked.connect(
            lambda: self.cpu_intra.export_pdf()
            if self.cpu_stack.currentIndex() == 0 else None)
        cb.addWidget(self.btn_cpu_png)
        lv.addLayout(cb)
        self.plots.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(2)
        rh = QHBoxLayout()
        # Which row of the table above gets folded into the slot. The plot used
        # to be hard-wired to occupancy, so the other five metrics could be
        # read per slot but never WITHIN one.
        rh.addWidget(_lbl("Metric", 12, LABEL, bold=True))
        self.metric_box = QComboBox()
        self.metric_box.addItems([m[0] for m in IntraSlotPlot.METRICS])
        self.metric_box.setStyleSheet(_style_combo())
        self.metric_box.setToolTip(
            "Which metric the intra-slot curve shows. Every one is folded over "
            "the same window\nof TTIs, so the shape is where inside the 500 us "
            "slot that quantity lands.")
        self.metric_box.currentIndexChanged.connect(self._metric_changed)
        rh.addWidget(self.metric_box)

        rh.addSpacing(_s(10))
        self.chk_energy = QCheckBox("energy")
        self.chk_energy.setChecked(True)
        self.chk_energy.setToolTip(
            "Overlay energy per window on the right-hand axis. Available with "
            "any metric;\nhidden automatically when Energy IS the metric, to "
            "avoid drawing it twice.")
        self.chk_band = QCheckBox("concurrency")
        self.chk_band.setChecked(True)
        self.chk_band.setToolTip(
            "Show the concurrent-kernel box-and-whisker band low in the frame.")
        for c in (self.chk_energy, self.chk_band):
            c.setStyleSheet(_style_check())
            c.stateChanged.connect(lambda _s: self._overlays_changed())
            rh.addWidget(c)
        rh.addSpacing(_s(10))
        self.intra_hdr = _lbl("", 10, MUTED)
        rh.addWidget(self.intra_hdr)
        rh.addStretch(1)
        self.btn_pdf = QPushButton("Export PNG")
        self.btn_pdf.setStyleSheet(_style_button())
        self.btn_pdf.clicked.connect(lambda: self.intra.export_pdf())
        rh.addWidget(self.btn_pdf)
        rv.addLayout(rh)
        self.intra = IntraSlotPlot()
        rv.addWidget(self.intra, 1)
        self.plots.addWidget(right)
        self.plots.setSizes([_s(520), _s(900)])
        self.plots.setMaximumHeight(_s(430))
        v.addWidget(self.plots)

        self.ktbl = _table(["kernel", "channel", "theo occ %", "limited by",
                            "slots seen", "launches/slot",
                            "dur/slot min", "max", "mean", "median",
                            "P95", "P99"])
        v.addWidget(self.ktbl, 1)

        self.note = _lbl("", 10, MUTED)
        self.note.setWordWrap(True)
        v.addWidget(self.note)

    def _metric_changed(self, i):
        self.intra.set_metric(i)
        self._intra_note()

    def _overlays_changed(self):
        self.intra.set_overlays(self.chk_energy.isChecked(),
                                self.chk_band.isChecked())

    def restyle(self):
        self.spin.setStyleSheet(_style_spin())
        self.metric_box.setStyleSheet(_style_combo())
        for c in (self.chk_energy, self.chk_band):
            c.setStyleSheet(_style_check())
        self.btn_pdf.setStyleSheet(_style_button())
        self.btn_cpu_png.setStyleSheet(_style_button())
        self.cpu.restyle()
        self.cpu_intra.restyle()
        self.tbl.setMaximumHeight(_s(190))
        self.plots.setMaximumHeight(_s(430))
        self.plots.setSizes([_s(520), _s(900)])
        self.recompute()

    def _intra_note(self):
        """One line about the metric on screen, not always about occupancy."""
        prof = getattr(self.intra, "_profile", None)
        if not prof:
            self.intra_hdr.setText("")
            return
        name, unit, key, _fx = IntraSlotPlot.METRICS[
            self.metric_box.currentIndex()]
        vals = prof.get(key) or []
        if not vals:
            self.intra_hdr.setText(
                f"{prof['n_slots']:,} TTI folded  ·  {name}: not available "
                f"in this capture")
            return
        b = int(max(range(len(vals)), key=lambda i: vals[i]))
        u = f" {unit}" if unit else ""
        self.intra_hdr.setText(
            f"{prof['n_slots']:,} TTI folded   ·   {name} peaks in the "
            f"{prof['edges_us'][b]:.0f}-{prof['edges_us'][b+1]:.0f} us window "
            f"at {vals[b]:,.1f}{u}   ·   {prof['conc'][b]:.1f} concurrent "
            f"kernels there")

    def set_store(self, st):
        self.store = st

    def set_cpu_profile(self, prof, matched_by="", intra=None, title=""):
        self.cpu.set_profile(prof)
        self.cpu_intra.set_profile(intra, title)
        self.cpu_stack.setCurrentIndex(0 if intra else 1)
        self.btn_cpu_png.setEnabled(bool(intra))
        if intra:
            b = max(range(intra["n_bins"]), key=lambda i: intra["total"][i])
            top = sorted(((sum(v), r) for r, v in intra["roles"].items()),
                         reverse=True)[:3]
            # A capture whose run died early yields far fewer slots than were
            # asked for. That is a property of the run, not of the analysis,
            # and it has to be visible or a 220-slot fold reads like a
            # 1000-slot one.
            want = self.spin.value()
            thin = ""
            if intra["slots_available"] < want:
                thin = (f"  ⚠ ONLY {intra['slots_available']:,} SLOTS OF LIVE "
                        f"TRAFFIC were available (asked for {want:,}) — the DU "
                        f"run ended after {intra['overlap_s']:.1f} s, so this "
                        f"distribution rests on a short sample.")
            self.cpu_hdr.setText(
                f"CPU intra-TTI · {intra['label']} · {matched_by}.  "
                f"{intra['n_slots']:,} TTI folded from {intra['n_samples']:,} "
                f"perf samples at {intra['sample_hz']:.0f} Hz "
                f"({intra['slots_available']:,} slots available in the "
                f"{intra['overlap_s']:.1f} s overlap).  Clock anchor: "
                f"{intra['n_anchor']} paired readings, "
                f"{intra['clock_drift_ns']} ns drift across the capture.  "
                f"Busiest {intra['edges_us'][b]:.0f}-{intra['edges_us'][b+1]:.0f} us "
                f"at {intra['total'][b]:.2f} cores; "
                + ", ".join(f"{r} {x:.1f}" for x, r in top if x > 0)
                + ".  Click the plot or Export CPU PNG for a 300 dpi figure."
                + thin)
            return
        if prof is None:
            self.cpu_hdr.setText(
                "No CPU perf capture matched this session.  "
                "Session > Add CPU perf folder… to point at a tree of "
                "perf_percore_<N>C_<pat>_<stamp> directories.")
            return
        if prof.note:
            self.cpu_hdr.setText(prof.note)
            return
        phy = prof.phy_cores()
        self.cpu_hdr.setText(
            f"CPU · {prof.label} — {matched_by}, NOT the same run as the GPU "
            f"tabs and NOT window-scoped (perf stamps CLOCK_MONOTONIC with no "
            f"realtime reference, so there is no shared clock to select on).  "
            f"Traffic window {prof.window[0]:.0f}-{prof.window[1]:.0f} s of a "
            f"{prof.span_s:.0f} s capture.  "
            f"Machine {prof.machine_pct:.1f}% of {prof.n_cores} cores "
            f"= {prof.cores_busy:.1f} cores busy; DU threads "
            f"{prof.du_pct:.1f}%.  "
            f"{len(phy)} cores carry PHY workers, peak "
            f"{max((c['pct'] for c in phy), default=0):.0f}%."
            + (f"  {self.cpu.n_hidden} further core(s) below the top "
               f"{self.cpu.MAX_ROWS} not shown."
               if self.cpu.n_hidden else ""))

    def set_window(self, lo, hi):
        self.sel = (lo, hi)
        self.centre = (lo + hi) // 2
        self.recompute()

    # -- computation --------------------------------------------------------
    def recompute(self):
        st = self.store
        if not st or not st.n_slots:
            return
        sel_lo, sel_hi = self.sel
        n = max(self.spin.value(), sel_hi - sel_lo + 1)
        if sel_hi - sel_lo + 1 >= self.spin.value():
            lo, hi = sel_lo, sel_hi          # the selection is big enough
        else:
            lo = max(0, self.centre - n // 2)
            hi = min(st.n_slots - 1, lo + n - 1)
            lo = max(0, hi - n + 1)

        rows = st.slot_stats(lo, hi)
        if not rows:
            return
        util, idle_us, energy, occ, demand, launches = [], [], [], [], [], []
        occ_cov = []
        for r in rows:
            dur = max(1, r[2] - r[1])
            util.append(100.0 * r[7] / dur)
            idle_us.append(r[10] / 1000.0)
            energy.append(r[11])
            occ.append(r[12])
            occ_cov.append(r[13])
            demand.append(100.0 * r[8] / (st.max_warps * dur))
            launches.append(r[5])

        series = [util, idle_us, energy, occ, demand, launches]
        self.tbl.setRowCount(len(self.METRICS))
        for i, ((label, unit, defn), vals) in enumerate(zip(self.METRICS, series)):
            cnt, mn, mx, mean, med, p95, p99 = _summary(vals)
            self.tbl.setItem(i, 0, QTableWidgetItem(label))
            self.tbl.setItem(i, 1, QTableWidgetItem(unit))
            self.tbl.setItem(i, 2, _num(f"{cnt:,}", cnt))
            for c, x in enumerate((mn, mx, mean, med, p95, p99), start=3):
                fmt = "—" if x is None else (f"{x:,.0f}" if abs(x) >= 1000
                                             else f"{x:.2f}")
                self.tbl.setItem(i, c, _num(fmt, x))
            self.tbl.setItem(i, 9, QTableWidgetItem(defn))
        _fit_columns(self.tbl)

        prof = st.intra_slot_profile(lo, hi)
        title = (f"{_pattern_label(st.label)} - {st.n_cells} cells - "
                 f"{st.slot_label(lo)} to {st.slot_label(hi)} "
                 f"({hi-lo+1} TTI)")
        self.intra.set_metric(self.metric_box.currentIndex())
        self.intra.set_overlays(self.chk_energy.isChecked(),
                                self.chk_band.isChecked())
        self.intra.set_profile(prof, title)
        self._intra_note()
        self._kernel_table(st, lo, hi)

        cov = [c for c in occ_cov if c is not None]
        mean_cov = 100.0 * sum(cov) / len(cov) if cov else 0.0
        # When modelled demand saturates the board limit the energy figure
        # stops being an estimate of demand and becomes the limit itself. Say
        # so rather than letting a flat-topped distribution look like a result.
        capped = 0
        for r in rows:
            if r[11] and r[11] * 1e-6 / ((r[2] - r[1]) * 1e-9) >= P_TDP - 1.0:
                capped += 1
        cap_note = ("" if not capped else
                    f"  In {100.0*capped/len(rows):.0f}% of these slots the "
                    f"modelled dynamic power saturates the {P_TDP:.0f} W board "
                    f"limit, so their energy is the CAP, not a demand estimate.")
        src = ("TTI window selection" if (lo, hi) == self.sel
               else f"{self.spin.value():,} TTI centred on the selection")
        self.hdr.setText(f"{st.slot_label(lo)} → {st.slot_label(hi)}   ·   "
                         f"{hi-lo+1:,} TTI   ·   "
                         f"{(hi-lo+1)*st.slot_dur_ns/1e6:.1f} ms   ·   {src}")
        self.note.setText(
            "Theoretical SM occupancy is the CUDA Occupancy Calculator figure "
            f"for each launch's shape (sm_90: 64 warps/SM, 65,536 regs/SM in "
            f"units of 256, warps in groups of 4, 228 KB smem/SM), weighted by "
            f"how long each launch ran in the slot. It covers "
            f"{mean_cov:.0f}% of busy time on average — the remainder is "
            f"launches with no measured duration or no register data. "
            + ENERGY_NOTE + cap_note)

    def _kernel_table(self, st, lo, hi):
        by = {}
        for (name, chan, occ, lim, res_src, _regs, _smem,
             _slot, busy, n) in st.kernel_slot_durations(lo, hi):
            e = by.setdefault(name, dict(chan=chan, occ=occ, lim=lim,
                                         res=res_src, durs=[], launches=[]))
            e["durs"].append(busy / 1000.0)     # us in this slot
            e["launches"].append(n)
        self.ktbl.setSortingEnabled(False)
        self.ktbl.setRowCount(len(by))
        for i, (name, e) in enumerate(by.items()):
            cnt, mn, mx, mean, med, p95, p99 = _summary(e["durs"])
            self.ktbl.setItem(i, 0, QTableWidgetItem(name))
            it = QTableWidgetItem(e["chan"])
            it.setForeground(_q(channel_color(e["chan"])))
            self.ktbl.setItem(i, 1, it)
            occ = e["occ"]
            cell = _num("—" if occ is None else
                        (f"≥{occ:.1f}" if e["res"] == "cuobjdump"
                         else f"{occ:.1f}"), occ)
            if e["res"] == "cuobjdump":
                cell.setForeground(_q("#E07A5F"))
            self.ktbl.setItem(i, 2, cell)
            self.ktbl.setItem(i, 3, QTableWidgetItem(e["lim"] or "—"))
            self.ktbl.setItem(i, 4, _num(f"{cnt:,}", cnt))
            lm = sum(e["launches"]) / len(e["launches"])
            self.ktbl.setItem(i, 5, _num(f"{lm:.2f}", lm))
            for c, x in enumerate((mn, mx, mean, med, p95, p99), start=6):
                self.ktbl.setItem(i, c, _num("—" if x is None else f"{x:.2f}", x))
        self.ktbl.setSortingEnabled(True)
        self.ktbl.sortItems(8, Qt.SortOrder.DescendingOrder)
        _fit_columns(self.ktbl)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel inventory
# ─────────────────────────────────────────────────────────────────────────────

# Kernel -> source file, generated from an Aerial tree by
# build_kernel_source_map.py and shipped next to this module. Maps both the
# qualified name (ldpc::ldpc_encode_in_bit_kernel) and the bare short name, so
# lookup works whether the caller has an nsys name, a KTRACE name or a taxonomy
# short name. Missing entries render as "—" rather than failing the row.
_KSRC_CACHE = None


def _kernel_source(name: str):
    """Return "path:line" for a kernel, or None if it is not in the map."""
    global _KSRC_CACHE
    if _KSRC_CACHE is None:
        import json
        f = Path(__file__).with_name("kernel_source_map.json")
        try:
            _KSRC_CACHE = json.loads(f.read_text())
        except Exception:
            _KSRC_CACHE = {}
    if not _KSRC_CACHE or not name:
        return None
    hit = _KSRC_CACHE.get(name)
    if hit is None:
        bare = name.split("<", 1)[0].rsplit("::", 1)[-1].strip()
        hit = _KSRC_CACHE.get(bare)
    if hit is None:
        return None
    return f"{hit[0]}:{hit[1]}"


class KernelInventory(QWidget):
    """The kernel table, computed over the selected TTI window.

    Previously whole-capture, which meant the same kernel showed one launch
    count here and a different one on the Channels tab for the same view. Every
    figure below is now derived from the selected slots only, on the fly.
    """

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.hdr = _lbl("", 11, MUTED)
        self.hdr.setWordWrap(True)
        v.addWidget(self.hdr)
        self.tbl = _table(["kernel", "channel", "visible to", "launches",
                           "per slot", "slots seen", "theo occ %",
                           "limited by", "regs", "smem",
                           "dur min", "dur median", "dur P95", "dur max",
                           "samples", "role", "source file"])
        v.addWidget(self.tbl, 1)
        self.store = None

    def set_store(self, st: SessionStore):
        self.store = st

    def set_window(self, lo, hi):
        st = self.store
        if not st:
            return
        rows = st.kernel_window_stats_cached(lo, hi)
        # Provenance by name, not by the nsys_visible flag: that flag was
        # stamped from a hardcoded list before any event was read, so it
        # described the stock run config rather than this capture.
        kid_of = {v[0]: k for k, v in st.kernels.items()}
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(rows))
        n_kt = n_df = n_nodur = 0
        for i, (name, chan, role, vis, kind, occ, lim, res_src, regs, smem,
                n, n_slots, dn, dmin, dmed, dp95, dmax, per_slot) in \
                enumerate(rows):
            devfn = kind == "device_fn"
            src_lbl = st.source_label(kid_of.get(name))
            seen_by_nsys = "nsys" in src_lbl
            if devfn:
                n_df += 1
            elif not seen_by_nsys:
                n_kt += 1
            if not dn and not devfn:
                n_nodur += 1

            self.tbl.setItem(i, 0, QTableWidgetItem(name))
            it = QTableWidgetItem(chan)
            it.setForeground(_q(channel_color(chan)))
            self.tbl.setItem(i, 1, it)
            it = QTableWidgetItem(
                "device fn, not a kernel" if devfn else
                (src_lbl or "no events"))
            if devfn:
                it.setForeground(_q("#BA7517"))
            elif not seen_by_nsys:
                it.setForeground(_q("#E07A5F"))
            self.tbl.setItem(i, 2, it)
            self.tbl.setItem(i, 3, _num(f"{n:,}", n))
            self.tbl.setItem(i, 4, _num(f"{per_slot:.2f}", per_slot))
            self.tbl.setItem(i, 5, _num(f"{n_slots:,}", n_slots))
            # '>=' because occupancy from the static cuobjdump table uses the
            # maximum registers across template instantiations.
            cell = _num("—" if occ is None else
                        (f"≥{occ:.1f}" if res_src == "cuobjdump"
                         else f"{occ:.1f}"), occ)
            if res_src == "cuobjdump":
                cell.setForeground(_q("#E07A5F"))
            self.tbl.setItem(i, 6, cell)
            self.tbl.setItem(i, 7, QTableWidgetItem(lim or "—"))
            self.tbl.setItem(i, 8, _num(f"{regs}" if regs else "—", regs))
            self.tbl.setItem(i, 9, _num(f"{smem:,}" if smem else "—", smem))
            for c, val in ((10, dmin), (11, dmed), (12, dp95), (13, dmax)):
                cell = _num(_fmt_ns(val) if dn else "—", val)
                if not dn:
                    cell.setForeground(_q(MUTED))
                self.tbl.setItem(i, c, cell)
            self.tbl.setItem(i, 14, _num(f"{dn:,}" if dn else "—", dn))
            self.tbl.setItem(i, 15, QTableWidgetItem(role))
            src = _kernel_source(name)
            cell = QTableWidgetItem(src or "—")
            if src:
                cell.setToolTip(src)
            else:
                cell.setForeground(_q(MUTED))
            self.tbl.setItem(i, 16, cell)
        self.tbl.setSortingEnabled(True)
        self.tbl.sortItems(3, Qt.SortOrder.DescendingOrder)
        _fit_columns(self.tbl)

        n_kern = len(rows) - n_df
        srcs = st.capture_sources()
        breakdown = (
            f"all {n_kern} recorded by nsys" if not n_kt else
            f"{n_kern-n_kt} recorded by nsys, {n_kt} only by AERIAL_KTRACE "
            f"(device-launched graph — no host launch event exists for CUPTI "
            f"to capture)")
        nodur = (f"   {n_nodur} have no duration: instrumented with "
                 f"KERNEL_TRACE_ENTRY but not KERNEL_TIME_END_PRINT, so only "
                 f"their launch instant is known." if n_nodur else "")
        self.hdr.setText(
            f"{st.slot_label(lo)} → {st.slot_label(hi)} ({hi-lo+1:,} TTI)   ·   "
            f"captured by {' + '.join(srcs) or 'nothing'}   ·   "
            f"{n_kern} kernels ran in this window: {breakdown}.{nodur}"
            + (f"   {n_df} further traced name(s) are __device__ FUNCTIONS, "
               f"not kernels — they report their parent's grid/block and are "
               f"excluded from launch counts." if n_df else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, paths=()):
        super().__init__()
        self.setWindowTitle(APP)
        self.setStyleSheet(_style_main())
        # 1680x980 was hardcoded, so on any display smaller than that the
        # window opened larger than the screen and maximising could not
        # recover - the frame was already bigger than the area it maximises
        # into. Size to the available geometry instead, and keep the floor low
        # enough that Ctrl+- can actually shrink the window.
        self.setMinimumSize(_s(640), _s(420))
        self._fit_to_screen(initial=True)

        self.lib = SessionLibrary()
        self.lib.load()
        self.store = None
        self.thread = None
        self._cpu_thread = None

        self.sessions = SessionPanel(self.lib)
        self.sessions.opened.connect(self.open_entry)
        self.sessions.refresh()

        self.timeline = GlobalTimeline()
        self.tti = TTIView()
        self.stats = StatsPanel()
        self.channels = ChannelPanel()
        self.inventory = KernelInventory()
        self.timeline.window_changed.connect(self._window_changed)
        # The TTI pane asks; the timeline still decides. Routing the jump
        # through set_window keeps one source of truth for the window, so the
        # other tabs follow a jump exactly as they follow a drag.
        self.tti.jump_requested.connect(self._jump_to)
        self._win = None
        self._dirty = {}
        self.tabs_ready = False

        self.tabs = QTabWidget()
        # The global timeline is no longer a tab. It stays alive as the window
        # MODEL - it owns the selected region and is what emits window_changed
        # - but its bar chart is not shown, and TTI is what opens. The TTI
        # tab's own Window pane is how a window gets chosen now.
        self.timeline.setVisible(False)
        self.tti_cmp = TTICompare(self.tti)
        self.tabs.addTab(self.tti_cmp, "TTI")
        self.llm = LLMPanel()
        self.tabs.addTab(self.llm, "LLM")
        self.tabs.addTab(self.stats, "Statistics")
        self.tabs.addTab(self.channels, "Channels")
        self.tabs.addTab(self.inventory, "Kernels")
        self.tabs.currentChanged.connect(self._refresh_tab)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        self.banner = _lbl("Open a session to begin.", 11, MUTED)
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(_style_banner())
        rv.addWidget(self.banner)
        rv.addWidget(self.tabs, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.sessions)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        w = max(_s(900), self.width())
        split.setSizes([int(w * 0.18), int(w * 0.82)])
        self.setCentralWidget(split)

        self.status = self.statusBar()
        self.status.setStyleSheet(f"color:{MUTED};font-size:{_s(10)}px;")
        self._build_menus()

        for i, key in enumerate(("1", "2", "3", "4", "5")):
            QShortcut(QKeySequence(f"Ctrl+{key}"), self,
                      lambda i=i: self.tabs.setCurrentIndex(i))
        # Ctrl +/-/0 scales the whole window. Both "Ctrl++" and "Ctrl+=" are
        # bound because the plus on a US layout is a shifted equals, and Qt
        # reports whichever the keyboard actually produced.
        self._zoom_shortcuts = []
        specs = [(QKeySequence.StandardKey.ZoomIn, +1),
                 (QKeySequence.StandardKey.ZoomOut, -1)]
        specs += [(QKeySequence(k), +1)
                  for k in ("Ctrl++", "Ctrl+=", "Ctrl+Shift+=", "Ctrl+Plus")]
        specs += [(QKeySequence(k), -1)
                  for k in ("Ctrl+-", "Ctrl+_", "Ctrl+Minus")]
        specs += [(QKeySequence(k), 0) for k in ("Ctrl+0",)]
        # Register each distinct sequence ONCE. Two QShortcuts on the same
        # sequence make it ambiguous, and Qt answers an ambiguous shortcut by
        # emitting activatedAmbiguously - so neither handler runs and the key
        # does nothing at all. Ctrl+- was bound three times over
        # (StandardKey.ZoomOut, "Ctrl+-", "Ctrl+Minus"), which is exactly why
        # zoom-out was dead while zoom-in still worked through Ctrl+= .
        # "Ctrl+Plus"/"Ctrl+Minus" are not portable spellings: Qt parses them
        # to an EMPTY sequence, which then collides with itself.
        seen = set()
        for seq, d in specs:
            q = QKeySequence(seq)
            key = q.toString()
            if not key or key in seen:
                continue
            seen.add(key)
            sc = QShortcut(q, self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(lambda d=d: self.zoom_ui(d))
            self._zoom_shortcuts.append(sc)
        QShortcut(QKeySequence("Left"), self, lambda: self._nudge(-1))
        QShortcut(QKeySequence("Right"), self, lambda: self._nudge(+1))
        # Keyboard path to vertical resize, so it does not depend on landing
        # the mouse on the grip.
        QShortcut(QKeySequence("Ctrl+Shift+Down"), self,
                  lambda: self.resize_canvas_h(1.25))
        QShortcut(QKeySequence("Ctrl+Shift+Up"), self,
                  lambda: self.resize_canvas_h(1 / 1.25))

        for p in paths:
            self.lib.add(p)
        self.sessions.refresh()
        # Open the first ready session so the app is never an empty shell.
        for row, e in enumerate(self.lib.entries):
            if e.built:
                self.sessions.list.setCurrentRow(row)
                self.open_entry(e)
                break

    def _build_menus(self):
        m = self.menuBar()
        m.setStyleSheet(f"background:{BG};color:{FG};font-size:{_s(11)}px;")
        f = m.addMenu("&Session")
        a = QAction("Add capture folder…", self)
        a.triggered.connect(self.sessions.add_folder)
        f.addAction(a)
        a = QAction("Add CPU perf folder…", self)
        a.triggered.connect(self.add_cpu_folder)
        f.addAction(a)
        a = QAction("Rebuild current store", self)
        a.triggered.connect(lambda: self.sessions._open_selected(force=True))
        f.addAction(a)
        f.addSeparator()
        a = QAction("Quit", self)
        a.setShortcut(QKeySequence.StandardKey.Quit)
        a.triggered.connect(self.close)
        f.addAction(a)

        vmenu = m.addMenu("&View")
        for label, seq, d in (("Zoom in", "Ctrl++", +1),
                              ("Zoom out", "Ctrl+-", -1),
                              ("Reset zoom", "Ctrl+0", 0)):
            a = QAction(label, self)
            a.setShortcut(QKeySequence(seq))
            a.triggered.connect(lambda _c=False, d=d: self.zoom_ui(d))
            vmenu.addAction(a)

        h = m.addMenu("&Help")
        a = QAction("How kernel visibility works", self)
        a.triggered.connect(self._explain)
        h.addAction(a)
        a = QAction("Energy model", self)
        a.triggered.connect(
            lambda: QMessageBox.information(self, "Energy model",
                                            energy_provenance()))
        h.addAction(a)
        a = QAction("Theoretical SM occupancy", self)
        a.triggered.connect(self._explain_occ)
        h.addAction(a)

    def _explain(self):
        QMessageBox.information(
            self, "Kernel visibility",
            "Nsight Systems records kernel launches through CUPTI, which "
            "observes the HOST-side launch API.\n\n"
            "Every Aerial pipeline uses CUDA graphs, but only PUSCH launches "
            "its decode graph from DEVICE code: a kernel calls cudaGraphLaunch "
            "with cudaStreamGraphFireAndForget. No host API call happens, so "
            "there is no event for CUPTI to record, and nsys reports "
            "'Node-level trace for CUDA graphs launched from device codes is "
            "not supported'.\n\n"
            "There are two ways to see them anyway.\n\n"
            "1. Turn the device-graph launch off. With "
            "pusch_workCancelMode: 0 and pusch_deviceGraphLaunchEn: 0 in "
            "cuphycontroller_*.yaml, the PUSCH kernels — the LDPC decoder "
            "cubins included — are launched from the host and nsys records "
            "them like any other kernel. No second tracer is involved.\n\n"
            "2. Instrument the kernels themselves. An AERIAL_KTRACE build has "
            "each kernel stamp %globaltimer on entry and print it; the two "
            "clocks are then cross-calibrated by matching launches both "
            "sources can see, and the residual spread of that fit is the "
            "alignment uncertainty shown on the TTI tab.\n\n"
            + self._capture_sources_note())

    def _capture_sources_note(self) -> str:
        """What the CURRENTLY OPEN capture used — never a fixed number.

        The old text asserted "those 23 kernels come from KTRACE" whatever was
        loaded, which is false for any capture taken with the device-graph
        launch disabled: there the same 23 arrive through nsys.
        """
        st = self.store
        if not st:
            return "No capture is open."
        srcs = st.capture_sources()
        c = st.source_counts()
        if not srcs:
            return "The open capture contains no kernel events."
        out = [f"THIS capture ({st.label}) was taken with: "
               f"{' + '.join(srcs)}."]
        if c["ktrace_only"]:
            out.append(f"{c['ktrace_only']} kernel(s) reached it only through "
                       f"KTRACE, {c['nsys_only']} only through nsys, "
                       f"{c['both']} through both.")
        else:
            out.append(f"All {c['nsys_only'] + c['both']} kernels reached it "
                       f"through nsys, so this run had the device-graph "
                       f"launch disabled.")
        return " ".join(out)

    def zoom_ui(self, direction: int):
        """Ctrl+ / Ctrl- / Ctrl+0. direction: +1 in, -1 out, 0 reset."""
        global SCALE
        old = SCALE
        if direction == 0:
            new = 1.0
        else:
            new = round(SCALE + direction * SCALE_STEP, 2)
        new = max(SCALE_MIN, min(SCALE_MAX, new))
        if abs(new - SCALE) < 1e-6:
            return
        SCALE = new
        self.setStyleSheet(_style_main())
        self.banner.setStyleSheet(_style_banner())
        self.status.setStyleSheet(f"color:{MUTED};font-size:{_s(10)}px;")
        self.menuBar().setStyleSheet(
            f"background:{BG};color:{FG};font-size:{_s(11)}px;")
        _rescale(QApplication.instance(), self)
        # Rescaling alone only changed the fonts: the frame kept whatever size
        # it had, so Ctrl+- shrank the text inside an oversized window instead
        # of shrinking the window. Scale the frame with it, and never past the
        # screen.
        if not self.isMaximized() and not self.isFullScreen():
            ratio = new / (old or 1.0)
            self.resize(int(self.width() * ratio), int(self.height() * ratio))
        self._fit_to_screen()
        self.status.showMessage(f"UI scale {SCALE*100:.0f}%", 2000)

    def _fit_to_screen(self, initial: bool = False):
        """Keep the frame inside the screen it is on.

        Called on startup and after every UI rescale. availableGeometry()
        excludes panels and docks, so the result is the area a window can
        actually occupy rather than the raw resolution.
        """
        scr = self.screen() or QApplication.primaryScreen()
        if not scr:
            if initial:
                self.resize(1280, 800)
            return
        a = scr.availableGeometry()
        if initial:
            w, h = int(a.width() * 0.92), int(a.height() * 0.92)
            self.resize(min(1680, w), min(980, h))
            fg = self.frameGeometry()
            fg.moveCenter(a.center())
            self.move(fg.topLeft())
            return
        w, h = min(self.width(), a.width()), min(self.height(), a.height())
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)
        g = self.frameGeometry()
        if not a.contains(g):
            g.moveLeft(max(a.left(), min(g.left(), a.right() - g.width())))
            g.moveTop(max(a.top(), min(g.top(), a.bottom() - g.height())))
            self.move(g.topLeft())

    def add_cpu_folder(self):
        d = QFileDialog.getExistingDirectory(
            self, "Add a folder of perf_percore_* CPU captures")
        if not d:
            return
        n = self.lib.scan_cpu(d)
        self.lib.save()
        if n:
            self._attach_cpu()
            self.status.showMessage(f"indexed {n} CPU capture(s)", 4000)
        else:
            QMessageBox.information(
                self, APP,
                "No CPU captures found there.\n\nExpected directories named "
                "perf_percore_<N>C_<pattern>_<stamp> each containing a "
                "*.perf.data file.")

    def _explain_occ(self):
        QMessageBox.information(
            self, "Theoretical SM occupancy",
            "The CUDA Occupancy Calculator figure: how many warps can be "
            "RESIDENT on an SM given a launch's threads/block, registers per "
            "thread and shared memory, as a fraction of the 64 the hardware "
            "allows.\n\n"
            "It is a property of the launch CONFIGURATION, not of time, and "
            "not of how busy the GPU was. A kernel at 50% occupancy running "
            "for 200 us and one at 50% running for 2 us have the same "
            "occupancy; only the per-slot duration column separates them.\n\n"
            "sm_90 constants used: 64 warps/SM, 32 blocks/SM, 65,536 "
            "registers/SM allocated in units of 256 per warp with warps in "
            "groups of 4, and 228 KB of shared memory per SM. Those two "
            "granularities are what a simplified regs-per-thread ratio omits, "
            "and for register-heavy kernels they dominate the answer.\n\n"
            "Registers and shared memory come from CUPTI for kernels nsys can "
            "see. For the 23 device-launched kernels they come from a "
            "cuobjdump table of the built libraries, which records the MAXIMUM "
            "across template instantiations — so those rows are a lower bound "
            "and are shown with a '>=' prefix.")

    # -- session ------------------------------------------------------------
    def open_entry(self, entry: SessionEntry):
        if self.thread and self.thread.isRunning():
            return
        force = getattr(entry, "_force", False)
        entry._force = False
        if entry.built and not force:
            try:
                self._activate(self.lib.open(entry, log=self.status.showMessage))
                return
            except Exception:
                pass
        self.banner.setText(f"Building {entry.label} — "
                            f"{entry.src_bytes/1e9:.1f} GB of sources…")
        self.thread = BuildThread(self.lib, entry, force=force)
        self.thread.progress.connect(
            lambda m: (self.status.showMessage(m), self.banner.setText(m)))
        self.thread.done.connect(self._activate)
        self.thread.failed.connect(self._build_failed)
        self.thread.start()

    def _build_failed(self, tb):
        self.banner.setText("Build failed.")
        QMessageBox.critical(self, APP, tb[-3000:])

    def _activate(self, st: SessionStore):
        self.store = st
        # The consumers must have the store before the timeline is populated:
        # set_store on the timeline emits window_changed synchronously, and a
        # panel that has not been given the store yet silently drops it.
        for w in (self.tti, self.stats, self.channels, self.inventory):
            w.set_store(st)
        self._attach_cpu()
        self.timeline.set_store(st)
        self.sessions.refresh()

        dfn = sum(1 for v in st.kernels.values() if v[4] == "device_fn")
        # Counted from the events this capture holds. The stored nsys_visible
        # flag is a statement about the stock run config, and reading it here
        # reported "KTRACE-only" kernels in captures that contain no KTRACE.
        vis = sum(1 for k, v in st.kernels.items()
                  if v[4] != "device_fn" and "nsys" in st.source_label(k))
        n_kern = len(st.kernels) - dfn
        srcs = st.capture_sources()
        iqr = st.clock_iqr_ns
        parts = [f"{st.label}", f"{st.gpu_name}",
                 f"{st.n_cells} cells" if st.n_cells else "",
                 f"{st.n_slots:,} TTI", f"{st.n_events:,} launches",
                 f"{n_kern} kernels"
                 + (f" ({vis} nsys + {n_kern-vis} KTRACE-only)"
                    if n_kern != vis else f" (all via {' + '.join(srcs)})"
                    if srcs else "")
                 + (f" + {dfn} device fn" if dfn else "")]
        line = "   ·   ".join(p for p in parts if p)
        if iqr is not None and st.n_ktrace:
            line += (f"\nKTRACE launches are placed on the nsys timeline to "
                     f"±{iqr/1000:.0f} us (cross-calibration residual IQR). "
                     f"Their bars are dashed; hollow diamonds are launches "
                     f"with no instrumented end.")
        elif st.n_nsys and not st.n_ktrace:
            line += ("\nSingle-source capture: every kernel above came from "
                     "nsys, so no cross-calibration was needed and no bar is "
                     "estimated.")
        if not st.utc_epoch_ns:
            line += ("\nNo nsys UTC epoch in this capture — slots are tiled, "
                     "not read from the DU's SFN.slot grid.")
        self.banner.setText(line)
        self.status.showMessage(f"{st.path.name}  "
                                f"({st.path.stat().st_size/1e6:.0f} MB store)")

    def _attach_cpu(self):
        """Match a perf capture to the open session by cell count."""
        e = self.sessions.current()
        cap = self.lib.cpu_for(e) if e else None
        if cap is None:
            self.stats.set_cpu_profile(None)
            return
        how = (f"matched by cell count ({cap.cells}C {cap.pattern}, "
               f"recorded {cap.when})")
        title = (f"CPU inside the TTI - {cap.cells} cells {cap.pattern} - "
                 f"{cap.path.name}")
        self.stats.cpu_hdr.setText(
            f"CPU · decoding {cap.perf_data.name} "
            f"({cap.perf_data.stat().st_size/1e6:.0f} MB) in the background…")
        if self._cpu_thread and self._cpu_thread.isRunning():
            self._cpu_thread.done.disconnect()
            self._cpu_thread.failed.disconnect()
        self._cpu_thread = CpuProfileThread(
            cap, max(2, self.stats.spin.value()))
        self._cpu_thread.done.connect(
            lambda prof, intra: self.stats.set_cpu_profile(
                prof, how, intra, title))
        self._cpu_thread.failed.connect(
            lambda _tb: self.stats.set_cpu_profile(None))
        self._cpu_thread.start()

    def _jump_to(self, lo, hi):
        if self.store is None:
            return
        self.timeline.set_window(lo, hi)

    def _window_changed(self, lo, hi):
        """Mark every panel stale; refresh only the one on screen.

        Updating all four on every drag cost the sum of their times - the TTI
        Gantt alone was seconds - even though three of them were not visible.
        Panels now recompute when their tab is actually shown, so a window
        change costs one panel, not four.
        """
        self._win = (lo, hi)
        # Keyed by live tab count rather than the literals 1..4, which silently
        # pointed at the wrong panels the moment a tab was removed.
        self._dirty = {i: True for i in range(self.tabs.count())}
        self._refresh_tab()

    def _refresh_tab(self, _idx=None):
        if self.store is None or self._win is None:
            return
        i = self.tabs.currentIndex()
        if not self._dirty.get(i):
            return
        w = self.tabs.widget(i)
        # TTICompare is a wrapper; the panel that owns set_window is the view
        # inside it.
        panel = self.tti if w is self.tti_cmp else w
        if not hasattr(panel, "set_window"):
            return
        lo, hi = self._win
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            panel.set_window(lo, hi)
        finally:
            QApplication.restoreOverrideCursor()
        self._dirty[i] = False

    def _nudge(self, d):
        if not self.store:
            return
        a, b = self.timeline.region.getRegion()
        w = b - a
        self.timeline.set_window(a + d * w, b + d * w)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_ui(+1 if ev.angleDelta().y() > 0 else -1)
            ev.accept()
            return
        super().wheelEvent(ev)

    def closeEvent(self, ev):
        self.lib.save()
        self.lib.close_all()
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    f = QFont()
    f.setPointSize(10)
    app.setFont(f)
    w = MainWindow([Path(p) for p in sys.argv[1:]])
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
