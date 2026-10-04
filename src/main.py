#!/usr/bin/env python3
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
"""
TTI-Scope launcher.

The application itself lives in ttiscope_app.py. This file stays because the
installed launcher (~/.local/bin/tti_scope) and the desktop entry both invoke
src/main.py by path.

  tti_scope                      open the session library
  tti_scope <capture_dir>        add and open a capture

A "capture directory" is a sweep output folder holding any of:
  cuphy_<N>C_<pat>.sqlite        nsys export       — GPU kernels, NVTX, epoch
  kernel_printf_<N>C_<pat>.log   AERIAL_KTRACE     — device-launched kernels
  cuphy_<N>C_<pat>.log           cuPHY nvlog       — SFN.slot grid, host tasks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    # Imported inside main() so that importing this module — which tooling and
    # the selftest will do — does not open a window as a side effect.
    from ttiscope_app import main as app_main
    app_main()


if __name__ == "__main__":
    main()
