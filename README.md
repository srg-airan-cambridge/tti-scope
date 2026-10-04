# TTI-Scope

**A 5G TTI-aware Nsys visualisation tool for practical AI-RAN.**

NVIDIA Aerial CUDA-Accelerated RAN launches CUDA kernels for physical layer signal processing computations.
TTI-Scope helps AI-RAN engineers to visualise kernel launching patterns for co-located NVIDIA Aerial and LLM inference.

TTI-Scope shows ~57 kernels the DL+UL pipelines actually execute, on
the DU's own **SFN.slot** grid, from nsys captures and Aerial logs.

TTI-Scope is under active development and may contain bugs.

## Install and run

Tested on Ubuntu 22.04 / 24.04 (x86-64 and aarch64), Python 3 with PyQt6 and pyqtgraph.

```bash
./install.sh                     # installs to ~/.local/share/tti_scope, launcher ~/.local/bin/tti_scope
tti_scope                        # open the session library
tti_scope /path/to/capture_dir   # add and open a capture
```

A capture directory holds any of: the nsys export (`*.sqlite`, or a `.nsys-rep`
that TTI-Scope exports on first open), the AERIAL_KTRACE printf log
(`kernel_printf_*.log`), and the cuPHY nvlog (`cuphy*.log`). The first open
builds an indexed session store next to the capture; later opens are instant.

Uninstall with `./uninstall.sh`.

## Credits

TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used in
coding various functions, implementing features for the GUIs, and for testing
TTI-Scope against logs.

## Citation

If you use TTI-Scope in your research, please cite:

```bibtex
@software{mitra2026ttiscope,
  author       = {Mitra, R. N.},
  title        = {{TTI-Scope}: A 5G TTI-aware Nsys Visualisation Tool for Practical {AI-RAN}},
  year         = {2026},
  publisher    = {GitHub},
  organization = {Systems Research Group, University of Cambridge},
  url          = {https://github.com/srg-airan-cambridge/tti-scope},
  note         = {Software repository}
}
```

For bibliography styles without `@software`, use:

```bibtex
@misc{mitra2026ttiscope,
  author       = {Mitra, R. N.},
  title        = {{TTI-Scope}: A 5G TTI-aware Nsys Visualisation Tool for Practical {AI-RAN}},
  year         = {2026},
  howpublished = {\url{https://github.com/srg-airan-cambridge/tti-scope}},
  note         = {Systems Research Group, University of Cambridge. Software repository}
}
```

## Licence

MIT — see [LICENSE](LICENSE).
