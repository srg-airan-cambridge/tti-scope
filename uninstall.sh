#!/usr/bin/env bash
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
set -euo pipefail
echo "Uninstalling TTI-Scope..."
rm -rf "$HOME/.local/share/tti_scope"
rm -f  "$HOME/.local/bin/tti_scope"
rm -f  "$HOME/.local/share/applications/tti_scope.desktop"
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
echo "Done."
