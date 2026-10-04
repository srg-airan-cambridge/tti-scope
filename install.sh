#!/usr/bin/env bash
# TTI-Scope is ideated and developed by R N Mitra at Systems Research Group at
# the University of Cambridge, UK, 2026. Anthropic's Claude Code has been used
# in coding various functions, implementing features for the GUIs, and for
# testing TTI-Scope against logs.
# =====================================================================
# TTI-Scope — Ubuntu installer. 
# Tested on Ubuntu 22.04 / 24.04 (x86-64 and aarch64)
# =====================================================================
set -euo pipefail

APP_NAME="tti_scope"
INSTALL_DIR="$HOME/.local/share/${APP_NAME}"
BIN_DIR="$HOME/.local/bin"
DESKTOP_DIR="$HOME/.local/share/applications"

echo "=== TTI-Scope installer ==="
echo ""

echo "[1/5] Checking system dependencies..."
MISSING=()
for pkg in python3 python3-pip python3-venv; do
    if ! dpkg -l "$pkg" &>/dev/null; then MISSING+=("$pkg"); fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  Installing: ${MISSING[*]}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING[@]}" libgl1 libglib2.0-0
else
    echo "  System packages OK."
fi

echo "[2/5] Installing application files to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}/src"
cp -r "$(dirname "$0")/src/"* "${INSTALL_DIR}/src/"
cp "$(dirname "$0")/requirements.txt" "${INSTALL_DIR}/"

echo "[3/5] Creating Python virtual environment..."
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip --quiet
"${INSTALL_DIR}/venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt" --quiet
echo "  Python environment ready."

echo "[4/5] Creating launcher..."
mkdir -p "${BIN_DIR}"
cat > "${BIN_DIR}/tti_scope" << LAUNCHER
#!/usr/bin/env bash
exec "${INSTALL_DIR}/venv/bin/python" "${INSTALL_DIR}/src/main.py" "\$@"
LAUNCHER
chmod +x "${BIN_DIR}/tti_scope"

echo "[5/5] Creating desktop entry..."
mkdir -p "${DESKTOP_DIR}"
cat > "${DESKTOP_DIR}/tti_scope.desktop" << DESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=TTI-Scope
Comment=GPU kernel & energy profiler for 5G cuPHY + LLM inference — Nsight Systems SQLite reader
Exec=${BIN_DIR}/tti_scope %f
Icon=utilities-system-monitor
Terminal=false
Categories=Science;Development;
MimeType=application/x-sqlite3;
DESKTOP
update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true

echo ""
echo "=== Installation complete ==="
echo ""
echo "  Run from terminal:  tti_scope [path/to/capture_dir]"
echo "  Or launch from application menu: 'TTI-Scope'"
echo ""
echo "  To generate the SQLite from an .nsys-rep:"
echo "    nsys export --type=sqlite your_profile.nsys-rep"
echo ""
echo "  NOTE: if ~/.local/bin is not in your PATH, add it:"
echo "    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc"
