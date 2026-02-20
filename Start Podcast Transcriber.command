#!/bin/zsh
# ─────────────────────────────────────────────────────────
# Podcast Transcriber – double-click launcher for macOS
# ─────────────────────────────────────────────────────────

# Ensure Homebrew binaries (uv, ffmpeg) are on PATH.
# Finder doesn't source shell profiles, so we add this explicitly.
export PATH="/opt/homebrew/bin:$PATH"

# cd to the directory where this script lives (the project root).
cd "$(dirname "$0")" || exit 1

# Load the API key (and any other env vars) from .env if it exists.
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
else
    echo "⚠  No .env file found. Claude formatting will be disabled."
    echo "   Create a .env file with: ANTHROPIC_API_KEY=sk-ant-..."
    echo ""
fi

# Create required directories if they don't exist.
mkdir -p uploads transcriptions output

# Open the browser after a short delay so the server has time to start.
(sleep 3 && open "http://localhost:8000") &

echo "Starting Podcast Transcriber at http://localhost:8000 ..."
echo "Close this window to stop the server."
echo ""

# Start the server (this blocks until the user closes the Terminal window).
uv run uvicorn app:app --host 0.0.0.0 --port 8000
