#!/bin/bash

# ── PoseAI Startup Script ───────────────────────────────────
# Starts the FastAPI backend and frontend server

set -e

cd "$(dirname "$0")"

echo "================================================"
echo "  Image to Pose     "
echo "================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10+"
    exit 1
fi

# Install dependencies if needed
echo ""
echo "[1/3] Checking dependencies..."
pip install -q -r requirements.txt
echo "Dependencies OK"

# Check checkpoints
echo ""
echo "[2/3] Checking model checkpoints..."

MISSING=0
for f in \
    "checkpoints/hrnet-to-test/hrnet_first.pth" \
    "checkpoints/hrnet-to-test/hrnet_mma.pth" \
    "checkpoints/martinez-to-test/mart_aug_two.pth"
do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        MISSING=1
    else
        echo "  OK: $f"
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Error: Missing model checkpoints. Please ensure all .pth files are present."
    exit 1
fi

echo ""
echo "[3/3] Starting servers..."
echo ""

# Start frontend server in background
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..
echo "  Frontend: http://localhost:3000"

# Start API server
echo "  Backend:  http://localhost:8000"
echo ""
echo "================================================"
echo "  Open http://localhost:3000 in your browser"
echo "  Press Ctrl+C to stop"
echo "================================================"
echo ""

# Cleanup on exit
trap "kill $FRONTEND_PID 2>/dev/null; exit" INT TERM

PYTHONPATH=. python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000

