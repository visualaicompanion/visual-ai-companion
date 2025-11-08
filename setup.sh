#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$ROOT_DIR"

echo "Vision Assistant Setup"
echo "======================="

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
    echo "pip3 is not installed. Please install pip first."
    exit 1
fi

echo "Python and pip detected"

echo "Installing backend dependencies..."
pushd backend_api >/dev/null
pip3 install -r requirements.txt

YOLO_MODEL="YOLO.pt"
DEPTH_CKPT="../depth-anything-v2/checkpoints/depth_anything_v2_vits.pth"

if [ ! -f "$YOLO_MODEL" ]; then
    echo "Warning: backend_api/$YOLO_MODEL not found."
    echo "         Copy your YOLO weights into backend_api/YOLO.pt before running the server."
fi

if [ ! -f "$DEPTH_CKPT" ]; then
    echo "Warning: DepthAnythingV2 checkpoint missing at $DEPTH_CKPT"
    echo "         Clone https://github.com/DepthAnything/Depth-Anything-V2 and download the checkpoints."
fi

popd >/dev/null

echo
echo "Setup complete."
echo
echo "Next steps:"
echo "1. Start the WebRTC server:" 
echo "     cd backend_api && python3 webrtc_server.py"
echo
echo "2. Run the Flutter app:" 
echo "     cd vision_app && flutter pub get && flutter run"
echo