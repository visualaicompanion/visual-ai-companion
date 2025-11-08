#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$ROOT_DIR"

echo "🚀 Vision Assistant Setup Script"
echo "================================"

# --- Sanity checks -----------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"
echo

# --- Backend setup -----------------------------------------------------------
echo "📦 Setting up backend dependencies..."
pushd backend_api >/dev/null

echo "Installing Python requirements from backend_api/requirements.txt ..."
pip3 install -r requirements.txt
echo "✅ Backend dependencies installed"

YOLO_MODEL="../YOLO.pt"
DEPTH_CKPT="../depth-anything-v2/checkpoints/depth_anything_v2_vits.pth"

if [ ! -f "$YOLO_MODEL" ]; then
    echo "⚠️  Warning: YOLO model $YOLO_MODEL not found."
    echo "   Place your trained model there or update backend_api/main.py to point to the correct file."
fi

if [ ! -f "$DEPTH_CKPT" ]; then
    echo "⚠️  Warning: DepthAnythingV2 checkpoint not found at:"
    echo "   $DEPTH_CKPT"
    echo "   Download checkpoints from https://github.com/DepthAnything/Depth-Anything-V2 and place them in that directory."
fi

popd >/dev/null

# --- Flutter setup -----------------------------------------------------------
echo
echo "📱 Setting up Flutter app..."
if ! command -v flutter >/dev/null 2>&1; then
    echo "⚠️  Flutter is not installed or not in PATH."
    echo "   Install Flutter (https://flutter.dev/docs/get-started/install) and run:"
    echo "   cd vision_app && flutter pub get"
else
    echo "Running flutter pub get ..."
    pushd vision_app >/dev/null
    flutter pub get
    echo "✅ Flutter dependencies installed"
    popd >/dev/null
fi

echo
echo "🎉 Setup complete!"
echo
echo "📋 Next steps:"
echo "1. Start the backend server:"
echo "     cd backend_api && python3 start_server.py"
echo
echo "2. Start the WebRTC server (for live streaming):"
echo "     cd backend_api && python3 webrtc_server.py"
echo
echo "3. Run the Flutter app:"
echo "     cd vision_app && flutter run"
echo
echo "4. API docs are available at http://localhost:8000/docs once the backend is running."
echo
echo "ℹ️  For more details, see README.md"