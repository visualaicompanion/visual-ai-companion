#!/bin/bash

echo "Starting WebRTC Video Streaming Server..."
echo "This server will receive live video streams from your phone"
echo "and process them with YOLO + DepthAnythingV2 models"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! python3 -c "import aiortc" 2>/dev/null; then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
fi

# Start the WebRTC server
echo "Starting server on port 8080..."
echo "Your phone should connect to: ws://YOUR_IP:8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 webrtc_server.py --host 0.0.0.0 --port 8080



