# WebRTC Live Streaming Setup

This setup enables **real-time video streaming** from your phone to your laptop, with instant AI analysis and text-to-speech feedback.

## 🚀 What You Get

- **Live video stream** from phone to laptop (no more image capture delays!)
- **Real-time processing** with YOLO + DepthAnythingV2 models
- **Instant results** via WebSocket back to phone
- **Text-to-speech** speaks analysis results in real-time
- **~100-200ms latency** (near real-time performance)

## 📱 Phone Setup

### 1. Install Dependencies
```bash
cd vision_app
flutter pub get
```

### 2. Update IP Address
In `lib/providers/vision_provider.dart`, update:
```dart
final String _host = "YOUR_LAPTOP_IP"; // e.g., "172.20.10.4"
```

### 3. Run the App
```bash
flutter run
```

## 💻 Laptop Setup

### 1. Install Python Dependencies
```bash
cd backend_api
pip install -r requirements.txt
```

### 2. Start WebRTC Server
```bash
# Option A: Use the startup script
./start_webrtc.sh

# Option B: Manual start
python3 webrtc_server.py --host 0.0.0.0 --port 8080
```

### 3. Verify Server is Running
You should see:
```
Starting WebRTC Video Streaming Server...
Starting server on port 8080...
Your phone should connect to: ws://YOUR_IP:8080
```

## 🔗 How It Works

### 1. **Phone → Laptop**
- Phone opens camera and starts WebRTC stream
- Live video flows to laptop at 30 FPS
- No more image capture delays!

### 2. **Laptop Processing**
- Receives live video stream
- Processes every 10th frame (for performance)
- Runs YOLO + DepthAnythingV2 models
- Generates analysis results

### 3. **Results → Phone**
- Analysis results sent via WebSocket
- Text-to-speech speaks results instantly
- Real-time feedback loop

## 📊 Performance

- **Video Stream**: 30 FPS (640x480)
- **Processing**: Every 10th frame (3 FPS analysis)
- **Latency**: ~100-200ms end-to-end
- **Bandwidth**: ~2-5 Mbps (adjustable)

## 🛠️ Troubleshooting

### Connection Issues
- **Check firewall**: Ensure port 8080 is open
- **Verify IP**: Make sure phone and laptop are on same network
- **STUN server**: Uses Google's public STUN server

### Performance Issues
- **Lower resolution**: Change video width/height in `webrtc_camera_screen.dart`
- **Reduce frame rate**: Change `frameRate: 30` to lower value
- **Processing frequency**: Change `frame_count % 10` in `webrtc_server.py`

### Model Issues
- **GPU memory**: Ensure CUDA is available for depth model
- **Model loading**: Check that model files exist in correct paths

## 🔧 Customization

### Video Quality
```dart
// In webrtc_camera_screen.dart
'video': {
  'width': 640,      // Lower = faster
  'height': 480,     // Lower = faster
  'frameRate': 30,   // Lower = less bandwidth
}
```

### Processing Frequency
```python
# In webrtc_server.py
if self.frame_count % 10 == 0:  # Process every 10th frame
    # Change to 5 for more frequent analysis
    # Change to 20 for less frequent analysis
```

### Analysis Triggers
```python
# In webrtc_server.py
COOLDOWN_SECONDS = 7  # Adjust cooldown between analyses
```

## 🎯 Next Steps

1. **Start WebRTC server** on laptop
2. **Run Flutter app** on phone
3. **Start streaming** - tap "Start Screening"
4. **Enjoy real-time analysis** with instant TTS feedback!

## 📝 Notes

- **First connection** may take 5-10 seconds to establish
- **WebRTC** automatically handles network changes and reconnection
- **Results** are streamed continuously as objects are detected
- **TTS** will speak new results immediately, interrupting previous speech

This setup gives you **true real-time AI vision** - no more waiting for image capture and processing delays!



