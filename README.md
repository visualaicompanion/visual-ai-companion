# Vision Assistant

A Flutter app with AI-powered object detection and depth analysis for visually impaired users. The app captures images and sends them to a FastAPI backend that performs YOLO object detection and depth estimation using DepthAnythingV2.

## Project Structure

```
Visual AI/
├── backend_api/             # FastAPI + WebRTC backend
│   ├── main.py             # (legacy REST server)
│   ├── webrtc_server.py    # WebRTC inference loop (primary entry point)
│   ├── requirements.txt    # Python dependencies
│   └── YOLO.pt             # YOLO detection weights (tracked)
├── vision_app/              # Flutter mobile application
│   ├── lib/
│   │   ├── screens/        # App screens
│   │   ├── widgets/        # UI components
│   │   ├── providers/      # State management
│   │   └── services/       # API services
│   └── pubspec.yaml        # Flutter dependencies
└── depth-anything-v2/       # DepthAnythingV2 repo (clone separately; ignored in git)
```

## Features

- **Real-time Streaming**: Continuous camera feed
- **Object Detection**: YOLO-based object detection
- **Depth Estimation**: DepthAnythingV2 for depth analysis
- **AI Analysis**: Azure OpenAI integration for natural language descriptions
- **Terminal Output**: Real-time results displayed in terminal
- **Accessibility Focused**: Designed for visually impaired users

## Setup Instructions

### Prerequisites

1. **Flutter SDK**: Install Flutter (https://flutter.dev/docs/get-started/install)
2. **Python 3.8+**: For the backend API
3. **CUDA** (optional): For GPU acceleration on the backend

### Backend Setup

> Note: DepthAnythingV2 assets are not tracked in git. Clone the repo and download checkpoints locally.

1. **Clone DepthAnythingV2**

   ```bash
   git clone https://github.com/DepthAnything/Depth-Anything-V2 depth-anything-v2
   ```

   Download the small checkpoint (or the variant you need) into `depth-anything-v2/checkpoints/`, e.g.:

   ```bash
   wget -P depth-anything-v2/checkpoints \
     https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
   ```

2. **Verify YOLO weights**

   - `backend_api/YOLO.pt` is committed with the project. Replace it if you trained your own model.

3. **Install dependencies via bundled script**

   ```bash
   ./setup.sh
   ```

   The script installs Python packages, runs `flutter pub get`, and warns if model files are missing.

4. **Start the WebRTC streaming server**
   ```bash
   cd backend_api
   python3 webrtc_server.py
   ```

### Flutter App Setup

Before running, edit `vision_app/lib/providers/vision_provider.dart` and set `_host` to the IP address of the machine running `backend_api/webrtc_server.py`.

If you ran `./setup.sh` the dependencies are already fetched. Otherwise:

```bash
cd vision_app
flutter pub get
flutter run
```

## Usage

1. **Start the backend server** (see Backend Setup)
2. **Launch the Flutter app** (see Flutter App Setup)
3. **Tap "Start Streaming"** to begin continuous analysis
4. **Point the camera** at objects you want to analyze
5. **View real-time results** in the terminal/console
6. **Monitor the app** for streaming status and frame count

## Configuration

### Backend Configuration

Edit `backend_api/main.py` to modify:

- Model paths
- Input size
- Confidence thresholds
- Device settings (CPU/GPU)

### Flutter App Configuration

Edit `vision_app/lib/providers/vision_provider.dart` to modify:

- API endpoint URL
- Request timeouts
- Error handling

## Troubleshooting

### Common Issues

1. **Camera Permission Denied**:

   - Grant camera permissions in device settings
   - Restart the app

2. **Backend Connection Failed**:

   - Ensure the backend server is running
   - Check if the API URL is correct in the Flutter app
   - Verify firewall settings

3. **Model Loading Errors**:

   - Ensure all model files are in the correct locations
   - Check CUDA installation if using GPU
   - Verify Python dependencies are installed

4. **Memory Issues**:
   - Reduce input image size
   - Use CPU instead of GPU
   - Close other applications

### Performance Optimization

- Use GPU acceleration for faster processing
- Reduce image resolution for faster analysis
- Implement image compression before sending to API
- Add caching for repeated analyses
