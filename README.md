# Vision Assistant

A Flutter app with AI-powered object detection and depth analysis for visually impaired users. The app captures images and sends them to a FastAPI backend that performs YOLO object detection and depth estimation using DepthAnythingV2.

## Project Structure

```
Visual AI/
├── backend_api/             # FastAPI backend
│   ├── main.py             # REST endpoints + inference logic
│   ├── webrtc_server.py    # WebRTC inference loop
│   ├── requirements.txt    # Python dependencies
│   └── start_server.py     # Convenience launcher
├── vision_app/              # Flutter mobile application
│   ├── lib/
│   │   ├── screens/        # App screens
│   │   ├── widgets/        # UI components
│   │   ├── providers/      # State management
│   │   └── services/       # API services
│   └── pubspec.yaml        # Flutter dependencies
├── YOLO.pt                  # YOLO detection weights (user supplied, ignored in git)
└── depth-anything-v2/       # DepthAnythingV2 repo (user clones separately; ignored in git)
```

## Features

- **Real-time Streaming**: Continuous camera feed at 50 FPS
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

> 💡 The backend depends on two local assets that are **not tracked in git**:
>
> 1. **YOLO.pt** – your exported YOLOv8/YOLO11 weights (place the file in the repo root).
> 2. **DepthAnythingV2 checkpoints + code** – clone the official repo next to this project.

1. **Clone DepthAnythingV2**
   ```bash
   git clone https://github.com/DepthAnything/Depth-Anything-V2 depth-anything-v2
   ```
   Download the small checkpoint (or the variant you need) into `depth-anything-v2/checkpoints/`, e.g.:
   ```bash
   wget -P depth-anything-v2/checkpoints \
     https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
   ```

2. **Copy your YOLO weights**
   ```bash
   cp /path/to/your/model.pt YOLO.pt
   ```

3. **Install dependencies via bundled script**
   ```bash
   ./setup.sh
   ```
   The script installs Python packages, runs `flutter pub get`, and warns if model files are missing.

4. **Start the REST API (FastAPI)**
   ```bash
   cd backend_api
   python3 start_server.py
   ```
   - Main API: http://localhost:8000  
   - API docs: http://localhost:8000/docs  
   - Health check: http://localhost:8000/health

5. **Optional: start the WebRTC streaming server**
   ```bash
   python3 webrtc_server.py
   ```

### Flutter App Setup

If you ran `./setup.sh` the dependencies are already fetched. Otherwise:

```bash
cd vision_app
flutter pub get
flutter run
```

## API Endpoints

### POST /analyze_stream
Analyzes streaming frames for real-time object detection and depth estimation.

**Request Body**:
```json
{
  "image": "base64_encoded_image_string",
  "frame_number": 1,
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

**Response**:
```json
{
  "analysis": "Natural language description of the scene",
  "objects_detected": [
    {
      "label": "person",
      "confidence": 0.95,
      "direction": "12 o'clock",
      "proximity": "nearby",
      "priority": 5,
      "depth_value": 0.6
    }
  ],
  "depth_info": {
    "min_depth": 0.1,
    "max_depth": 0.9,
    "mean_depth": 0.5
  },
  "frame_number": 1,
  "processing_time": 0.85
}
```

### POST /analyze
Analyzes a single image for object detection and depth estimation (legacy endpoint).

### GET /health
Health check endpoint to verify API status.

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

## Development

### Adding New Features

1. **Backend**: Add new endpoints in `backend_api/main.py`
2. **Flutter**: Add new screens in `lib/screens/`
3. **State Management**: Update providers in `lib/providers/`

### Testing

1. **API Testing**: Use the interactive docs at http://localhost:8000/docs
2. **Flutter Testing**: Run `flutter test` in the app directory
3. **Integration Testing**: Test the full pipeline with real images

## Git ignore recommendations

The repo ships with a root `.gitignore` covering:

- `depth-anything-v2/` (clone locally but keep out of version control)
- `YOLO.pt` and other `*.pt` / `*.pth` weight files
- Python build artifacts (`__pycache__/`, `*.pyc`, virtualenvs)
- Logs / temporary outputs (`depth_logs/`, `*.log`)
- OS cruft (`.DS_Store`)
- Flutter build folders (`build/`, `.dart_tool/`)

Feel free to add additional entries if your workflow creates other transient files.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check the logs for error messages
4. Open an issue with detailed information 