import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'dart:io';
import 'dart:async';
import '../providers/vision_provider.dart';

class CameraScreen extends StatefulWidget {
  final Function(File)? onImageCaptured;
  final VoidCallback? onBackPressed;

  const CameraScreen({super.key, this.onImageCaptured, this.onBackPressed});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  bool _isInitialized = false;
  Timer? _captureTimer;
  int _capturedFrames = 0;
  bool _isCapturing = false; // Track if we're currently capturing

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    _stopContinuousCapture();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Camera permission is required to use this feature.'),
            backgroundColor: Colors.red,
          ),
        );
      }
      return;
    }

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('No cameras found on this device.'),
            backgroundColor: Colors.red,
          ),
        );
      }
      return;
    }

    _controller = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false, // This disables camera sounds
    );

    try {
      print('DEBUG: Initializing camera controller...');
      await _controller!.initialize();
      print('DEBUG: Camera controller initialized successfully');

      if (mounted) {
        setState(() {
          _isInitialized = true;
        });

        print('DEBUG: Camera initialized, starting continuous capture');
        _startContinuousCapture();
      }
    } catch (e) {
      print('DEBUG: Error initializing camera: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to initialize camera: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _startContinuousCapture() {
    print('DEBUG: Starting continuous capture...');
    final provider = context.read<VisionProvider>();
    provider.startStreaming();
    provider.setController(_controller!);

    print('DEBUG: Provider streaming status: ${provider.isStreaming}');

    // Use a more camera-friendly interval - 1 second for analysis
    _captureTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!provider.isStreaming) {
        print('DEBUG: Stopping timer - provider not streaming');
        timer.cancel();
        return;
      }
      print('DEBUG: Timer triggered - attempting to capture frame');
      _captureFrame();
    });

    print('DEBUG: Capture timer started');
  }

  void _stopContinuousCapture() {
    _captureTimer?.cancel();
    _captureTimer = null;
    context.read<VisionProvider>().stopStreaming();
  }

  Future<void> _captureFrame() async {
    if (_controller == null || !_isInitialized) {
      print(
        'DEBUG: Skipping capture - controller: ${_controller != null}, initialized: $_isInitialized',
      );
      return;
    }

    // Check if we're already capturing
    if (_isCapturing) {
      print('DEBUG: Skipping capture - previous capture still in progress');
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      print('DEBUG: Taking picture...');
      final image = await _controller!.takePicture();
      final file = File(image.path);
      _capturedFrames++;

      print('DEBUG: Captured frame $_capturedFrames at path: ${image.path}');
      print('DEBUG: File exists: ${await file.exists()}');
      print('DEBUG: File size: ${await file.length()} bytes');

      // Convert file to bytes for analysis
      final bytes = await file.readAsBytes();
      await context.read<VisionProvider>().analyzeFrame(bytes);

      widget.onImageCaptured?.call(file);
    } catch (e) {
      print('DEBUG: Error capturing frame: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to capture frame: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isCapturing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Camera at the very bottom layer (full screen)
          if (_controller != null)
            Positioned.fill(child: CameraPreview(_controller!)),

          // Gradient Header overlay
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              height: 120,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.centerLeft,
                  end: Alignment.centerRight,
                  colors: [
                    const Color(0xFF6B46C1), // Purple
                    const Color(0xFFFF8A65), // Orange
                  ],
                ),
              ),
              child: SafeArea(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 10),
                  child: Column(
                    children: [
                      SizedBox(height: 10),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          // Recording indicator
                          Row(
                            children: [
                              Container(
                                width: 12,
                                height: 12,
                                decoration: BoxDecoration(
                                  color: Colors.red,
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 8),
                              Text(
                                '[REC]',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),

                          // App title
                          Text(
                            'Visual AI Companion',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),

                          // Info icon
                          Icon(
                            Icons.info_outline,
                            color: Colors.white,
                            size: 24,
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          // Back button overlay
          Positioned(
            top: 140,
            left: 20,
            child: GestureDetector(
              onTap: () {
                _stopContinuousCapture();
                widget.onBackPressed?.call();
              },
              child: Align(
                alignment: Alignment.centerLeft,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF59D), // Light yellow
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.arrow_back, size: 16, color: Colors.black87),
                      const SizedBox(width: 8),
                      Text(
                        'BACK',
                        style: TextStyle(
                          color: Colors.black87,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),

          // Text-to-Speech Status Indicator
          Positioned(
            top: 150,
            right: 20,
            child: Consumer<VisionProvider>(
              builder: (context, provider, child) {
                if (provider.isSpeaking) {
                  return Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.green.withOpacity(0.8),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.volume_up,
                          color: Colors.white,
                          size: 20,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          "Speaking",
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 14,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  );
                }
                return const SizedBox.shrink();
              },
            ),
          ),

          // Analyzing text overlay
          Positioned(
            bottom: 500,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  context.watch<VisionProvider>().isAnalyzing && !context.watch<VisionProvider>().hasFirstResult
                      ? "Analyzing..."
                      : "Camera Active",
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
          ),

          // Analysis Result Card overlay
          Positioned(
            bottom: 16,
            left: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Header with icon and title
                  Row(
                    children: [
                      Icon(
                        Icons.analytics,
                        color: const Color(0xFF6B46C1),
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Analysis Result',
                        style: TextStyle(
                          color: const Color(0xFF6B46C1),
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 12),

                  // Result text
                  Consumer<VisionProvider>(
                    builder: (context, provider, child) {
                      if (provider.isAnalyzing && !provider.hasFirstResult) {
                        return Text(
                          'Processing...',
                          style: TextStyle(
                            color: Colors.grey[600],
                            fontSize: 14,
                          ),
                        );
                      }

                      final result = provider.analysisResult;
                      if (result.isEmpty) {
                        return Text(
                          'No analysis available yet.',
                          style: TextStyle(
                            color: Colors.grey[600],
                            fontSize: 14,
                          ),
                        );
                      }

                      return Text(
                        result,
                        style: TextStyle(
                          color: Colors.black87,
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                          height: 1.3,
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
} 