import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'dart:async';
import '../providers/vision_provider.dart';
import '../widgets/webrtc_camera_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isCameraActive = false;
  bool _isConnecting = false;
  bool _isConnected = false;
  bool _hasAnnouncedConnection = false;
  Timer? _statusTimer;
  Timer? _voiceTimer;
  int _tapCount = 0;
  Timer? _tapTimer;

  @override
  void initState() {
    super.initState();
    _startStatusMonitoring();
    _startVoiceAnnouncements();
  }

  @override
  void dispose() {
    _statusTimer?.cancel();
    _voiceTimer?.cancel();
    _tapTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _isCameraActive ? _buildCameraView() : _buildBlindFriendlyScreen(),
    );
  }

  void _startStatusMonitoring() {
    _statusTimer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      if (!_isCameraActive) return;
      
      final provider = context.read<VisionProvider>();
      final isConnected = await provider.testConnection();
      
      if (mounted) {
        setState(() {
          _isConnected = isConnected;
          _isConnecting = !isConnected;
        });
      }
    });
  }

  void _startVoiceAnnouncements() {
    _voiceTimer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      final provider = context.read<VisionProvider>();
      
      if (_isCameraActive) {
        if (_isConnecting) {
          await provider.speak("Connecting");
        } else if (_isConnected && !_hasAnnouncedConnection) {
          await provider.speak("Connected, start analyzing");
          _hasAnnouncedConnection = true;
        }
      } else {
        await provider.speak("Touch screen to connect");
      }
    });
  }

  void _handleTap() {
    _tapCount++;
    _tapTimer?.cancel();
    
    if (_tapCount == 1) {
      _tapTimer = Timer(const Duration(milliseconds: 500), () {
        _handleSingleTap();
        _tapCount = 0;
      });
    } else if (_tapCount == 2) {
      _tapTimer?.cancel();
      _handleDoubleTap();
      _tapCount = 0;
    }
  }

  void _handleSingleTap() {
    if (!_isCameraActive) {
      _startCamera();
    }
  }

  void _handleDoubleTap() {
    if (_isCameraActive) {
      _stopCamera();
    }
  }

  void _startCamera() {
    setState(() {
      _isCameraActive = true;
      _isConnecting = true;
      _hasAnnouncedConnection = false;
    });
    context.read<VisionProvider>().setCameraActive(true);
  }

  void _stopCamera() {
    setState(() {
      _isCameraActive = false;
      _isConnecting = false;
      _isConnected = false;
      _hasAnnouncedConnection = false;
    });
    context.read<VisionProvider>().setCameraActive(false);
    context.read<VisionProvider>().stopStreaming();
  }

  Widget _buildBlindFriendlyScreen() {
    return GestureDetector(
      onTap: _handleTap,
      child: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              const Color(0xFF6B46C1), // Purple
              const Color(0xFFFF8A65), // Orange
            ],
          ),
        ),
        child: Stack(
          children: [
            // Robot character
            Center(
              child: Image.asset(
                'assets/robot3.png',
                width: 250,
                height: 250,
                fit: BoxFit.contain,
              ),
            ),
            
            // Status indicator (invisible but for debugging)
            Positioned(
              top: 50,
              left: 20,
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _isCameraActive 
                    ? (_isConnected ? 'Connected' : 'Connecting...')
                    : 'Tap to connect',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraView() {
    return GestureDetector(
      onTap: _handleTap,
      child: WebRTCCameraScreen(
        onBackPressed: () {
          _stopCamera();
        },
      ),
    );
  }
}
