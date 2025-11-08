import 'dart:convert';
import 'dart:typed_data';
import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as ws_status;
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';

class VisionProvider extends ChangeNotifier {
  CameraController? _controller;
  bool _isCameraActive = false;
  bool _isStreaming = false;
  String _analysisResult = "Ready to analyze...";
  String _connectionStatus = "Disconnected";
  bool _isAnalyzing = false;

  // Streaming variables
  Timer? _streamTimer;
  Timer? _analysisTimer;
  int _frameCount = 0;
  int _lastAnalysisTime = 0;
  int _lastImageSent = 0;
  bool _hasFirstResult = false;

  // Text-to-speech
  FlutterTts? _flutterTts;
  bool _isSpeaking = false;

  // Spatial audio (8D audio) method channel
  static const MethodChannel _spatialAudioChannel = MethodChannel(
    'com.visionapp.spatial_audio',
  );

  // Audio player for spatial audio playback
  AudioPlayer? _audioPlayer;

  // WebSocket
  WebSocketChannel? _ws;
  bool _wsConnected = false;
  bool _wsSending = false; // backpressure for legacy path
  int _lastWSSentMs = 0;
  Timer? _wsReconnectTimer;
  Duration _wsReconnectDelay = const Duration(seconds: 3);

  // Configuration
  // TODO: Set _host to the IP address of the machine running backend_api/webrtc_server.py
  final String _host = "YOUR_IP";
  final int _port = 9000;
  final int _fps = 50;
  final int _frameInterval = 20; // 1000ms / 50fps
  final int _analysisInterval = 1000; // 1 second

  // Getters
  CameraController? get controller => _controller;
  bool get isCameraActive => _isCameraActive;
  bool get isStreaming => _isStreaming;
  String get analysisResult => _analysisResult;
  String get connectionStatus => _connectionStatus;
  bool get isAnalyzing => _isAnalyzing;
  bool get isSpeaking => _isSpeaking;
  bool get hasFirstResult => _hasFirstResult;
  bool get isWebSocketConnected => _wsConnected;
  String get host => _host;
  int get port => _port;

  VisionProvider() {
    _initializeTts();
  }

  void _initializeTts() {
    _flutterTts = FlutterTts();
    _flutterTts!.setLanguage("en-US");
    _flutterTts!.setSpeechRate(0.8); // Faster but still clear
    _flutterTts!.setVolume(1.0);
    _flutterTts!.setPitch(1.0);

    _flutterTts!.setStartHandler(() {
      _isSpeaking = true;
      notifyListeners();
    });

    _flutterTts!.setCompletionHandler(() {
      _isSpeaking = false;
      notifyListeners();
    });

    _flutterTts!.setErrorHandler((msg) {
      _isSpeaking = false;
      print("TTS Error: $msg");
      notifyListeners();
    });
  }

  Future<void> speak(String text) async {
    if (_flutterTts != null && text.isNotEmpty) {
      try {
        if (_isSpeaking) {
          await _flutterTts!.stop();
          await Future.delayed(const Duration(milliseconds: 100));
        }
        await _flutterTts!.speak(text);
      } catch (e) {
        print("Error speaking text: $e");
      }
    }
  }

  Future<void> stopSpeaking() async {
    if (_flutterTts != null && _isSpeaking) {
      try {
        await _flutterTts!.stop();
      } catch (e) {
        print("Error stopping speech: $e");
      }
    }
  }

  Future<void> speakImmediately(String text) async {
    if (_flutterTts != null && text.isNotEmpty) {
      try {
        await _flutterTts!.stop();
        await Future.delayed(const Duration(milliseconds: 100));
        await _flutterTts!.speak(text);
      } catch (e) {
        print("Error speaking text immediately: $e");
      }
    }
  }

  // Convert clock direction to short direction word for fast speech
  String _directionToWord(String direction) {
    switch (direction) {
      case "10 o'clock":
        return "Left"; // Left side
      case "11 o'clock":
        return "Left"; // Slightly left
      case "12 o'clock":
        return "Ahead"; // Straight ahead
      case "1 o'clock":
        return "Right"; // Slightly right
      case "2 o'clock":
        return "Right"; // Right side
      default:
        return "Ahead"; // Default to ahead
    }
  }

  // Speak with direction - FAST approach using short direction words
  Future<void> speakWithSpatialAudio(
    String objectName,
    String direction,
  ) async {
    if (objectName.isEmpty) return;

    try {
      // Get short direction word (Left/Right/Ahead)
      String directionWord = _directionToWord(direction);

      // Create fast message: "Left door" or "Right car" or "Ahead person"
      // For center (12 o'clock), we can skip "Ahead" to save time
      String message;
      if (directionWord == "Ahead") {
        // For straight ahead, just say the object (faster)
        message = objectName;
      } else {
        // For left/right, add direction word: "Left door"
        message = "$directionWord $objectName";
      }

      // Speak immediately (fast!)
      await speak(message);

      print("FAST: Spoke '$message' (direction: $direction)");
    } catch (e) {
      print("Error speaking with direction: $e");
      // Fallback to regular TTS
      await speak(objectName);
    }
  }

  // Camera control methods
  void setCameraActive(bool active) {
    _isCameraActive = active;
    notifyListeners();
  }

  void setController(CameraController controller) {
    _controller = controller;
    notifyListeners();
  }

  // WebSocket (connect to WebRTC server /ws)
  Future<void> connectWebSocket() async {
    // Avoid duplicate connects
    if (_wsConnected || _ws != null) return;

    try {
      final uri = Uri.parse('ws://$_host:$_port/ws');
      print('DEBUG: Connecting to WS: $uri');
      _ws = WebSocketChannel.connect(uri);
      _wsConnected = true;
      setConnectionStatus("WS Connected");
      notifyListeners();

      _ws!.stream.listen(
        (message) {
          try {
            final data = jsonDecode(message);
            if (data is Map && data.containsKey('analysis_result')) {
              final result = data['analysis_result'] ?? '';
              final spatialAudioData = data['spatial_audio'];

              if (result is String && result.isNotEmpty) {
                // Use spatial audio if available, otherwise fallback to regular TTS
                if (spatialAudioData != null &&
                    spatialAudioData is List &&
                    spatialAudioData.isNotEmpty) {
                  setAnalysisResultWithSpatialAudio(result, spatialAudioData);
                } else {
                  setAnalysisResult(result);
                }
              }
            }
          } catch (e) {
            print('WS parse error: $e');
          }
        },
        onDone: () {
          print('DEBUG: WS closed');
          _wsConnected = false;
          _ws = null;
          setConnectionStatus("WS Disconnected");
          if (_isStreaming) _scheduleWsReconnect();
          notifyListeners();
        },
        onError: (e) {
          print('DEBUG: WS error: $e');
          _wsConnected = false;
          _ws = null;
          setConnectionStatus("WS Error");
          if (_isStreaming) _scheduleWsReconnect();
          notifyListeners();
        },
      );
    } catch (e) {
      print('DEBUG: Failed to connect WS: $e');
      _wsConnected = false;
      _ws = null;
      setConnectionStatus("WS Connect Failed");
      if (_isStreaming) _scheduleWsReconnect();
      notifyListeners();
    }
  }

  void _scheduleWsReconnect() {
    _wsReconnectTimer?.cancel();
    _wsReconnectTimer = Timer(_wsReconnectDelay, () {
      if (_isStreaming && !_wsConnected && _ws == null) {
        connectWebSocket();
      }
    });
  }

  Future<void> disconnectWebSocket() async {
    _wsReconnectTimer?.cancel();
    try {
      await _ws?.sink.close(ws_status.normalClosure);
    } catch (_) {}
    _ws = null;
    _wsConnected = false;
    notifyListeners();
  }

  // Streaming methods
  void startStreaming() {
    if (_isStreaming) return;

    _isStreaming = true;
    _frameCount = 0;
    _lastAnalysisTime = 0;
    _lastImageSent = 0;
    _hasFirstResult = false;
    _analysisResult = "Ready to analyze...";

    // Ensure WS connected to receive analysis results from WebRTC server
    connectWebSocket();

    print('DEBUG: Starting streaming');
    notifyListeners();
  }

  void stopStreaming() {
    _isStreaming = false;
    _streamTimer?.cancel();
    _analysisTimer?.cancel();
    disconnectWebSocket();
    _hasFirstResult = false;
    _analysisResult = "Ready to analyze...";
    print('DEBUG: Streaming stopped');
    notifyListeners();
  }

  void incrementFrameCount() {
    _frameCount++;
    notifyListeners();
  }

  // Analysis methods
  void setAnalyzing(bool analyzing) {
    _isAnalyzing = analyzing;
    notifyListeners();
  }

  void setAnalysisResult(String result) {
    _analysisResult = result;
    _hasFirstResult = true;

    // Speak the analysis result immediately (stopping any current speech)
    if (result.isNotEmpty && result != "Ready to analyze...") {
      bool isUrgent =
          result.toLowerCase().contains('person') ||
          result.toLowerCase().contains('car') ||
          result.toLowerCase().contains('truck') ||
          result.toLowerCase().contains('right in front of you');

      if (isUrgent) {
        speakImmediately(result);
      } else {
        speak(result);
      }
    }

    notifyListeners();
  }

  // Set analysis result with spatial audio (8D audio)
  void setAnalysisResultWithSpatialAudio(
    String result,
    List<dynamic> spatialAudioData,
  ) {
    _analysisResult = result;
    _hasFirstResult = true;

    // Process each object with spatial audio
    if (spatialAudioData.isNotEmpty) {
      // Speak objects sequentially with spatial audio
      _speakSpatialObjectsSequentially(spatialAudioData);
    }

    notifyListeners();
  }

  // Speak objects sequentially with spatial audio
  Future<void> _speakSpatialObjectsSequentially(List<dynamic> objects) async {
    for (var obj in objects) {
      try {
        final name = obj['name'] as String? ?? '';
        final direction = obj['direction'] as String? ?? '12 o\'clock';

        if (name.isNotEmpty) {
          // Speak with spatial audio (only object name, direction handled by panning)
          await speakWithSpatialAudio(name, direction);

          // Wait a bit between objects to avoid overlap
          await Future.delayed(const Duration(milliseconds: 500));
        }
      } catch (e) {
        print("Error processing spatial audio object: $e");
      }
    }
  }

  void setConnectionStatus(String status) {
    _connectionStatus = status;
    notifyListeners();
  }

  // Test API connection (HTTP fallback)
  Future<bool> testConnection() async {
    try {
      print('DEBUG: Testing backend connection...');
      final response = await http
          .get(Uri.parse('http://$_host:$_port/health'))
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        setConnectionStatus(_wsConnected ? "Connected (WS)" : "Connected");
        print('DEBUG: Backend connection successful');
        return true;
      } else {
        setConnectionStatus("Error: ${response.statusCode}");
        print('DEBUG: Backend returned status: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      setConnectionStatus("Disconnected");
      print('DEBUG: Backend connection failed: $e');
      return false;
    }
  }

  // Send frame to backend for analysis via WS (fallback HTTP)
  // Not used in WebRTC flow; kept for legacy compatibility
  Future<void> analyzeFrame(Uint8List imageBytes) async {
    if (!_isStreaming) return;
    // In WebRTC mode, frames are streamed directly; this method is not used.
  }

  @override
  void dispose() {
    stopStreaming();
    stopSpeaking();
    _flutterTts?.stop();
    super.dispose();
  }
}
