import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../providers/vision_provider.dart';

class WebRTCCameraScreen extends StatefulWidget {
  final VoidCallback? onBackPressed;

  const WebRTCCameraScreen({super.key, this.onBackPressed});

  @override
  State<WebRTCCameraScreen> createState() => _WebRTCCameraScreenState();
}

class _WebRTCCameraScreenState extends State<WebRTCCameraScreen> {
  RTCPeerConnection? _peerConnection;
  MediaStream? _localStream;
  final RTCVideoRenderer _localRenderer = RTCVideoRenderer();
  bool _isInitialized = false;
  bool _isConnecting = false;
  bool _isConnected = false;
  Timer? _connectionTimer;
  int _tapCount = 0;
  Timer? _tapTimer;

  Future<void> _waitForIceGatheringComplete({
    Duration timeout = const Duration(seconds: 7),
  }) async {
    if (_peerConnection == null) return;
    final pc = _peerConnection!;

    if (pc.iceGatheringState ==
        RTCIceGatheringState.RTCIceGatheringStateComplete) {
      return;
    }

    final completer = Completer<void>();
    late void Function(RTCIceGatheringState) sub;
    sub = (state) {
      if (state == RTCIceGatheringState.RTCIceGatheringStateComplete &&
          !completer.isCompleted) {
        completer.complete();
      }
    };

    pc.onIceGatheringState = sub;

    try {
      await completer.future.timeout(timeout);
    } catch (_) {
      // proceed with whatever candidates are available
    } finally {
      pc.onIceGatheringState = null;
    }
  }

  @override
  void initState() {
    super.initState();
    _initializeWebRTC();
  }

  @override
  void dispose() {
    _tapTimer?.cancel();
    _disposeWebRTC();
    super.dispose();
  }

  Future<void> _initializeWebRTC() async {
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

    try {
      await _localRenderer.initialize();

      final config = {
        'iceServers': [
          {'urls': 'stun:stun.l.google.com:19302'},
        ],
      };

      _peerConnection = await createPeerConnection(config, {
        'mandatory': {},
        'optional': [
          {'DtlsSrtpKeyAgreement': true},
        ],
      });

      _localStream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {'width': 480, 'height': 360, 'frameRate': 15},
      });

      _localStream!.getTracks().forEach((track) {
        _peerConnection!.addTrack(track, _localStream!);
      });

      _localRenderer.srcObject = _localStream;

      _peerConnection!.onIceCandidate = (candidate) {
        print('ICE candidate: $candidate');
      };

      _peerConnection!.onIceConnectionState = (state) {
        print('ICE connection state: $state');
        final isNowConnected =
            state == RTCIceConnectionState.RTCIceConnectionStateConnected ||
            state == RTCIceConnectionState.RTCIceConnectionStateCompleted;
        if (mounted) {
          setState(() {
            _isConnected = isNowConnected;
          });
        }
        if (isNowConnected) {
          _connectionTimer?.cancel();
          _isConnecting = false;
          _onStreamingStarted();
        }
      };

      _peerConnection!.onConnectionState = (state) {
        print('Connection state: $state');
        if (mounted) {
          setState(() {
            _isConnected =
                state == RTCPeerConnectionState.RTCPeerConnectionStateConnected;
          });
        }
        if (_isConnected) {
          _connectionTimer?.cancel();
          _isConnecting = false;
          _onStreamingStarted();
        }
      };

      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }

      _startStreaming();
    } catch (e) {
      print('Error initializing WebRTC: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to initialize WebRTC: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _startStreaming() {
    if (!_isInitialized || _isConnecting) return;

    setState(() {
      _isConnecting = true;
    });

    _connectionTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_isConnected) {
        timer.cancel();
        _onStreamingStarted();
      }
    });

    _connectToServer();
  }

  Future<void> _connectToServer() async {
    try {
      if (_peerConnection == null) return;

      final offer = await _peerConnection!.createOffer({
        'offerToReceiveVideo': false,
        'offerToReceiveAudio': false,
      });
      await _peerConnection!.setLocalDescription(offer);

      await _waitForIceGatheringComplete();

      final localDesc = await _peerConnection!.getLocalDescription();
      if (localDesc == null) {
        throw Exception('LocalDescription is null after ICE gathering');
      }

      final provider = context.read<VisionProvider>();
      final url = 'http://${provider.host}:${provider.port}/offer';
      final response = await _sendHttpRequest(url, {
        'sdp': offer.sdp,
        'type': offer.type,
      });

      if (response != null) {
        final remoteDesc = RTCSessionDescription(
          response['sdp'],
          response['type'],
        );
        await _peerConnection!.setRemoteDescription(remoteDesc);
      } else {
        throw Exception('No SDP answer from server');
      }
    } catch (e) {
      print('Error connecting to server: $e');
      setState(() {
        _isConnecting = false;
      });
    }
  }

  Future<Map<String, dynamic>?> _sendOfferToServer(
    RTCSessionDescription offer,
  ) async {
    try {
      final provider = context.read<VisionProvider>();
      final url = 'http://${provider.host}:${provider.port}/offer';
      final response = await _sendHttpRequest(url, {
        'sdp': offer.sdp,
        'type': offer.type,
      });

      if (response != null) {
        return response;
      }
    } catch (e) {
      print('Error sending offer: $e');
    }
    return null;
  }

  Future<Map<String, dynamic>?> _sendHttpRequest(
    String url,
    Map<String, dynamic> data,
  ) async {
    try {
      final res = await http
          .post(
            Uri.parse(url),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(data),
          )
          .timeout(const Duration(seconds: 10));

      if (res.statusCode == 200) {
        return jsonDecode(res.body) as Map<String, dynamic>;
      } else {
        print('HTTP status ${res.statusCode}: ${res.body}');
      }
    } catch (e) {
      print('HTTP request error: $e');
    }
    return null;
  }

  void _onStreamingStarted() {
    print('WebRTC streaming started');
    final provider = context.read<VisionProvider>();
    provider.startStreaming();
    provider.setConnectionStatus("WebRTC Connected");
  }

  void _stopStreaming() {
    _connectionTimer?.cancel();
    final provider = context.read<VisionProvider>();
    provider.stopStreaming();
    provider.setConnectionStatus("WebRTC Disconnected");
  }

  void _disposeWebRTC() {
    _connectionTimer?.cancel();
    _localStream?.dispose();
    _peerConnection?.dispose();
    _localRenderer.dispose();
  }

  void _handleTap() {
    _tapCount++;
    _tapTimer?.cancel();

    if (_tapCount == 1) {
      _tapTimer = Timer(const Duration(milliseconds: 500), () {
        _tapCount = 0;
      });
    } else if (_tapCount == 2) {
      _tapTimer?.cancel();
      _handleDoubleTap();
      _tapCount = 0;
    }
  }

  void _handleDoubleTap() {
    _stopStreaming();
    widget.onBackPressed?.call();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        onTap: _handleTap,
        child: _isInitialized
            ? RTCVideoView(
                _localRenderer,
                objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
              )
            : Container(
                color: Colors.black,
                child: const Center(
                  child: CircularProgressIndicator(color: Colors.white),
                ),
              ),
      ),
    );
  }
}
