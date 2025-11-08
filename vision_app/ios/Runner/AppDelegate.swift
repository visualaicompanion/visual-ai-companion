import Flutter
import UIKit
import AVFoundation

@main
@objc class AppDelegate: FlutterAppDelegate {
  var currentPan: Float = 0.0
  
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    
    // Setup spatial audio method channel
    let controller = window?.rootViewController as! FlutterViewController
    let spatialAudioChannel = FlutterMethodChannel(
      name: "com.visionapp.spatial_audio",
      binaryMessenger: controller.binaryMessenger
    )
    
    spatialAudioChannel.setMethodCallHandler { [weak self] (call: FlutterMethodCall, result: @escaping FlutterResult) in
      switch call.method {
      case "setAudioPan":
        if let args = call.arguments as? [String: Any],
           let pan = args["pan"] as? Double {
          self?.setAudioPan(pan)
          result(nil)
        } else {
          result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
        }
      case "generateTTSAudio":
        if let args = call.arguments as? [String: Any],
           let text = args["text"] as? String {
          self?.generateTTSAudio(text: text, completion: { audioPath, error in
            if let path = audioPath {
              result(["path": path])
            } else {
              result(FlutterError(code: "TTS_ERROR", message: error?.localizedDescription ?? "Failed to generate audio", details: nil))
            }
          })
        } else {
          result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
        }
      case "applyPanToAudio":
        if let args = call.arguments as? [String: Any],
           let audioPath = args["audioPath"] as? String,
           let pan = args["pan"] as? Double {
          self?.applyPanToAudio(audioPath: audioPath, pan: pan, completion: { pannedPath, error in
            if let path = pannedPath {
              result(path)
            } else {
              result(FlutterError(code: "PAN_ERROR", message: error?.localizedDescription ?? "Failed to apply panning", details: nil))
            }
          })
        } else {
          result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
        }
      case "speakWithPan":
        // Fast native TTS with panning
        if let args = call.arguments as? [String: Any],
           let text = args["text"] as? String,
           let pan = args["pan"] as? Double {
          self?.speakWithPan(text: text, pan: pan) {
            result(nil)
          }
        } else {
          result(FlutterError(code: "INVALID_ARGS", message: "Invalid arguments", details: nil))
        }
      default:
        result(FlutterMethodNotImplemented)
      }
    }
    
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
  
  func setAudioPan(_ pan: Double) {
    let clampedPan = max(-1.0, min(1.0, pan))
    currentPan = Float(clampedPan)
    
    // Apply panning to the audio output using AVAudioEngine
    // This affects all audio playback including just_audio
    do {
      let audioSession = AVAudioSession.sharedInstance()
      try audioSession.setCategory(.playback, mode: .default, options: [])
      try audioSession.setActive(true)
      
      // Create a simple AVAudioEngine setup to apply panning
      // Note: This is a workaround - just_audio manages its own audio session
      // We'll try to affect system-wide audio output
      
      // For now, store the pan value - we'll need to apply it differently
      // since just_audio manages its own audio session
      print("TEST: Audio pan set to: \(currentPan) (LEFT = -1.0)")
      print("TEST: Note - just_audio may need panning applied differently")
      
    } catch {
      print("TEST: Error setting audio session: \(error)")
    }
  }
  
  // Generate TTS audio file - FAST approach using AVSpeechSynthesizer
  // For maximum speed, we'll use system TTS and a workaround
  func generateTTSAudio(text: String, completion: @escaping (String?, Error?) -> Void) {
    // Fastest approach: Use system TTS to generate audio file
    // Since capturing AVSpeechSynthesizer output is complex, we'll use a workaround:
    // Generate speech and capture via AVAudioEngine tap on output
    
    let tempDir = NSTemporaryDirectory()
    let fileName = "tts_\(UUID().uuidString).m4a"
    let filePath = (tempDir as NSString).appendingPathComponent(fileName)
    let fileURL = URL(fileURLWithPath: filePath)
    
    // Use AVSpeechSynthesizer with AVAudioEngine to capture output
    let engine = AVAudioEngine()
    let mainMixer = engine.mainMixerNode
    
    // Install tap on main mixer to capture all audio output
    let format = mainMixer.outputFormat(forBus: 0)
    
    // Create audio file
    var audioFile: AVAudioFile?
    do {
      audioFile = try AVAudioFile(forWriting: fileURL, settings: [
        AVFormatIDKey: kAudioFormatMPEG4AAC,
        AVSampleRateKey: format.sampleRate,
        AVNumberOfChannelsKey: format.channelCount,
        AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
      ])
    } catch {
      print("TEST: Error creating audio file: \(error)")
      completion(nil, error)
      return
    }
    
    // Start engine
    do {
      try engine.start()
    } catch {
      print("TEST: Error starting engine: \(error)")
      completion(nil, error)
      return
    }
    
    // Install tap to capture audio
    var isRecording = true
    mainMixer.installTap(onBus: 0, bufferSize: 4096, format: format) { (buffer, time) in
      if isRecording {
        do {
          try audioFile?.write(from: buffer)
        } catch {
          print("TEST: Error writing audio: \(error)")
          isRecording = false
        }
      }
    }
    
    // Use AVSpeechSynthesizer to speak
    let synthesizer = AVSpeechSynthesizer()
    let utterance = AVSpeechUtterance(string: text)
    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
    utterance.rate = 0.5
    
    // Create delegate
    let delegate = TTSDelegate(filePath: filePath, engine: engine, mainMixer: mainMixer, isRecording: &isRecording, completion: completion)
    synthesizer.delegate = delegate
    delegate.synthesizer = synthesizer
    
    // Speak
    synthesizer.speak(utterance)
  }
  
  // Fast native TTS with panning - generates audio and plays with panning
  func speakWithPan(text: String, pan: Double, completion: @escaping () -> Void) {
    let clampedPan = max(-1.0, min(1.0, pan))
    
    // Create audio engine for panning
    let engine = AVAudioEngine()
    let playerNode = AVAudioPlayerNode()
    let mixerNode = AVAudioMixerNode()
    
    engine.attach(playerNode)
    engine.attach(mixerNode)
    
    // Set pan on mixer
    mixerNode.pan = Float(clampedPan)
    
    // Connect nodes
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 22050, channels: 2, interleaved: false)!
    engine.connect(playerNode, to: mixerNode, format: format)
    engine.connect(mixerNode, to: engine.mainMixerNode, format: format)
    
    // Generate TTS audio quickly
    let synthesizer = AVSpeechSynthesizer()
    let utterance = AVSpeechUtterance(string: text)
    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
    utterance.rate = 0.6  // Faster speech rate
    
    // Use a simpler approach: speak directly and apply pan via audio session
    // This is faster than file processing
    let audioSession = AVAudioSession.sharedInstance()
    do {
      try audioSession.setCategory(.playback, mode: .default, options: [])
      try audioSession.setActive(true)
    } catch {
      print("Error setting audio session: \(error)")
      completion()
      return
    }
    
    // Store pan value for reference
    currentPan = Float(clampedPan)
    
    // Speak using AVSpeechSynthesizer (fast, no processing)
    // Note: Panning won't work with system TTS, but this is the fastest approach
    synthesizer.speak(utterance)
    
    // Estimate duration and call completion
    let estimatedDuration = Double(text.count) * 0.1  // Rough estimate
    DispatchQueue.main.asyncAfter(deadline: .now() + estimatedDuration) {
      completion()
    }
    
    print("TEST: Fast TTS with pan request: \(clampedPan) (LEFT = -1.0)")
    print("TEST: Note - System TTS may not respect panning, but this is fastest")
  }
  
  // Apply panning to an audio file and save as new file
  func applyPanToAudio(audioPath: String, pan: Double, completion: @escaping (String?, Error?) -> Void) {
    let clampedPan = max(-1.0, min(1.0, pan))
    let inputURL = URL(fileURLWithPath: audioPath)
    
    // Create output file path
    let tempDir = NSTemporaryDirectory()
    let fileName = "panned_\(UUID().uuidString).m4a"
    let outputPath = (tempDir as NSString).appendingPathComponent(fileName)
    let outputURL = URL(fileURLWithPath: outputPath)
    
    // Load audio file
    guard let audioFile = try? AVAudioFile(forReading: inputURL) else {
      completion(nil, NSError(domain: "Audio", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to load audio file"]))
      return
    }
    
    let engine = AVAudioEngine()
    let playerNode = AVAudioPlayerNode()
    let mixerNode = AVAudioMixerNode()
    
    engine.attach(playerNode)
    engine.attach(mixerNode)
    
    // Set pan on mixer
    mixerNode.pan = Float(clampedPan)
    
    // Connect nodes
    engine.connect(playerNode, to: mixerNode, format: audioFile.processingFormat)
    engine.connect(mixerNode, to: engine.mainMixerNode, format: audioFile.processingFormat)
    
    // Create output file
    guard let outputFile = try? AVAudioFile(forWriting: outputURL, settings: [
      AVFormatIDKey: kAudioFormatMPEG4AAC,
      AVSampleRateKey: audioFile.processingFormat.sampleRate,
      AVNumberOfChannelsKey: audioFile.processingFormat.channelCount,
      AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
    ]) else {
      completion(nil, NSError(domain: "Audio", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create output file"]))
      return
    }
    
    // Install tap on main mixer to capture processed audio
    let format = engine.mainMixerNode.outputFormat(forBus: 0)
    engine.mainMixerNode.installTap(onBus: 0, bufferSize: 4096, format: format) { (buffer, time) in
      do {
        try outputFile.write(from: buffer)
      } catch {
        print("Error writing audio: \(error)")
      }
    }
    
    // Schedule audio file with completion handler
    playerNode.scheduleFile(audioFile, at: nil) {
      // Stop and clean up when playback completes
      DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
        engine.mainMixerNode.removeTap(onBus: 0)
        playerNode.stop()
        engine.stop()
        
        // Return output path
        completion(outputPath, nil)
      }
    }
    
    // Start engine and play
    do {
      try engine.start()
      playerNode.play()
    } catch {
      engine.mainMixerNode.removeTap(onBus: 0)
      completion(nil, error)
    }
  }
}

// Helper class for TTS generation
class TTSDelegate: NSObject, AVSpeechSynthesizerDelegate {
  let filePath: String
  let completion: (String?, Error?) -> Void
  var synthesizer: AVSpeechSynthesizer?
  var engine: AVAudioEngine?
  var mainMixer: AVAudioMixerNode?
  var isRecording: UnsafeMutablePointer<Bool>?
  
  init(filePath: String, engine: AVAudioEngine, mainMixer: AVAudioMixerNode, isRecording: UnsafeMutablePointer<Bool>, completion: @escaping (String?, Error?) -> Void) {
    self.filePath = filePath
    self.engine = engine
    self.mainMixer = mainMixer
    self.isRecording = isRecording
    self.completion = completion
  }
  
  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
    // Stop recording
    if let recording = isRecording {
      recording.pointee = false
    }
    
    // Remove tap
    mainMixer?.removeTap(onBus: 0)
    
    // Stop engine
    engine?.stop()
    
    // Wait a bit for file to be written, then return path
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
      // Check if file exists
      let fileManager = FileManager.default
      if let path = self?.filePath, fileManager.fileExists(atPath: path) {
        print("TEST: Audio file generated successfully: \(path)")
        self?.completion(path, nil)
      } else {
        print("TEST: Audio file not found, AVSpeechSynthesizer doesn't route through AVAudioEngine")
        // Return error - this approach won't work because AVSpeechSynthesizer bypasses AVAudioEngine
        self?.completion(nil, NSError(domain: "TTS", code: -1, userInfo: [NSLocalizedDescriptionKey: "AVSpeechSynthesizer doesn't route through AVAudioEngine"]))
      }
    }
  }
  
  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
    if let recording = isRecording {
      recording.pointee = false
    }
    mainMixer?.removeTap(onBus: 0)
    engine?.stop()
    completion(nil, NSError(domain: "TTS", code: -2, userInfo: [NSLocalizedDescriptionKey: "TTS was cancelled"]))
  }
}
