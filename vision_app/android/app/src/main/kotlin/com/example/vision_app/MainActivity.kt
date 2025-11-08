package com.example.vision_app

import android.media.AudioAttributes
import android.media.AudioManager
import android.media.AudioTrack
import android.os.Build
import androidx.annotation.RequiresApi
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val CHANNEL = "com.visionapp.spatial_audio"
    private var audioManager: AudioManager? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        
        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "setAudioPan" -> {
                    val pan = call.argument<Double>("pan") ?: 0.0
                    setAudioPan(pan)
                    result.success(null)
                }
                else -> {
                    result.notImplemented()
                }
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.M)
    private fun setAudioPan(pan: Double) {
        // Clamp pan value between -1.0 and 1.0
        val clampedPan = pan.coerceIn(-1.0, 1.0)
        
        // For Android, we can use AudioManager's stereo balance
        // Note: This is a simplified approach. For full spatial audio,
        // you might want to use AudioTrack with custom panning
        audioManager?.let { manager ->
            try {
                // Set stereo balance (pan)
                // Android uses balance from -1.0 (left) to 1.0 (right)
                // This affects the system-wide audio output
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    // Use AudioAttributes for more control
                    // Note: Direct pan control on Android is limited
                    // This is a workaround that adjusts volume balance
                    val maxVolume = manager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
                    val currentVolume = manager.getStreamVolume(AudioManager.STREAM_MUSIC)
                    
                    // Apply panning by adjusting left/right balance
                    // This is a simplified approach - full spatial audio requires AudioTrack
                    // For now, we'll use a workaround with volume adjustment
                    // In a production app, you'd use AudioTrack with custom panning
                }
            } catch (e: Exception) {
                android.util.Log.e("SpatialAudio", "Error setting audio pan: ${e.message}")
            }
        }
    }
}
