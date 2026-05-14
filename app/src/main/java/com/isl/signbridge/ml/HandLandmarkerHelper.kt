package com.isl.signbridge.ml

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult

// ─────────────────────────────────────────────────────────────────────────────
// HandLandmarkerHelper
//
// Wraps MediaPipe Hand Landmark detection in LIVE_STREAM mode.
// Provides normalized (x,y,z) landmark coordinates for up to 2 hands.
//
// SETUP:
//   The MediaPipe model 'hand_landmarker.task' is bundled automatically
//   when you add the MediaPipe dependency. No manual asset needed.
// ─────────────────────────────────────────────────────────────────────────────
class HandLandmarkerHelper(
    private val context: Context,
    private val listener: LandmarkListener
) {
    companion object {
        private const val TAG = "HandLandmarkerHelper"
        private const val MP_HAND_LANDMARKER_TASK = "hand_landmarker.task"
        private const val NUM_HANDS = 2
        private const val MIN_HAND_DETECTION_CONFIDENCE = 0.6f
        private const val MIN_HAND_PRESENCE_CONFIDENCE = 0.6f
        private const val MIN_TRACKING_CONFIDENCE = 0.6f
    }

    // ── MediaPipe Hand Landmarker instance ───────────────────────────────────
    private var handLandmarker: HandLandmarker? = null

    // ─────────────────────────────────────────────────────────────────────────
    // Callback interface for landmark results
    // ─────────────────────────────────────────────────────────────────────────
    interface LandmarkListener {
        /**
         * Called on every frame with results.
         * @param results    MediaPipe result (may contain 0, 1, or 2 hands)
         * @param inputWidth  Width of the input image
         * @param inputHeight Height of the input image
         */
        fun onResults(
            results: HandLandmarkerResult,
            inputWidth: Int,
            inputHeight: Int
        )

        /**
         * Called when MediaPipe encounters an error.
         */
        fun onError(error: String)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Setup — call once during initialization
    // ─────────────────────────────────────────────────────────────────────────
    fun setupHandLandmarker() {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MP_HAND_LANDMARKER_TASK)
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(NUM_HANDS)
                .setMinHandDetectionConfidence(MIN_HAND_DETECTION_CONFIDENCE)
                .setMinHandPresenceConfidence(MIN_HAND_PRESENCE_CONFIDENCE)
                .setMinTrackingConfidence(MIN_TRACKING_CONFIDENCE)
                // LIVE_STREAM mode: async callbacks, does not block camera thread
                .setRunningMode(RunningMode.LIVE_STREAM)
                .setResultListener { result, inputImage ->
                    // Called on MediaPipe's own thread — post to listener
                    // NOTE: dimensions come from MPImage, not HandLandmarkerResult
                    val inputImageWidth = inputImage.width
                    val inputImageHeight = inputImage.height
                    listener.onResults(result, inputImageWidth, inputImageHeight)
                }
                .setErrorListener { error ->
                    listener.onError(error.message ?: "Unknown MediaPipe error")
                }
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
            Log.i(TAG, "HandLandmarker initialized successfully.")

        } catch (e: Exception) {
            listener.onError("HandLandmarker setup failed: ${e.message}")
            Log.e(TAG, "HandLandmarker setup failed", e)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Detection — call this for every camera frame from ImageAnalysis
    // ─────────────────────────────────────────────────────────────────────────
    fun detectLiveStream(bitmap: Bitmap, isFrontCamera: Boolean) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        // Timestamp must be unique and monotonically increasing
        val frameTimestamp = SystemClock.uptimeMillis()
        handLandmarker?.detectAsync(mpImage, frameTimestamp)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utility: Flatten landmarks into a Float array of 126 values
    //
    //   Format: [hand0_lm0_x, hand0_lm0_y, hand0_lm0_z,
    //            hand0_lm1_x, ..., hand1_lm20_z]
    //   If only one hand detected, second hand slots are zero-padded.
    //   If no hands, returns null.
    // ─────────────────────────────────────────────────────────────────────────
    fun flattenLandmarks(result: HandLandmarkerResult): FloatArray? {
        if (result.landmarks().isEmpty()) return null

        val flat = FloatArray(126) { 0f }  // 21 landmarks × 3 × 2 hands

        result.landmarks().forEachIndexed { handIndex, handLandmarks ->
            if (handIndex >= 2) return@forEachIndexed  // Max 2 hands
            val offset = handIndex * 63   // 21 landmarks × 3 coords
            handLandmarks.forEachIndexed { lmIndex, landmark ->
                flat[offset + lmIndex * 3 + 0] = landmark.x()
                flat[offset + lmIndex * 3 + 1] = landmark.y()
                flat[offset + lmIndex * 3 + 2] = landmark.z()
            }
        }
        return flat
    }

    /**
     * Extracts the wrist landmark position for velocity tracking.
     * Landmark 0 is the wrist in MediaPipe's hand topology.
     *
     * @param result  MediaPipe result
     * @param handIndex  0 = first detected hand, 1 = second
     * @return Pair(x, y) normalized to [0, 1], or null if no hand detected
     */
    fun getWristPosition(result: HandLandmarkerResult, handIndex: Int = 0): Pair<Float, Float>? {
        if (result.landmarks().size <= handIndex) return null
        val wrist = result.landmarks()[handIndex].firstOrNull() ?: return null
        return Pair(wrist.x(), wrist.y())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────
    fun close() {
        handLandmarker?.close()
        handLandmarker = null
    }
}
