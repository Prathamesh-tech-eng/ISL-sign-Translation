package com.isl.signbridge

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.isl.signbridge.databinding.ActivityMainBinding
import com.isl.signbridge.ml.HandLandmarkerHelper
import com.isl.signbridge.ui.BorderController
import com.isl.signbridge.ui.RecognitionState
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// ─────────────────────────────────────────────────────────────────────────────
// MainActivity
//
// Responsibilities:
//   1. Request camera permission
//   2. Initialize CameraX (PreviewView + ImageAnalysis)
//   3. Set up MediaPipe HandLandmarker in LIVE_STREAM mode
//   4. Observe ViewModel LiveData and update UI
//   5. Route camera frames → ViewModel → TFLite → LLM API
// ─────────────────────────────────────────────────────────────────────────────
class MainActivity : AppCompatActivity(), HandLandmarkerHelper.LandmarkListener {

    companion object {
        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION = Manifest.permission.CAMERA
    }

    // ── View Binding ─────────────────────────────────────────────────────────
    private lateinit var binding: ActivityMainBinding

    // ── ViewModel ────────────────────────────────────────────────────────────
    private val viewModel: SignLanguageViewModel by viewModels()

    // ── Camera infrastructure ─────────────────────────────────────────────────
    private lateinit var cameraExecutor: ExecutorService
    private var handLandmarkerHelper: HandLandmarkerHelper? = null

    // ── UI feedback controller ────────────────────────────────────────────────
    private lateinit var borderController: BorderController

    // ── Permission launcher ───────────────────────────────────────────────────
    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(
                this,
                "Camera permission is required for ISL recognition",
                Toast.LENGTH_LONG
            ).show()
            finish()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Keep screen on during active signing
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Make the app truly full-screen (below status bar)
        window.setDecorFitsSystemWindows(false)

        setupBorderController()
        setupClickListeners()
        observeViewModel()
        checkCameraPermission()
    }

    override fun onDestroy() {
        super.onDestroy()
        handLandmarkerHelper?.close()
        cameraExecutor.shutdown()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UI Setup
    // ─────────────────────────────────────────────────────────────────────────

    private fun setupBorderController() {
        borderController = BorderController(
            overlayView = binding.overlayView,
            tvStateBadge = binding.tvStateBadge,
            statusDot = binding.statusDot,
            apiProgressBar = binding.apiProgressBar
        )
    }

    private fun setupClickListeners() {
        binding.btnClear.setOnClickListener {
            viewModel.clearAll()
            Toast.makeText(this, "Cleared", Toast.LENGTH_SHORT).show()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ViewModel Observers — Update UI from LiveData
    // ─────────────────────────────────────────────────────────────────────────

    private fun observeViewModel() {

        // ── State → Border color ──────────────────────────────────────────────
        viewModel.recognitionState.observe(this) { state ->
            when (state) {
                RecognitionState.IDLE -> borderController.setIdle()
                RecognitionState.WORD_MODE -> borderController.setWordMode()
                RecognitionState.SPELL_MODE -> {
                    borderController.setSpellMode()
                    binding.tvSpellHint.visibility = android.view.View.VISIBLE
                }
                RecognitionState.API_CALLING -> borderController.setApiCalling()
                else -> {}
            }
            if (state != RecognitionState.SPELL_MODE) {
                binding.tvSpellHint.visibility = android.view.View.GONE
            }
        }

        // ── Raw gloss string ──────────────────────────────────────────────────
        viewModel.rawGlossString.observe(this) { gloss ->
            binding.tvRawGloss.text = gloss.ifBlank { "..." }
            val wordCount = if (gloss.isBlank()) 0 else gloss.trim().split(" ").size
            binding.tvWordCount.text = "$wordCount word${if (wordCount == 1) "" else "s"}"
        }

        // ── Final LLM sentence ────────────────────────────────────────────────
        viewModel.finalSentence.observe(this) { sentence ->
            binding.tvFinalSentence.text = sentence
            // Signal API complete
            if (viewModel.recognitionState.value == RecognitionState.API_CALLING) {
                borderController.setApiComplete()
            }
        }

        // ── Errors → Toast ────────────────────────────────────────────────────
        viewModel.errorMessage.observe(this) { error ->
            error ?: return@observe
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show()
            viewModel.clearErrorMessage()
        }

        // ── Landmarks → Overlay view ──────────────────────────────────────────
        viewModel.latestLandmarks.observe(this) { triple ->
            triple?.let { (result, width, height) ->
                binding.overlayView.updateLandmarks(result, width, height)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Camera Permission
    // ─────────────────────────────────────────────────────────────────────────

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            shouldShowRequestPermissionRationale(CAMERA_PERMISSION) -> {
                Toast.makeText(
                    this,
                    "Camera is needed to detect hand signs",
                    Toast.LENGTH_LONG
                ).show()
                cameraPermissionLauncher.launch(CAMERA_PERMISSION)
            }
            else -> {
                cameraPermissionLauncher.launch(CAMERA_PERMISSION)
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CameraX Setup
    // ─────────────────────────────────────────────────────────────────────────

    private fun startCamera() {
        // Dedicated thread for camera image analysis (off the main thread)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize MediaPipe on camera thread
        cameraExecutor.execute {
            handLandmarkerHelper = HandLandmarkerHelper(this, this)
            handLandmarkerHelper!!.setupHandLandmarker()
        }

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases(cameraProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases(cameraProvider: ProcessCameraProvider) {
        // ── Use the FRONT camera for signing (mirrors user) ───────────────────
        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

        // ── Preview use case ──────────────────────────────────────────────────
        val preview = Preview.Builder().build().also { preview ->
            preview.setSurfaceProvider(binding.previewView.surfaceProvider)
        }

        // ── Image Analysis use case ───────────────────────────────────────────
        // STRATEGY_KEEP_ONLY_LATEST drops frames that can't be processed in time
        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { analysis ->
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    processImageProxy(imageProxy)
                }
            }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )
            Log.i(TAG, "Camera bound successfully.")
        } catch (e: Exception) {
            Log.e(TAG, "CameraX use case binding failed: ${e.message}", e)
            Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_SHORT).show()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Frame Processing Pipeline
    //   ImageProxy (camera frame) → Bitmap → MediaPipe → ViewModel
    // ─────────────────────────────────────────────────────────────────────────

    private fun processImageProxy(imageProxy: ImageProxy) {
        // The camera sensor delivers frames ROTATED relative to the screen.
        // imageInfo.rotationDegrees tells us how much to rotate clockwise to get upright.
        // We must correct this BEFORE passing to MediaPipe, so that the returned landmark
        // coordinates (0–1, 0–1) map correctly to the portrait overlay.
        val rawBitmap = imageProxy.toBitmap()
        val degrees = imageProxy.imageInfo.rotationDegrees

        val bitmap = if (degrees != 0) {
            val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
            val rotated = Bitmap.createBitmap(
                rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true
            )
            rawBitmap.recycle()   // Free the unrotated copy immediately
            rotated
        } else {
            rawBitmap
        }

        handLandmarkerHelper?.detectLiveStream(bitmap, isFrontCamera = true)
        imageProxy.close()  // CRITICAL: must close to receive next frame
    }

    // ─────────────────────────────────────────────────────────────────────────
    // HandLandmarkerHelper.LandmarkListener implementation
    // These callbacks come from the MediaPipe background thread
    // ─────────────────────────────────────────────────────────────────────────

    override fun onResults(result: HandLandmarkerResult, inputWidth: Int, inputHeight: Int) {
        // Route results to ViewModel (which posts to LiveData safely)
        viewModel.processLandmarkResult(result, inputWidth, inputHeight)
    }

    override fun onError(error: String) {
        Log.e(TAG, "HandLandmarker error: $error")
        runOnUiThread {
            Toast.makeText(this, "MediaPipe: $error", Toast.LENGTH_SHORT).show()
        }
    }
}
