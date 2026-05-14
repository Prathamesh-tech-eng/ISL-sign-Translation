package com.isl.signbridge

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.isl.signbridge.api.LLMRepository
import com.isl.signbridge.ml.HandLandmarkerHelper
import com.isl.signbridge.ml.TFLiteInferenceHelper
import com.isl.signbridge.ui.RecognitionState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.sqrt

// ─────────────────────────────────────────────────────────────────────────────
// SignLanguageViewModel
//
// THE CORE BRAIN of the app.
//
// Contains the complete state machine:
//   IDLE → WORD_MODE (hands appear)
//   WORD_MODE → [inference] → append word to gloss string
//              → if low confidence → SPELL_MODE
//   SPELL_MODE → letter-by-letter spelling
//              → swipe gesture → finalize spelled word → WORD_MODE
//   Hands dropped >3s → LLM API call → final sentence shown
//
// Architecture: AndroidViewModel (to hold Application context for TFLite)
// ─────────────────────────────────────────────────────────────────────────────
class SignLanguageViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "SignLangViewModel"

        // ── Velocity tracking thresholds (normalized coords) ─────────────────
        // A "micro-pause" in fingerspelling — wrist barely moving
        private const val SPELL_PAUSE_THRESHOLD = 0.008f
        // Minimum wrist speed for a "swipe" exit gesture
        private const val SWIPE_EXIT_THRESHOLD = 0.045f
        // Swipe must be predominantly horizontal (x-movement >> y-movement)
        private const val SWIPE_HORIZONTAL_RATIO = 2.5f

        // ── Timing constants ─────────────────────────────────────────────────
        // Hands absent for 3s → trigger LLM
        private const val HANDS_ABSENT_TIMEOUT_MS = 3000L
        // Micro-pause duration before capturing a fingerspelling letter
        private const val SPELL_CAPTURE_PAUSE_MS = 400L
    }

    // ── Dependencies ─────────────────────────────────────────────────────────
    private val tfliteHelper = TFLiteInferenceHelper(application)
    private val llmRepository = LLMRepository()

    // ─────────────────────────────────────────────────────────────────────────
    // LiveData — Observed by MainActivity to update the UI
    // ─────────────────────────────────────────────────────────────────────────

    private val _recognitionState = MutableLiveData(RecognitionState.IDLE)
    val recognitionState: LiveData<RecognitionState> = _recognitionState

    /** The building raw gloss string (e.g., "I HOSPITAL GO") */
    private val _rawGlossString = MutableLiveData("")
    val rawGlossString: LiveData<String> = _rawGlossString

    /** The current word being spelled in SPELL_MODE (e.g., "D-O-C-T") */
    private val _spelledWord = MutableLiveData("")
    val spelledWord: LiveData<String> = _spelledWord

    /** Final LLM-smoothed English sentence */
    private val _finalSentence = MutableLiveData("Translation will appear here...")
    val finalSentence: LiveData<String> = _finalSentence

    /** Error message for Toasts */
    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage

    /** Latest hand landmarks for the overlay view */
    private val _latestLandmarks = MutableLiveData<Triple<HandLandmarkerResult, Int, Int>?>()
    val latestLandmarks: LiveData<Triple<HandLandmarkerResult, Int, Int>?> = _latestLandmarks

    // ─────────────────────────────────────────────────────────────────────────
    // Internal state
    // ─────────────────────────────────────────────────────────────────────────

    /** Accumulated landmark frames for LSTM word inference */
    private val frameBuffer = mutableListOf<FloatArray>()

    /** Current raw gloss words */
    private val glossWords = mutableListOf<String>()

    /** Letters accumulated in current spell-mode word */
    private val spellBuffer = StringBuilder()

    /** Wrist position from previous frame for velocity calculation */
    private var previousWristPos: Pair<Float, Float>? = null

    /** Timer coroutine — fires LLM call after hands absent for 3s */
    private var handsAbsentJob: Job? = null

    /** Timer for spell-mode micro-pause detection */
    private var spellPauseJob: Job? = null
    private var lastCapturedLetter: String? = null

    /** Are hands currently in frame? */
    private var handsPresent = false

    // ─────────────────────────────────────────────────────────────────────────
    // Initialization
    // ─────────────────────────────────────────────────────────────────────────
    init {
        viewModelScope.launch(Dispatchers.IO) {
            if (!tfliteHelper.initialize()) {
                // Show the actual exception so it's diagnosable without Logcat
                _errorMessage.postValue(
                    "TFLite load failed: ${tfliteHelper.lastError}"
                )
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MAIN ENTRY POINT
    // Called from HandLandmarkerHelper's callback (MediaPipe thread)
    // ─────────────────────────────────────────────────────────────────────────

    fun processLandmarkResult(result: HandLandmarkerResult, imgWidth: Int, imgHeight: Int) {
        // Post the raw landmarks for drawing
        _latestLandmarks.postValue(Triple(result, imgWidth, imgHeight))

        val hasHands = result.landmarks().isNotEmpty()

        if (hasHands) {
            onHandsDetected(result)
        } else {
            onHandsAbsent()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STATE: HANDS PRESENT
    // ─────────────────────────────────────────────────────────────────────────

    private fun onHandsDetected(result: HandLandmarkerResult) {
        // Cancel any "hands absent" timer
        handsAbsentJob?.cancel()
        handsAbsentJob = null
        handsPresent = true

        val currentState = _recognitionState.value ?: RecognitionState.IDLE

        // Transition: IDLE → WORD_MODE
        if (currentState == RecognitionState.IDLE || currentState == RecognitionState.API_CALLING) {
            _recognitionState.postValue(RecognitionState.WORD_MODE)
        }

        // Get flattened landmark array
        val helper = HandLandmarkerHelper(getApplication(), object : HandLandmarkerHelper.LandmarkListener {
            override fun onResults(r: HandLandmarkerResult, w: Int, h: Int) {}
            override fun onError(error: String) {}
        })
        // Use static approach: flatten the result directly
        val landmarks = flattenLandmarks(result)

        // Calculate wrist velocity for gesture detection
        val wristPos = getWristPosition(result)
        val velocity = calculateVelocity(previousWristPos, wristPos)
        previousWristPos = wristPos

        when (_recognitionState.value) {
            RecognitionState.WORD_MODE -> handleWordMode(landmarks, result)
            RecognitionState.SPELL_MODE -> handleSpellMode(landmarks, velocity, wristPos)
            else -> {}
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // WORD MODE — Buffer frames, infer on hand drop
    // ─────────────────────────────────────────────────────────────────────────

    private fun handleWordMode(landmarks: FloatArray?, result: HandLandmarkerResult) {
        landmarks ?: return
        frameBuffer.add(landmarks)
        // Cap buffer to prevent unbounded memory growth
        if (frameBuffer.size > 120) {
            frameBuffer.removeAt(0)
        }
    }

    /**
     * Called when hands just dropped while in WORD_MODE.
     * Runs the LSTM inference on the buffered frames.
     */
    private fun runWordInference() {
        if (frameBuffer.size < TFLiteInferenceHelper.MIN_FRAMES_FOR_WORD) {
            Log.d(TAG, "Buffer too small (${frameBuffer.size} frames), skipping inference.")
            frameBuffer.clear()
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            val result = tfliteHelper.runWordInference(frameBuffer.toList())
            frameBuffer.clear()

            Log.i(TAG, "Word inference: ${result.label} @ ${(result.confidence * 100).toInt()}%")

            if (result.isHighConfidence && result.label != "UNKNOWN") {
                // ── HIGH CONFIDENCE → append word to gloss ────────────────────
                glossWords.add(result.label)
                _rawGlossString.postValue(glossWords.joinToString(" "))
            } else {
                // ── LOW CONFIDENCE → trigger fingerspelling fallback ──────────
                _recognitionState.postValue(RecognitionState.SPELL_MODE)
                _errorMessage.postValue("Word not recognized. Please spell it.")
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SPELL MODE — Letter-by-letter DNN on micro-pause frames
    // ─────────────────────────────────────────────────────────────────────────

    private fun handleSpellMode(
        landmarks: FloatArray?,
        velocity: Float,
        wristPos: Pair<Float, Float>?
    ) {
        landmarks ?: return

        // ── Check for exit swipe gesture ──────────────────────────────────────
        if (isSwipeGesture(velocity, wristPos)) {
            finalizeSpelledWord()
            return
        }

        // ── Micro-pause detection: velocity below threshold ───────────────────
        if (velocity < SPELL_PAUSE_THRESHOLD) {
            // Debounce: capture after sustained pause
            if (spellPauseJob == null) {
                spellPauseJob = viewModelScope.launch(Dispatchers.IO) {
                    delay(SPELL_CAPTURE_PAUSE_MS)
                    captureSpellLetter(landmarks)
                    spellPauseJob = null
                }
            }
        } else {
            // Movement resumed — cancel pending capture
            spellPauseJob?.cancel()
            spellPauseJob = null
        }
    }

    private fun captureSpellLetter(landmarks: FloatArray) {
        val result = tfliteHelper.runAlphabetInference(landmarks)
        if (result.isHighConfidence && result.label != lastCapturedLetter) {
            lastCapturedLetter = result.label
            spellBuffer.append(result.label)
            _spelledWord.postValue("Spelling: ${spellBuffer}")
            // Show partial spelled word in raw gloss display
            updateRawGlossWithSpelling()
        }
    }

    private fun updateRawGlossWithSpelling() {
        val current = glossWords.joinToString(" ")
        val spelling = spellBuffer.toString()
        _rawGlossString.postValue(
            if (current.isNotEmpty()) "$current [${spelling}_]"
            else "[${spelling}_]"
        )
    }

    /**
     * Exit trigger: rapid horizontal wrist movement.
     * Check that velocity is large AND predominantly horizontal.
     */
    private fun isSwipeGesture(velocity: Float, wristPos: Pair<Float, Float>?): Boolean {
        if (velocity < SWIPE_EXIT_THRESHOLD) return false
        val prev = previousWristPos ?: return false
        val curr = wristPos ?: return false
        val dx = Math.abs(curr.first - prev.first)
        val dy = Math.abs(curr.second - prev.second)
        return dy > 0 && (dx / dy) > SWIPE_HORIZONTAL_RATIO
    }

    /** Finalize the spelled word and return to WORD_MODE */
    private fun finalizeSpelledWord() {
        val word = spellBuffer.toString()
        if (word.isNotBlank()) {
            glossWords.add(word)
            _rawGlossString.postValue(glossWords.joinToString(" "))
        }
        spellBuffer.clear()
        lastCapturedLetter = null
        spellPauseJob?.cancel()
        spellPauseJob = null
        _spelledWord.postValue("")
        _recognitionState.postValue(RecognitionState.WORD_MODE)
        Log.i(TAG, "Spelled word finalized: '$word'. Returned to WORD_MODE.")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STATE: HANDS ABSENT
    // ─────────────────────────────────────────────────────────────────────────

    private fun onHandsAbsent() {
        val wasPresent = handsPresent
        handsPresent = false

        val currentState = _recognitionState.value

        // If we just dropped from WORD_MODE, run word inference on the buffered frames
        if (wasPresent && currentState == RecognitionState.WORD_MODE && frameBuffer.isNotEmpty()) {
            runWordInference()
        }

        // Transition to IDLE visually (only on the transition frame)
        if (wasPresent &&
            (currentState == RecognitionState.WORD_MODE || currentState == RecognitionState.SPELL_MODE)
        ) {
            _recognitionState.postValue(RecognitionState.IDLE)
        }

        // ── Start the 3-second LLM countdown ONLY on the first absent frame ─────
        // Bug fix: previously handsAbsentJob was cancelled and restarted every frame
        // (~30fps), so the 3-second timer was perpetually reset and never fired.
        // Now we only start it once when hands JUST disappeared (wasPresent == true).
        if (wasPresent && glossWords.isNotEmpty() && currentState != RecognitionState.API_CALLING) {
            handsAbsentJob?.cancel()
            handsAbsentJob = viewModelScope.launch {
                delay(HANDS_ABSENT_TIMEOUT_MS)   // 3 seconds — uninterrupted
                triggerLLMCall()
            }
            Log.d(TAG, "LLM countdown started (${HANDS_ABSENT_TIMEOUT_MS}ms). Gloss: ${glossWords.joinToString(" ")}")
        }
        // Do NOT touch handsAbsentJob on subsequent absent frames —
        // let the running countdown complete naturally.
    }

    // ─────────────────────────────────────────────────────────────────────────
    // LLM API CALL
    // ─────────────────────────────────────────────────────────────────────────

    private fun triggerLLMCall() {
        val rawGloss = glossWords.joinToString(" ")
        if (rawGloss.isBlank()) return

        Log.i(TAG, "Triggering LLM API for: \"$rawGloss\"")
        _recognitionState.postValue(RecognitionState.API_CALLING)

        viewModelScope.launch(Dispatchers.IO) {
            val result = llmRepository.translateGlossToSentence(rawGloss)

            result.onSuccess { sentence ->
                Log.i(TAG, "LLM response: \"$sentence\"")
                _finalSentence.postValue(sentence)
            }.onFailure { error ->
                Log.e(TAG, "LLM call failed: ${error.message}")
                _errorMessage.postValue("Translation failed: ${error.message}")
                // Fallback: show raw gloss as the sentence
                _finalSentence.postValue(rawGloss)
            }

            _recognitionState.postValue(RecognitionState.IDLE)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public actions from MainActivity
    // ─────────────────────────────────────────────────────────────────────────

    /** Called when the user taps the "Clear" button */
    fun clearAll() {
        handsAbsentJob?.cancel()
        handsAbsentJob = null
        spellPauseJob?.cancel()
        spellPauseJob = null
        frameBuffer.clear()
        glossWords.clear()
        spellBuffer.clear()
        previousWristPos = null
        lastCapturedLetter = null
        _rawGlossString.postValue("")
        _spelledWord.postValue("")
        _finalSentence.postValue("Translation will appear here...")
        _recognitionState.postValue(RecognitionState.IDLE)
    }

    fun clearErrorMessage() {
        _errorMessage.postValue(null)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Geometry utilities (inlined to avoid creating a helper object per frame)
    // ─────────────────────────────────────────────────────────────────────────

    private fun flattenLandmarks(result: HandLandmarkerResult): FloatArray? {
        if (result.landmarks().isEmpty()) return null
        val flat = FloatArray(126) { 0f }
        result.landmarks().forEachIndexed { hi, hand ->
            if (hi >= 2) return@forEachIndexed
            val off = hi * 63
            hand.forEachIndexed { li, lm ->
                flat[off + li * 3 + 0] = lm.x()
                flat[off + li * 3 + 1] = lm.y()
                flat[off + li * 3 + 2] = lm.z()
            }
        }
        return flat
    }

    private fun getWristPosition(result: HandLandmarkerResult): Pair<Float, Float>? {
        val wrist = result.landmarks().firstOrNull()?.firstOrNull() ?: return null
        return Pair(wrist.x(), wrist.y())
    }

    private fun calculateVelocity(
        prev: Pair<Float, Float>?,
        curr: Pair<Float, Float>?
    ): Float {
        if (prev == null || curr == null) return 0f
        val dx = curr.first - prev.first
        val dy = curr.second - prev.second
        return sqrt((dx * dx + dy * dy).toDouble()).toFloat()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────

    override fun onCleared() {
        super.onCleared()
        tfliteHelper.close()
        handsAbsentJob?.cancel()
        spellPauseJob?.cancel()
    }
}
