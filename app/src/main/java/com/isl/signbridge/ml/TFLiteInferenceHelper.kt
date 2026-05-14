package com.isl.signbridge.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

// ─────────────────────────────────────────────────────────────────────────────
// TFLiteInferenceHelper
//
// Handles loading and running both TFLite models:
//   • word_model.tflite  — LSTM model for continuous gesture sequences
//   • alphabet_model.tflite — DNN model for single-frame fingerspelling
//
// HOW TO ADD YOUR MODELS:
//   1. In Android Studio, right-click app/src/main → New > Folder > Assets Folder
//   2. Copy your .tflite files into app/src/main/assets/
//   3. Files must be named exactly: word_model.tflite, alphabet_model.tflite
// ─────────────────────────────────────────────────────────────────────────────
class TFLiteInferenceHelper(private val context: Context) {

    companion object {
        private const val TAG = "TFLiteHelper"

        // ── Model file names (must match files in assets/) ──────────────────
        const val WORD_MODEL_FILE = "word_model.tflite"
        const val ALPHABET_MODEL_FILE = "alphabet_model.tflite"

        // ── Inference thresholds ─────────────────────────────────────────────
        const val WORD_CONFIDENCE_THRESHOLD = 0.50f   // 50% → recognized word (lower for testing)
        const val MIN_FRAMES_FOR_WORD = 10            // Minimum frames before word inference

        // ── Input dimensions ─────────────────────────────────────────────────
        // Each hand frame: 21 landmarks × 3 coords (x,y,z) × 2 hands = 126 floats
        const val HAND_LANDMARK_SIZE = 126
        // Word model expects a temporal sequence; adjust to match your training
        // e.g., if trained with 30-frame windows: SEQ_LENGTH = 30
        const val SEQ_LENGTH = 75

        // ── Labels — replace with your actual trained labels ─────────────────
        // These must match the integer indices your model was trained on
        val WORD_LABELS = arrayOf(
            "Actor", "Adult", "Afternoon", "Alright", "Animal", "Artist",
            "Attack", "Author", "Baby", "Bag", "Ball", "Bank",
            "Bathroom", "Beautiful", "Bed", "Bedroom", "Bicycle", "Bill",
            "Bird", "Black", "Blind", "Blue", "Boat", "Book",
            "Box", "Boy", "Brother", "Brown", "Bus", "Camera",
            "Car", "Card", "Cat", "Cell phone", "Chair", "Child",
            "City", "Clock", "Clothing", "Colour", "Computer", "Court",
            "Cow", "Crowd", "Daughter", "Deaf", "Death", "Doctor",
            "Dog", "Door", "Dream", "Dress", "Election", "Energy",
            "Evening", "Exercise", "Fall", "Family", "Fan", "Father",
            "Fish", "Friday", "Friend", "Gift", "Girl", "God",
            "Good Morning", "Good afternoon", "Good evening", "Good night", "Grandfather", "Grandmother",
            "Green", "Grey", "Ground", "Gun", "Hat", "Hello",
            "Horse", "Hospital", "Hour", "House", "How are you", "Husband",
            "I", "India", "Job", "Key", "King", "Kitchen",
            "Lamp", "Laptop", "Lawyer", "Letter", "Library", "Location",
            "Lock", "Man", "Manager", "Market", "Marriage", "Mean",
            "Medicine", "Minute", "Monday", "Money", "Monsoon", "Month",
            "Morning", "Mother", "Mouse", "Neighbour", "Newspaper", "Nice",
            "Night", "Office", "Orange", "Page", "Paint", "Pant",
            "Paper", "Parent", "Park", "Patient", "Peace", "Pen",
            "Pencil", "Photograph", "Pink", "Plane", "Player", "Pleased",
            "Pocket", "Police", "President", "Price", "Priest", "Queen",
            "Race (ethnicity)", "Radio", "Red", "Religion", "Reporter", "Restaurant",
            "Ring", "Saturday", "School", "Science", "Screen", "Season",
            "Second", "Secretary", "Shirt", "Shoes", "Sign", "Sister",
            "Skirt", "Soap", "Soldier", "Son", "Sport", "Spring",
            "Store or Shop", "Street or Road", "Student", "Suit", "Summer", "Sunday",
            "T-Shirt", "Table", "Teacher", "Team", "Technology", "Telephone",
            "Television", "Temple", "Thank you", "Thursday", "Time", "Today",
            "Tomorrow", "Tool", "Train", "Train Station", "Transportation", "Truck",
            "Tuesday", "Ugly", "University", "Waiter", "War", "Wednesday",
            "Week", "White", "Wife", "Window", "Winter", "Woman",
            "Year", "Yellow", "Yesterday", "alive", "bad", "big large",
            "cheap", "clean", "cold", "cool", "curved", "dead",
            "deep", "dirty", "dry", "expensive", "famous", "fast",
            "female", "flat", "good", "happy", "hard", "he",
            "healthy", "heavy", "high", "hot", "it", "light",
            "long", "loose", "loud", "low", "male", "narrow",
            "new", "old", "poor", "quiet", "rich", "sad",
            "shallow", "she", "short", "sick", "slow", "small little",
            "soft", "strong", "tall", "they", "thick", "thin",
            "tight", "train ticket", "warm", "we", "weak", "wet",
            "wide", "you", "you (plural)", "young",
            // ── Model outputs 262 classes; 32 extra labels below ─────────────
            // TODO: Replace these placeholders with your ACTUAL training labels
            // in the EXACT same order as your label encoder used during training.
            // Check your Colab training code for the labels list (e.g. le.classes_)
            "index_230", "index_231", "index_232", "index_233", "index_234", "index_235",
            "index_236", "index_237", "index_238", "index_239", "index_240", "index_241",
            "index_242", "index_243", "index_244", "index_245", "index_246", "index_247",
            "index_248", "index_249", "index_250", "index_251", "index_252", "index_253",
            "index_254", "index_255", "index_256", "index_257", "index_258", "index_259",
            "index_260", "index_261"
        )

        val ALPHABET_LABELS = arrayOf(
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z"
            // TODO: Add ISL-specific fingerspelling variants if needed
        )
    }

    // ── Interpreter instances ────────────────────────────────────────────────
    private var wordInterpreter: Interpreter? = null
    private var alphabetInterpreter: Interpreter? = null

    // ── Actual model shapes (read at runtime from tensor metadata) ───────────
    // These override the compile-time constants to avoid shape mismatch crashes
    private var actualWordSeqLen = SEQ_LENGTH
    private var actualWordOutputSize = WORD_LABELS.size
    private var actualAlphabetOutputSize = ALPHABET_LABELS.size

    // ── Last error message (exposed for diagnostic Toast in ViewModel) ──────────
    var lastError: String? = null
        private set

    // ── TFLite Options ───────────────────────────────────────────────────────
    // CPU-only execution with 4 threads.
    // NNAPI is intentionally disabled: the hardware NNAPI driver on most devices
    // does NOT support FULLY_CONNECTED op v12, causing model load to fail.
    // The CPU kernel handles v12 correctly.
    private val interpreterOptions = Interpreter.Options().apply {
        numThreads = 4
    }

    // ── Inference result data class ──────────────────────────────────────────
    data class InferenceResult(
        val label: String,
        val confidence: Float,
        val isHighConfidence: Boolean
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Initialization
    // ─────────────────────────────────────────────────────────────────────────

    fun initialize(): Boolean {
        return try {
            wordInterpreter = Interpreter(loadModelFile(WORD_MODEL_FILE), interpreterOptions)
            alphabetInterpreter = Interpreter(loadModelFile(ALPHABET_MODEL_FILE), interpreterOptions)

            // ── Read actual tensor shapes from the loaded models ──────────────
            // This prevents crashes when SEQ_LENGTH or label count doesn't match
            wordInterpreter!!.getInputTensor(0).shape().let { shape ->
                // shape = [batch, seqLen, features]
                if (shape.size >= 3) {
                    actualWordSeqLen = shape[1]
                    Log.i(TAG, "Word model input : ${shape.toList()}  → using seqLen=$actualWordSeqLen")
                }
            }
            wordInterpreter!!.getOutputTensor(0).shape().let { shape ->
                // shape = [batch, numClasses]
                if (shape.size >= 2) {
                    actualWordOutputSize = shape[1]
                    Log.i(TAG, "Word model output: ${shape.toList()}  → ${actualWordOutputSize} classes")
                    if (actualWordOutputSize > WORD_LABELS.size) {
                        Log.w(TAG, "Model has $actualWordOutputSize classes but only ${WORD_LABELS.size} labels defined — extra classes will show index")
                    }
                }
            }
            alphabetInterpreter!!.getOutputTensor(0).shape().let { shape ->
                if (shape.size >= 2) {
                    actualAlphabetOutputSize = shape[1]
                    Log.i(TAG, "Alphabet model output: ${shape.toList()}  → ${actualAlphabetOutputSize} classes")
                }
            }

            Log.i(TAG, "Both TFLite models loaded. WordSeqLen=$actualWordSeqLen, WordClasses=$actualWordOutputSize, AlphaClasses=$actualAlphabetOutputSize")
            true
        } catch (e: Exception) {
            lastError = e.message ?: "Unknown exception"
            Log.e(TAG, "Failed to load TFLite model: ${e.message}", e)
            false
        }
    }

    /**
     * Loads a TFLite model from the assets folder using memory-mapped IO.
     * Memory-mapping is faster than reading into a byte array.
     */
    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelFileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // WORD MODEL INFERENCE (LSTM — processes frame sequences)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Runs the LSTM word model on a buffered sequence of landmark frames.
     *
     * @param frameBuffer  List of normalized landmark arrays.
     *                     Each array has HAND_LANDMARK_SIZE (126) floats.
     *                     The last SEQ_LENGTH frames are used.
     * @return InferenceResult with the top word prediction and confidence
     */
    fun runWordInference(frameBuffer: List<FloatArray>): InferenceResult {
        val interpreter = wordInterpreter
            ?: return InferenceResult("UNKNOWN", 0f, false)

        // ── Prepare fixed-length input ────────────────────────────────────────
        // Shape: [1, actualWordSeqLen, HAND_LANDMARK_SIZE]
        // actualWordSeqLen is read from the model at init time — safe regardless of SEQ_LENGTH setting
        val inputBuffer = ByteBuffer.allocateDirect(
            1 * actualWordSeqLen * HAND_LANDMARK_SIZE * Float.SIZE_BYTES
        ).order(ByteOrder.nativeOrder())

        val framesToUse = if (frameBuffer.size >= actualWordSeqLen) {
            frameBuffer.takeLast(actualWordSeqLen)
        } else {
            // Pad with zeros at the beginning (pre-padding)
            val padding = Array(actualWordSeqLen - frameBuffer.size) { FloatArray(HAND_LANDMARK_SIZE) }
            padding.asList() + frameBuffer
        }

        for (frame in framesToUse) {
            for (value in frame) {
                inputBuffer.putFloat(value)
            }
        }

        // ── Output buffer ─────────────────────────────────────────────────────
        // Shape: [1, num_classes] — size from model, not hardcoded
        val outputBuffer = Array(1) { FloatArray(actualWordOutputSize) }

        try {
            interpreter.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "Word model inference failed: ${e.message}")
            return InferenceResult("UNKNOWN", 0f, false)
        }

        // ── Softmax is assumed applied in the model; find argmax ─────────────
        val probabilities = outputBuffer[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[maxIndex]
        val label = WORD_LABELS.getOrElse(maxIndex) { "UNKNOWN" }

        Log.d(TAG, "Word prediction: $label (${(confidence * 100).toInt()}%)")

        return InferenceResult(
            label = label,
            confidence = confidence,
            isHighConfidence = confidence >= WORD_CONFIDENCE_THRESHOLD
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ALPHABET MODEL INFERENCE (DNN — processes single frames)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Runs the DNN alphabet model on a single frame's landmark array.
     *
     * @param landmarks  Normalized FloatArray of 126 values
     *                   (21 landmarks × 3 coords × 2 hands)
     * @return InferenceResult with the recognized letter
     */
    fun runAlphabetInference(landmarks: FloatArray): InferenceResult {
        val interpreter = alphabetInterpreter
            ?: return InferenceResult("?", 0f, false)

        if (landmarks.size != HAND_LANDMARK_SIZE) {
            Log.w(TAG, "Landmark array size mismatch: expected $HAND_LANDMARK_SIZE, got ${landmarks.size}")
            return InferenceResult("?", 0f, false)
        }

        // ── Prepare input: shape [1, 126] ─────────────────────────────────────
        val inputBuffer = ByteBuffer.allocateDirect(
            HAND_LANDMARK_SIZE * Float.SIZE_BYTES
        ).order(ByteOrder.nativeOrder())

        for (value in landmarks) {
            inputBuffer.putFloat(value)
        }

        // ── Output buffer: shape [1, actualAlphabetOutputSize] ───────────────
        val outputBuffer = Array(1) { FloatArray(actualAlphabetOutputSize) }

        try {
            interpreter.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            Log.e(TAG, "Alphabet model inference failed: ${e.message}")
            return InferenceResult("?", 0f, false)
        }

        val probabilities = outputBuffer[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[maxIndex]
        val letter = ALPHABET_LABELS.getOrElse(maxIndex) { "?" }

        Log.d(TAG, "Alphabet prediction: $letter (${(confidence * 100).toInt()}%)")

        return InferenceResult(
            label = letter,
            confidence = confidence,
            isHighConfidence = confidence >= 0.70f
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────

    fun close() {
        wordInterpreter?.close()
        alphabetInterpreter?.close()
        wordInterpreter = null
        alphabetInterpreter = null
        Log.i(TAG, "TFLite interpreters closed.")
    }
}
