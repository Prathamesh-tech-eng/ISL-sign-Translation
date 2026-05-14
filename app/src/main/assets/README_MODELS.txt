# ─────────────────────────────────────────────────────────────────────────────
# assets/ — TFLITE MODEL PLACEMENT GUIDE
# ─────────────────────────────────────────────────────────────────────────────
#
# Place your trained .tflite model files here:
#
#   app/src/main/assets/word_model.tflite
#   app/src/main/assets/alphabet_model.tflite
#
# ─────────────────────────────────────────────────────────────────────────────
# HOW TO CREATE THE ASSETS FOLDER IN ANDROID STUDIO:
# ─────────────────────────────────────────────────────────────────────────────
#
#   1. In the Project pane, switch to "Android" view
#   2. Right-click on: app/src/main
#   3. Select: New > Folder > Assets Folder
#   4. Click Finish
#   5. Drag your .tflite files into app/src/main/assets/
#
# ─────────────────────────────────────────────────────────────────────────────
# MODEL SPECIFICATIONS (must match your trained models):
# ─────────────────────────────────────────────────────────────────────────────
#
#   word_model.tflite:
#     Input shape:  [1, 30, 126]   (batch=1, seq=30, landmarks=126)
#     Output shape: [1, N_WORDS]   (softmax probabilities)
#     Architecture: LSTM-based
#
#   alphabet_model.tflite:
#     Input shape:  [1, 126]       (batch=1, landmarks=126)
#     Output shape: [1, 26]        (softmax probabilities, A-Z)
#     Architecture: Dense/DNN
#
#   Update SEQ_LENGTH and WORD_LABELS/ALPHABET_LABELS in:
#     app/src/main/java/com/isl/signbridge/ml/TFLiteInferenceHelper.kt
#
# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE MODEL (auto-bundled, no action needed):
# ─────────────────────────────────────────────────────────────────────────────
#
#   hand_landmarker.task is automatically downloaded by the MediaPipe
#   dependency when you sync Gradle. You do NOT need to manually add it.
#
# ─────────────────────────────────────────────────────────────────────────────

PLACE_YOUR_TFLITE_MODELS_HERE=true
