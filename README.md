# SignBridge ISL — Real-Time Indian Sign Language Recognition

> **Android application** that recognises Indian Sign Language (ISL) gestures using MediaPipe hand tracking, TFLite machine-learning models, and translates recognised gloss words into natural English sentences using an LLM (Groq API).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File Structure](#3-file-structure)
4. [How It Works — End-to-End Pipeline](#4-how-it-works--end-to-end-pipeline)
5. [ML Models](#5-ml-models)
6. [Setup & Prerequisites](#6-setup--prerequisites)
7. [Building the APK](#7-building-the-apk)
8. [Running on Device](#8-running-on-device)
9. [Using the App](#9-using-the-app)
10. [Demo Mode (Presentation)](#10-demo-mode-presentation)
11. [Switching to Real Inference](#11-switching-to-real-inference)
12. [Key Configuration Constants](#12-key-configuration-constants)
13. [Troubleshooting](#13-troubleshooting)
14. [Retraining the Models](#14-retraining-the-models)
15. [Known Limitations](#15-known-limitations)

---

## 1. Project Overview

SignBridge is a mini-project for ISL communication assistance. The goal is to allow a deaf/mute person to sign in front of their phone camera, have the app recognise the signs, and output a natural English sentence in real time.

### Core features

| Feature | Details |
|---|---|
| **Hand tracking** | MediaPipe HandLandmarker — 21 landmarks × 2 hands × 3 axes = 126 features per frame |
| **Word recognition** | BiLSTM over a 40-frame sliding window → one of 264 ISL words |
| **Fingerspelling (Alpha)** | DNN on single frame → A–Z letter |
| **Gloss → English** | Groq LLM (`llama-3.1-8b-instant`) translates raw gloss into a fluent sentence |
| **Demo mode** | Scripted word/letter injection for presentations — hand overlay is still live |
| **Camera flip** | Toggle front ↔ back camera at runtime |
| **Mode toggle** | Switch between Word-level and Alphabet-level recognition |

---

## 2. Architecture

```
Camera (CameraX)
      │
      ▼  ImageProxy (RGBA_8888)
processImageProxy()          ← scale to 320px, rotate to portrait
      │
      ▼  Bitmap
HandLandmarkerHelper         ← MediaPipe Tasks-Vision 0.10.9
  (LIVE_STREAM mode)
      │
      ▼  HandLandmarkerResult (21 landmarks/hand)
SignLanguageViewModel
  ├── DEMO_MODE = true  →  startDemoWordInjection()
  │                           drip-feed scripted tokens
  │
  └── DEMO_MODE = false →  State Machine
        ├── IDLE
        ├── WORD_MODE  →  TFLiteInferenceHelper.runWordInference()
        │                   sliding window (40 frames, slide by 20)
        │                   → label + confidence
        │                   → append to glossWords[]
        ├── SPELL_MODE →  TFLiteInferenceHelper.runAlphabetInference()
        │                   single frame DNN
        │                   → letter appended to spellBuffer
        └── API_CALLING →  LLMRepository.translateGlossToSentence()
                            Groq REST API (Retrofit + OkHttp)
                            → final English sentence
```

### UI Layer
```
MainActivity
  ├── PreviewView          — live camera feed
  ├── CameraOverlayView    — hand skeleton + animated border
  ├── topCard
  │     ├── status badge   — LISTENING / WORD MODE / SPELL MODE / TRANSLATING
  │     └── tvFinalSentence — LLM output
  └── bottomCard
        ├── tvModeLabel    — "RAW GLOSS · WORD MODE" or "· ALPHA MODE"
        ├── tvRawGloss     — building gloss string ("TRAIN LEAVE WHEN")
        ├── btnToggleMode  — switch WORD ↔ ALPHA
        └── btnClear       — reset everything
  + btnFlipCamera (FAB)    — top-right, toggle front/back camera
```

---

## 3. File Structure

```
ISLApp/
├── app/
│   ├── src/main/
│   │   ├── assets/
│   │   │   ├── word_model.tflite        ← ISL word model (264 classes, seq=40)
│   │   │   ├── alphabet_model.tflite    ← Fingerspelling model (26 classes)
│   │   │   ├── hand_landmarker.task     ← MediaPipe hand landmark model
│   │   │   ├── convert_models.py        ← Colab script: Keras → TFLite export
│   │   │   └── README_MODELS.txt        ← Model placement guide
│   │   │
│   │   ├── java/com/isl/signbridge/
│   │   │   │
│   │   │   ├── MainActivity.kt
│   │   │   │     Camera setup, permission, UI observers,
│   │   │   │     camera flip logic, mode toggle wiring
│   │   │   │
│   │   │   ├── SignLanguageViewModel.kt
│   │   │   │     Central state machine, demo mode injection,
│   │   │   │     sliding-window logic, LLM trigger
│   │   │   │
│   │   │   ├── ml/
│   │   │   │   ├── HandLandmarkerHelper.kt
│   │   │   │   │     MediaPipe initialisation + LIVE_STREAM callback
│   │   │   │   └── TFLiteInferenceHelper.kt
│   │   │   │         TFLite Interpreter setup (FlexDelegate),
│   │   │   │         runWordInference(), runAlphabetInference(),
│   │   │   │         264-label WORD_LABELS map, A–Z ALPHABET_LABELS
│   │   │   │
│   │   │   ├── api/
│   │   │   │   ├── GroqApiService.kt    ← Retrofit interface
│   │   │   │   ├── GroqModels.kt        ← Request/Response data classes
│   │   │   │   └── LLMRepository.kt     ← API call + prompt construction
│   │   │   │
│   │   │   └── ui/
│   │   │       ├── CameraOverlayView.kt ← Canvas: hand skeleton + border anim
│   │   │       ├── BorderController.kt  ← Animates the glowing border per state
│   │   │       └── RecognitionState.kt  ← Enum: IDLE, WORD_MODE, SPELL_MODE,
│   │   │                                         API_CALLING
│   │   │
│   │   └── res/layout/activity_main.xml
│   │
│   └── build.gradle                     ← Dependencies, TFLite 2.16.1 + select-tf-ops
│
├── build.gradle                         ← Root build file
├── settings.gradle                      ← Repo declarations (google + mavenCentral)
├── gradle.properties                    ← Jetifier ignorelist, JVM args
├── local.properties                     ← SDK path + GROQ_API_KEY (git-ignored)
├── local.properties.example             ← Template for local.properties
└── push_to_github.ps1                   ← PowerShell script to push to GitHub
```

---

## 4. How It Works — End-to-End Pipeline

### Step 1 — Frame capture

CameraX `ImageAnalysis` delivers frames off the main thread. Before any processing:

```
Raw frame (e.g., 944×704)
  → scale to 320px wide   (reduces rotation cost by ~8×, improves fps)
  → rotate to portrait    (based on imageProxy.imageInfo.rotationDegrees)
  → pass to MediaPipe
```

### Step 2 — Hand landmark detection (MediaPipe)

`HandLandmarkerHelper` runs in `LIVE_STREAM` mode — asynchronous, never blocks the camera thread.

- Detects up to **2 hands**
- Each hand: **21 landmarks** × (x, y, z) = **63 floats**
- Both hands together: **126 floats per frame**
- If only one hand detected, the second 63 floats are zero-padded

### Step 3 — State machine decision (ViewModel)

`processLandmarkResult()` is called on every MediaPipe result:

```
Has hands?
  YES → onHandsDetected()
          cancel LLM countdown timer
          if DEMO_MODE: startDemoWordInjection() (skip inference)
          else state = WORD_MODE or SPELL_MODE
               → collect frames into frameBuffer
  NO  → onHandsAbsent()
          handsPresent = false
          state = IDLE
          if glossWords not empty:
            start 3-second countdown
            → triggerLLMCall()
```

### Step 4 — Word inference (Sliding Window)

In `WORD_MODE`, every new frame is appended to `frameBuffer`.

```
frameBuffer fills to 40 frames
  → runWordInference()
      flatten buffer → [1, 40, 126] input tensor
      Interpreter.run() with FlexDelegate (BiLSTM ops)
      argmax of [1, 264] output → label + confidence
      if confidence ≥ 0.20 → append word to glossWords
  → slide buffer: keep frames [20..39], discard [0..19]
  → next 20 frames fill again → repeat
```

### Step 5 — Fingerspelling (Spell Mode)

When confidence from word inference is low, the system may switch to `SPELL_MODE`. Each frame runs:

```
flatten single frame → [1, 126] input tensor
Interpreter.run() → [1, 26] softmax
argmax → letter A–Z
micro-pause detection (wrist velocity < 0.008) → capture letter
swipe gesture (velocity > 0.045, horizontal ratio > 2.5) → finalize word
```

### Step 6 — LLM Translation

After 3 seconds of hands-absent:

```
glossWords = ["TRAIN", "LEAVE", "WHEN"]
  → LLMRepository.translateGlossToSentence("TRAIN LEAVE WHEN")
      Groq API: POST /openai/v1/chat/completions
      Model: llama-3.1-8b-instant
      Prompt: "You are an ISL interpreter. Convert this ISL gloss into 
               natural English: TRAIN LEAVE WHEN"
      Response: "When will the train leave?"
  → _finalSentence.postValue("When will the train leave?")
```

---

## 5. ML Models

### word_model.tflite

| Property | Value |
|---|---|
| **Task** | ISL word/phrase classification |
| **Architecture** | Bidirectional LSTM (BiLSTM) |
| **Input shape** | `[1, 40, 126]` — batch=1, seq=40 frames, features=126 |
| **Output shape** | `[1, 264]` — softmax probabilities over 264 ISL classes |
| **Trained with** | TensorFlow 2.20 (Keras) |
| **Op requirements** | SELECT_TF_OPS (TensorListReserve, TensorListStack) → needs FlexDelegate |
| **File size** | ~228 KB |
| **Confidence threshold** | 0.20 (see `WORD_CONFIDENCE_THRESHOLD` in TFLiteInferenceHelper) |

### alphabet_model.tflite

| Property | Value |
|---|---|
| **Task** | ISL fingerspelling (A–Z) |
| **Architecture** | Dense DNN |
| **Input shape** | `[1, 126]` — single frame landmarks |
| **Output shape** | `[1, 26]` — A through Z |
| **File size** | ~107 KB |
| **Confidence threshold** | 0.70 |

### hand_landmarker.task

MediaPipe's pre-trained hand landmark model. **Already bundled** — no manual download needed.

---

## 6. Setup & Prerequisites

### Software required

| Tool | Version |
|---|---|
| Android Studio | Hedgehog (2023.1.1) or later |
| Android SDK | API 35 (compileSdk) |
| Min Android | API 26 (Android 8.0) |
| Java | 1.8 (configured in build.gradle) |
| Kotlin | 1.9.25 |
| Gradle | 9.0.0 |

### Get a Groq API key (free)

1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up and go to **API Keys**
3. Create a new key — it starts with `gsk_`

### Configure local.properties

> ⚠️ This file is **git-ignored** and must never be committed.

Create `local.properties` in the project root (copy from `local.properties.example`):

```properties
# Your Android SDK path — Android Studio sets this automatically
sdk.dir=C\:\\Users\\YOUR_USERNAME\\AppData\\Local\\Android\\Sdk

# Groq API key
GROQ_API_KEY=gsk_YOUR_KEY_HERE
```

---

## 7. Building the APK

### Option A — Android Studio (recommended)

1. Open Android Studio
2. `File → Open` → select the `ISLApp` folder
3. Wait for Gradle sync to complete
4. Connect your Android device via USB (enable USB debugging)
5. Click **▶ Run** (Shift+F10)

### Option B — Command line

```powershell
# From the ISLApp directory
./gradlew assembleDebug

# APK will be at:
# app/build/outputs/apk/debug/app-debug.apk
```

### First-time Gradle sync notes

- Gradle downloads `tensorflow-lite:2.16.1` (~15 MB) and `tensorflow-lite-select-tf-ops:2.16.1` (~200 MB) from Maven Central
- This is a **one-time download** — subsequent builds are fast
- Ensure internet connection during first sync

---

## 8. Running on Device

### Enable USB debugging on Android

1. Settings → About Phone → tap **Build Number** 7 times
2. Settings → Developer Options → enable **USB Debugging**
3. Connect via USB → trust the computer on your phone

### App permissions required

- `CAMERA` — requested at runtime on first launch
- `INTERNET` — for Groq API calls (granted automatically)

---

## 9. Using the App

### Main screen layout

```
┌─────────────────────────────────┐
│  ● LISTENING           [■ ⟳]    │  ← Status badge + camera flip (■)
│  ─────────────────────────────  │
│  SignBridge ISL                 │
│  Translation will appear here.. │  ← LLM output
├─────────────────────────────────┤
│                                 │
│      [live camera feed]         │
│      [hand skeleton overlay]    │
│                                 │
├─────────────────────────────────┤
│  RAW GLOSS · WORD MODE          │
│  TRAIN LEAVE WHEN               │  ← Building gloss
│                          [⟳]   │
│  0 words    [WORD] [Clear]      │  ← Mode toggle + clear
└─────────────────────────────────┘
```

### Controls

| Button | Action |
|---|---|
| **Camera icon (top-right)** | Flip between front and back camera |
| **WORD / ALPHA button** | Toggle word recognition ↔ fingerspelling mode |
| **Clear** | Reset gloss, cancel LLM countdown, start fresh |

### Workflow

1. **Raise your hands** → skeleton overlay appears → state changes to WORD MODE (green border)
2. **Sign an ISL word/phrase** → words appear in the gloss bar one by one
3. **Lower your hands** → 3-second countdown starts
4. **Wait 3 seconds** → LLM translates gloss → sentence appears in the top card
5. Tap **Clear** to sign a new sentence

### Signing tips for best accuracy

- Sign at **1–1.5 metres** from the camera (matches training distance)
- Keep both hands **fully visible** in frame
- **Good lighting** — avoid backlight (don't stand in front of a window)
- Sign each word **clearly and distinctly**, with a brief pause between words
- The dataset uses a plain background — complex backgrounds reduce accuracy

---

## 10. Demo Mode (Presentation)

Demo mode is enabled by default (`DEMO_MODE = true` in `SignLanguageViewModel.kt`).

### What is real vs scripted

| Component | Status |
|---|---|
| Hand skeleton overlay | ✅ **Real** — MediaPipe tracks your actual hands |
| Hand detection trigger | ✅ **Real** — inference starts when real hands appear |
| Words appearing | 🎭 **Scripted** — injected from DEMO_SENTENCES on a timer |
| LLM translation | ✅ **Real** — actual Groq API call with the scripted gloss |
| Camera flip | ✅ **Real** |

### Word mode demo sentences (cycles automatically)

| Round | Gloss shown | Expected translation |
|---|---|---|
| 1 | `TRAIN LEAVE WHEN` | "When will the train leave?" |
| 2 | `I HUNGRY` | "I am hungry" |
| 3 | `HELP ME` | "Please help me" |
| 4 | `GOOD MORNING HOW ARE YOU` | "Good morning! How are you?" |

Words appear every **2 seconds** while hands are raised.

### Alpha mode demo sentences

Tap **WORD → ALPHA** button first, then raise hands:

| Round | Letters shown | Expected word |
|---|---|---|
| 1 | `H E L L O` | "Hello" |
| 2 | `T H A N K Y O U` | "Thank you" |
| 3 | `H E L P` | "Help" |
| 4 | `I S L` | "ISL" |

Letters appear every **1.2 seconds**.

### Demo workflow for a smooth presentation

```
1. Open app → raise hands → wait for words to appear (2s each)
2. Lower hands → wait 3 seconds → translation appears
3. Tap [Clear]
4. Tap [WORD] button → switches to ALPHA mode
5. Raise hands → letters appear → lower hands → word translation appears
6. Tap camera flip button → show it works with back camera too
```

---

## 11. Switching to Real Inference

When you want to test actual model inference (not demo):

**In `SignLanguageViewModel.kt`, line ~58:**

```kotlin
// Change this:
const val DEMO_MODE = true

// To this:
const val DEMO_MODE = false
```

Rebuild and install. The app will now:
- Use TFLite inference on real hand landmark data
- Run the BiLSTM word model on every 40-frame window
- Fall back to fingerspelling DNN if word confidence is low

> **Note:** If you see "TFLite load failed" in the top card, check the Logcat for the exact error. Common causes are listed in the Troubleshooting section.

---

## 12. Key Configuration Constants

All tunable parameters are in `TFLiteInferenceHelper.kt` and `SignLanguageViewModel.kt`.

### TFLiteInferenceHelper.kt

```kotlin
// Model filenames (must match files in assets/)
private const val WORD_MODEL_FILE     = "word_model.tflite"
private const val ALPHABET_MODEL_FILE = "alphabet_model.tflite"

// Sequence length — must match model's input shape[1]
// (auto-read from model tensor shape at runtime, this is the fallback)
var wordSeqLen: Int = 40

// Confidence thresholds
private const val WORD_CONFIDENCE_THRESHOLD     = 0.20f  // lower = more permissive
private const val ALPHABET_CONFIDENCE_THRESHOLD = 0.70f

// Min frames needed before attempting inference
const val MIN_FRAMES_FOR_WORD = 10
```

### SignLanguageViewModel.kt

```kotlin
// DEMO vs real inference
const val DEMO_MODE = true   // ← set false for real model

// LLM trigger: how long hands must be absent before calling LLM
private const val HANDS_ABSENT_TIMEOUT_MS = 3000L  // 3 seconds

// Sliding window slide amount (frames to keep after inference)
// Currently: slide by half (keep last 20 of 40)

// Demo timing
private const val DEMO_WORD_INTERVAL_MS  = 2000L   // 2s per word
private const val DEMO_ALPHA_INTERVAL_MS = 1200L   // 1.2s per letter
```

### LLMRepository.kt

```kotlin
// Groq model being used
private const val MODEL = "llama-3.1-8b-instant"
// Base URL
private const val BASE_URL = "https://api.groq.com/"
```

---

## 13. Troubleshooting

### "TFLite load failed: Internal error: Cannot create interpreter: Didn't find op for builtin opcode 'FULLY_CONNECTED' version '12'"

**Cause:** The model was trained with TensorFlow 2.16+ which generates FULLY_CONNECTED op v12. Older TFLite runtimes only support v11.

**Fix (already applied):** `tensorflow-lite:2.16.1` in `build.gradle` — supports FC v12. If you see this error, clean the Gradle cache:
```powershell
./gradlew clean
./gradlew assembleDebug
```

---

### "TFLite load failed: Internal error: Cannot create interpreter: Didn't find op ... (TensorListReserve)"

**Cause:** The BiLSTM model needs TensorFlow Select ops (Flex delegate) which weren't registered.

**Fix (already applied):** `FlexDelegate()` is added to `Interpreter.Options` in `TFLiteInferenceHelper.kt`. Ensure `tensorflow-lite-select-tf-ops:2.16.1` is in dependencies.

---

### "LLM call failed: model_decommissioned"

**Cause:** Groq deprecated the model name.

**Fix:** Update the model name in `LLMRepository.kt`:
```kotlin
private const val MODEL = "llama-3.1-8b-instant"  // current working model
```

---

### App opens but hand overlay is off-centre / wrong size

**Cause:** The `isFrontCamera` flag passed to MediaPipe affects landmark mirroring.

**Fix:** Ensure `isFrontCamera = true` when using front camera (default). After tapping the flip button, the flag toggles correctly.

---

### Words never appear (only "..." in gloss bar)

In real-inference mode, check:
1. Enough light? Hands clearly visible?
2. `DEMO_MODE = false`? (check the constant)
3. Is the frame buffer filling? Look for `Word model input` log in Logcat
4. Lower the threshold: `WORD_CONFIDENCE_THRESHOLD = 0.10f`
5. Sign at 1–1.5m distance (dataset was recorded at this range)

---

### Build error: "Could not find org.tensorflow:tensorflow-lite-select-tf-ops:X.X.X"

Only these versions exist on Maven Central for `select-tf-ops`:
`2.9.0, 2.10.0, 2.11.0, 2.12.0, 2.13.0, 2.14.0, 2.15.0, 2.16.1`

> ⚠️ 2.16.0, 2.17.0 and 2.17+ were **never published** for select-tf-ops. Use `2.16.1`.

---

### Gradle sync fails first time (download timeout)

`tensorflow-lite-select-tf-ops` is ~200 MB. On a slow connection:
```properties
# In gradle.properties, increase timeout:
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
```
Or run with `--no-daemon` for better error output:
```powershell
./gradlew assembleDebug --no-daemon
```

---

## 14. Retraining the Models

### Training environment

Models were trained on **Google Colab** with:
- TensorFlow 2.20.0
- MediaPipe Holistic / HandLandmarker for feature extraction
- Custom ISL dataset (videos of 264 ISL words)

### Export script

`app/src/main/assets/convert_models.py` converts a trained Keras `.keras` file to TFLite:

```python
# The script handles:
# 1. Loading the .keras model
# 2. Creating a TFLiteConverter
# 3. Enabling SELECT_TF_OPS (required for LSTM layers)
# 4. Saving the .tflite file
```

Run on Colab after training:
```bash
python convert_models.py --input isl_word_model.keras --output word_model.tflite
```

### After retraining

1. Copy new `.tflite` to `app/src/main/assets/`
2. Update label count/list in `TFLiteInferenceHelper.kt` → `WORD_LABELS`
3. Update `SEQ_LENGTH` if the sequence length changed
4. Rebuild the APK

---

## 15. Known Limitations

| Limitation | Details |
|---|---|
| **Distance sensitivity** | Model trained at ~1–1.5m distance. Accuracy drops if you sign too close or too far |
| **Lighting dependency** | Poor lighting degrades landmark detection quality |
| **Signer-specific** | Model trained on specific signers — other signing styles may have lower accuracy |
| **No continuous recognition** | Each "utterance" is bounded by hand raise/lower events |
| **LLM latency** | Groq API adds ~500ms–2s for translation after hands drop |
| **264 word vocabulary** | Only words from the training dataset are recognisable |
| **Single model** | No user fine-tuning — accuracy improvement requires retraining |
| **DEMO_MODE default = true** | Remember to set `false` before real testing |

---

## Credits

| Component | Source |
|---|---|
| Hand landmark detection | [MediaPipe Tasks-Vision 0.10.9](https://developers.google.com/mediapipe) |
| TFLite runtime | [TensorFlow Lite 2.16.1](https://www.tensorflow.org/lite) |
| LLM translation | [Groq Cloud — llama-3.1-8b-instant](https://console.groq.com) |
| CameraX | [AndroidX CameraX 1.3.1](https://developer.android.com/training/camerax) |
| HTTP client | [Retrofit 2.9 + OkHttp 4.12](https://square.github.io/retrofit/) |

---

*SignBridge ISL — 6th Semester Mini Project*
