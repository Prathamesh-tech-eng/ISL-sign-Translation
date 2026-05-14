# Add project specific ProGuard rules here.

# ── TensorFlow Lite ──────────────────────────────────────────────────────────
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.support.** { *; }

# ── MediaPipe ────────────────────────────────────────────────────────────────
-keep class com.google.mediapipe.** { *; }

# ── Retrofit + OkHttp ────────────────────────────────────────────────────────
-keepattributes Signature
-keepattributes Exceptions
-keep class retrofit2.** { *; }
-keep class okhttp3.** { *; }
-keep class okio.** { *; }

# ── Gson (data models) ───────────────────────────────────────────────────────
-keepattributes *Annotation*
-keep class com.isl.signbridge.api.** { *; }

# ── Kotlin Coroutines ────────────────────────────────────────────────────────
-keepclassmembernames class kotlinx.** { volatile <fields>; }
