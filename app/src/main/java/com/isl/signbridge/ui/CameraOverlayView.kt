package com.isl.signbridge.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.max
import kotlin.math.min

// ─────────────────────────────────────────────────────────────────────────────
// CameraOverlayView
//
// A transparent custom View drawn on top of the camera PreviewView.
// Responsibilities:
//   1. Draw the animated border with color based on current system state
//   2. Draw MediaPipe hand landmarks (dots + connecting lines)
// ─────────────────────────────────────────────────────────────────────────────
class CameraOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // ── Current landmarks to draw ────────────────────────────────────────────
    private var handLandmarkerResult: HandLandmarkerResult? = null
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // ── Border state ─────────────────────────────────────────────────────────
    private var borderColor: Int = Color.parseColor("#555577")
    private var borderAlpha: Float = 1.0f
    private var borderAnimDir: Float = 1.0f   // For pulsing

    // ── Paint objects (pre-allocated to avoid GC pressure in onDraw) ─────────
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        strokeCap = Paint.Cap.ROUND
    }

    private val landmarkPointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#00E5FF")
        strokeWidth = 4f
    }

    private val landmarkLinePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = Color.parseColor("#0088AA")
        strokeWidth = 3f
        strokeCap = Paint.Cap.ROUND
    }

    // ── MediaPipe hand connection indices (21 landmarks, standard topology) ──
    private val HAND_CONNECTIONS = listOf(
        // Palm
        0 to 1, 1 to 2, 2 to 3, 3 to 4,
        // Index
        0 to 5, 5 to 6, 6 to 7, 7 to 8,
        // Middle
        0 to 9, 9 to 10, 10 to 11, 11 to 12,
        // Ring
        0 to 13, 13 to 14, 14 to 15, 15 to 16,
        // Pinky
        0 to 17, 17 to 18, 18 to 19, 19 to 20,
        // Knuckle bar
        5 to 9, 9 to 13, 13 to 17
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Public update methods
    // ─────────────────────────────────────────────────────────────────────────

    /** Updates the landmark data to display. Called from ViewModel observer. */
    fun updateLandmarks(result: HandLandmarkerResult?, imgWidth: Int, imgHeight: Int) {
        handLandmarkerResult = result
        imageWidth = imgWidth
        imageHeight = imgHeight
        invalidate()  // Request redraw
    }

    /** Sets the border color based on the current system state. */
    fun setBorderColor(color: Int) {
        borderColor = color
        borderPaint.color = color
        cornerPaint.color = color
        invalidate()
    }

    /** Trigger a pulse animation (for LLM API call state). */
    fun startPulseAnimation() {
        // Simple alpha pulsing — runs on the UI thread via postDelayed
        pulseStep()
    }

    private fun pulseStep() {
        if (borderColor != Color.parseColor("#2196F3")) return  // Only pulse in blue state
        borderAlpha = (borderAlpha + 0.05f * borderAnimDir).coerceIn(0.3f, 1.0f)
        if (borderAlpha >= 1.0f || borderAlpha <= 0.3f) borderAnimDir *= -1
        borderPaint.alpha = (borderAlpha * 255).toInt()
        cornerPaint.alpha = (borderAlpha * 255).toInt()
        invalidate()
        postDelayed({ pulseStep() }, 50)  // ~20fps pulse
    }

    fun stopPulse() {
        borderAlpha = 1.0f
        borderPaint.alpha = 255
        cornerPaint.alpha = 255
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Drawing
    // ─────────────────────────────────────────────────────────────────────────

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawAnimatedBorder(canvas)
        drawLandmarks(canvas)
    }

    private fun drawAnimatedBorder(canvas: Canvas) {
        val w = width.toFloat()
        val h = height.toFloat()
        val margin = 24f
        val cornerRadius = 28f
        val cornerLength = 60f

        borderPaint.color = borderColor
        cornerPaint.color = borderColor

        // ── Faint full border ─────────────────────────────────────────────────
        borderPaint.alpha = 80
        val borderRect = RectF(margin, margin, w - margin, h - margin)
        canvas.drawRoundRect(borderRect, cornerRadius, cornerRadius, borderPaint)

        // ── Bright corner accents ─────────────────────────────────────────────
        cornerPaint.alpha = (borderAlpha * 255).toInt()

        // Top-left
        canvas.drawLine(margin, margin + cornerLength, margin, margin + cornerRadius, cornerPaint)
        canvas.drawLine(margin + cornerRadius, margin, margin + cornerLength, margin, cornerPaint)

        // Top-right
        canvas.drawLine(w - margin, margin + cornerRadius, w - margin, margin + cornerLength, cornerPaint)
        canvas.drawLine(w - margin - cornerLength, margin, w - margin - cornerRadius, margin, cornerPaint)

        // Bottom-left
        canvas.drawLine(margin, h - margin - cornerLength, margin, h - margin - cornerRadius, cornerPaint)
        canvas.drawLine(margin + cornerRadius, h - margin, margin + cornerLength, h - margin, cornerPaint)

        // Bottom-right
        canvas.drawLine(w - margin, h - margin - cornerLength, w - margin, h - margin - cornerRadius, cornerPaint)
        canvas.drawLine(w - margin - cornerLength, h - margin, w - margin - cornerRadius, h - margin, cornerPaint)
    }

    private fun drawLandmarks(canvas: Canvas) {
        val result = handLandmarkerResult ?: return
        if (imageWidth <= 1 || imageHeight <= 1) return

        // ── Replicate PreviewView "fillCenter" transform ──────────────────────
        // fillCenter scales by the LARGER ratio so one axis fills exactly
        // and the other axis overflows (is cropped). Content is centered.
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        val scale = maxOf(viewW / imageWidth.toFloat(), viewH / imageHeight.toFloat())
        val scaledW = imageWidth * scale
        val scaledH = imageHeight * scale
        // Negative offset means content extends beyond the view edge (cropped)
        val offsetX = (viewW - scaledW) / 2f
        val offsetY = (viewH - scaledH) / 2f

        result.landmarks().forEach { handLandmarks ->
            // Draw connecting lines
            for ((start, end) in HAND_CONNECTIONS) {
                if (start < handLandmarks.size && end < handLandmarks.size) {
                    val sLm = handLandmarks[start]
                    val eLm = handLandmarks[end]
                    canvas.drawLine(
                        toScreenX(sLm.x(), scaledW, offsetX),
                        toScreenY(sLm.y(), scaledH, offsetY),
                        toScreenX(eLm.x(), scaledW, offsetX),
                        toScreenY(eLm.y(), scaledH, offsetY),
                        landmarkLinePaint
                    )
                }
            }

            // Draw landmark dots
            handLandmarks.forEachIndexed { idx, lm ->
                val dotRadius = if (idx == 0) 8f else 5f
                canvas.drawCircle(
                    toScreenX(lm.x(), scaledW, offsetX),
                    toScreenY(lm.y(), scaledH, offsetY),
                    dotRadius, landmarkPointPaint
                )
            }
        }
    }

    /**
     * Mirror X for front camera (front camera sees a horizontally flipped world),
     * then apply fillCenter scale + offset.
     */
    private fun toScreenX(normalizedX: Float, scaledW: Float, offsetX: Float): Float =
        (1f - normalizedX) * scaledW + offsetX

    private fun toScreenY(normalizedY: Float, scaledH: Float, offsetY: Float): Float =
        normalizedY * scaledH + offsetY
}
