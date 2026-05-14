package com.isl.signbridge.ui

import android.graphics.Color
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView

// ─────────────────────────────────────────────────────────────────────────────
// BorderController
//
// Manages the dynamic visual state of the camera border and status indicators.
// Decouples UI state display from the ViewModel — only knows about Views.
//
// State → Color mapping:
//   IDLE        → Muted grey  (#555577)  — no hands detected
//   WORD_MODE   → Green       (#4CAF50)  — hands present, buffering
//   SPELL_MODE  → Orange      (#FF9800)  — fingerspelling fallback active
//   API_CALLING → Blue pulse  (#2196F3)  — LLM API call in progress
// ─────────────────────────────────────────────────────────────────────────────
class BorderController(
    private val overlayView: CameraOverlayView,
    private val tvStateBadge: TextView,
    private val statusDot: View,
    private val apiProgressBar: ProgressBar
) {
    // ── Color constants ──────────────────────────────────────────────────────
    private val colorIdle = Color.parseColor("#555577")
    private val colorWordMode = Color.parseColor("#4CAF50")
    private val colorSpellMode = Color.parseColor("#FF9800")
    private val colorApiCall = Color.parseColor("#2196F3")

    // ── Current state tracking ───────────────────────────────────────────────
    private var currentState: RecognitionState = RecognitionState.IDLE

    // ─────────────────────────────────────────────────────────────────────────
    // State transition methods — call from ViewModel observers
    // ─────────────────────────────────────────────────────────────────────────

    /** No hands detected */
    fun setIdle() {
        if (currentState == RecognitionState.IDLE) return
        currentState = RecognitionState.IDLE
        overlayView.stopPulse()
        overlayView.setBorderColor(colorIdle)
        statusDot.setBackgroundResource(android.R.color.transparent)
        tvStateBadge.text = "● IDLE"
        tvStateBadge.setTextColor(colorIdle)
        apiProgressBar.visibility = View.GONE
    }

    /** Hands present — buffering frames for LSTM inference */
    fun setWordMode() {
        if (currentState == RecognitionState.WORD_MODE) return
        currentState = RecognitionState.WORD_MODE
        overlayView.stopPulse()
        overlayView.setBorderColor(colorWordMode)
        tvStateBadge.text = "● LISTENING"
        tvStateBadge.setTextColor(colorWordMode)
        apiProgressBar.visibility = View.GONE
    }

    /** Low-confidence word → fell back to finger spelling */
    fun setSpellMode() {
        if (currentState == RecognitionState.SPELL_MODE) return
        currentState = RecognitionState.SPELL_MODE
        overlayView.stopPulse()
        overlayView.setBorderColor(colorSpellMode)
        tvStateBadge.text = "● SPELL MODE"
        tvStateBadge.setTextColor(colorSpellMode)
        apiProgressBar.visibility = View.GONE
    }

    /** LLM API call in progress — pulse the border blue */
    fun setApiCalling() {
        if (currentState == RecognitionState.API_CALLING) return
        currentState = RecognitionState.API_CALLING
        overlayView.setBorderColor(colorApiCall)
        overlayView.startPulseAnimation()
        tvStateBadge.text = "◎ PROCESSING"
        tvStateBadge.setTextColor(colorApiCall)
        apiProgressBar.visibility = View.VISIBLE
    }

    /** Reset after API call completes */
    fun setApiComplete() {
        apiProgressBar.visibility = View.GONE
        overlayView.stopPulse()
        setIdle()
    }

    fun getCurrentState(): RecognitionState = currentState
}

// ─────────────────────────────────────────────────────────────────────────────
// Recognition State Enum
// The core state machine has 4 states
// ─────────────────────────────────────────────────────────────────────────────
enum class RecognitionState {
    /** No hands in frame */
    IDLE,
    /** Hands present — LSTM buffering */
    WORD_MODE,
    /** Fingerspelling fallback — DNN frame-by-frame */
    SPELL_MODE,
    /** LLM API call is in flight */
    API_CALLING
}
