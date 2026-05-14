package com.isl.signbridge.api

import com.google.gson.annotations.SerializedName

// ─────────────────────────────────────────────────────────────────────────────
// Data classes for Groq Chat Completion API
// Docs: https://console.groq.com/docs/openai
// ─────────────────────────────────────────────────────────────────────────────

data class GroqRequest(
    val model: String = "llama-3.1-8b-instant",  // Successor to decommissioned llama3-8b-8192
    val messages: List<Message>,
    @SerializedName("max_tokens") val maxTokens: Int = 256,
    val temperature: Double = 0.3            // Low temp = deterministic translation
)

data class Message(
    val role: String,   // "system" or "user"
    val content: String
)

data class GroqResponse(
    val id: String,
    val choices: List<Choice>,
    val usage: Usage?
)

data class Choice(
    val index: Int,
    val message: Message,
    @SerializedName("finish_reason") val finishReason: String
)

data class Usage(
    @SerializedName("prompt_tokens") val promptTokens: Int,
    @SerializedName("completion_tokens") val completionTokens: Int,
    @SerializedName("total_tokens") val totalTokens: Int
)
