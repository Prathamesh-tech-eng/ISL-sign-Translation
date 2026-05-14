package com.isl.signbridge.api

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.Header
import retrofit2.http.POST

// ─────────────────────────────────────────────────────────────────────────────
// Retrofit Interface for the Groq LLM API
// Base URL: https://api.groq.com/openai/v1/
// ─────────────────────────────────────────────────────────────────────────────
interface GroqApiService {

    /**
     * Sends a chat completion request to the Groq API.
     *
     * @param authorization  Bearer token from BuildConfig.GROQ_API_KEY
     * @param request        The request body with model, messages, and params
     * @return               A Response wrapping the GroqResponse
     */
    @POST("chat/completions")
    suspend fun getChatCompletion(
        @Header("Authorization") authorization: String,
        @Body request: GroqRequest
    ): Response<GroqResponse>
}
