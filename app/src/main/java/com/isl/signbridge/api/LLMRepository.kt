package com.isl.signbridge.api

import com.isl.signbridge.BuildConfig
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

// ─────────────────────────────────────────────────────────────────────────────
// Singleton Retrofit client
// ─────────────────────────────────────────────────────────────────────────────
object RetrofitClient {

    private const val BASE_URL = "https://api.groq.com/openai/v1/"

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BODY   // Remove in production
    }

    private val okHttpClient = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    val groqApiService: GroqApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(GroqApiService::class.java)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LLM Repository — wraps API call with business logic & error handling
// ─────────────────────────────────────────────────────────────────────────────
class LLMRepository {

    /**
     * Converts a raw ISL gloss string into a grammatically correct English sentence.
     *
     * System prompt is crafted to give the LLM minimal context, forcing
     * it to produce only the clean output sentence — no preamble.
     *
     * @param rawGloss  e.g. "I HOSPITAL GO"
     * @return          e.g. "I need to go to the hospital." or null on error
     */
    suspend fun translateGlossToSentence(rawGloss: String): Result<String> {
        return try {
            val apiKey = "Bearer ${BuildConfig.GROQ_API_KEY}"

            val request = GroqRequest(
                model = "llama-3.1-8b-instant",
                messages = listOf(
                    Message(
                        role = "system",
                        content = """You are an expert Indian Sign Language (ISL) interpreter.
Your job is to convert raw ISL gloss notation into a natural, grammatically correct English sentence.
ISL gloss uses topic-comment structure and drops articles/prepositions.
Output ONLY the final English sentence — no explanation, no preamble, no quotes.""".trimIndent()
                    ),
                    Message(
                        role = "user",
                        content = "Convert these raw sign language glosses into a grammatically correct, conversational English sentence. Output ONLY the sentence: $rawGloss"
                    )
                ),
                maxTokens = 256,
                temperature = 0.3
            )

            val response = RetrofitClient.groqApiService.getChatCompletion(apiKey, request)

            if (response.isSuccessful) {
                val body = response.body()
                val sentence = body?.choices?.firstOrNull()?.message?.content?.trim()
                if (!sentence.isNullOrBlank()) {
                    Result.success(sentence)
                } else {
                    Result.failure(Exception("Empty response from LLM"))
                }
            } else {
                Result.failure(
                    Exception("API Error ${response.code()}: ${response.errorBody()?.string()}")
                )
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
