from typing import Any, Dict, List

import google.generativeai as genai

from ..config import get_settings


class GeminiAgent:
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("gemini-pro")
        self.settings = settings

    async def generate_response(
        self, query: str, context: List[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate a response using Google Gemini with optional context."""
        try:
            # Prepare the prompt with context if available
            prompt = self._prepare_prompt(query, context)

            # Generate response
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": kwargs.get("temperature", self.settings.TEMPERATURE),
                    "top_p": kwargs.get("top_p", self.settings.TOP_P),
                    "top_k": kwargs.get("top_k", self.settings.TOP_K),
                    "max_output_tokens": kwargs.get(
                        "max_tokens", self.settings.MAX_TOKENS
                    ),
                },
            )

            return {
                "answer": response.text,
                "metadata": {
                    "model": "gemini-pro",
                    "finish_reason": getattr(response, "finish_reason", None),
                    "prompt_tokens": getattr(response, "prompt_token_count", None),
                    "completion_tokens": getattr(
                        response, "completion_token_count", None
                    ),
                },
            }

        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "metadata": {"error_type": type(e).__name__},
            }

    def _prepare_prompt(self, query: str, context: List[str] = None) -> str:
        """Prepare the prompt with context for the Gemini model."""
        if not context:
            return query

        context_str = "\n".join(context)
        prompt = f"""Context information:
{context_str}

Based on the above context, please answer the following question:
{query}
"""
        return prompt

    async def analyze_relevance(self, query: str, document: str) -> float:
        """Analyze the relevance of a document to the query."""
        try:
            prompt = f"""On a scale of 0 to 1, how relevant is this document to the query?
Query: {query}
Document: {document}

Provide only the numerical score without any explanation."""

            response = await self.model.generate_content_async(prompt)
            score = float(response.text.strip())
            return min(max(score, 0), 1)  # Ensure score is between 0 and 1

        except Exception:
            return 0.0  # Return 0 relevance on error
