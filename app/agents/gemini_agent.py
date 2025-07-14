from typing import Any, Dict, List

import google.generativeai as genai

from app.config import get_settings

"""
GeminiAgent class for interacting with Google Gemini. This class provides methods to generate responses,
analyze relevance, and prepare prompts for the Gemini model.
Attributes:
    model: The Gemini model instance.
    settings: The application settings.
Methods:
    generate_response(query: str, context: List[str] = None, **kwargs) -> Dict[str, Any]:
        Generate a response using Google Gemini with email context.
    _prepare_prompt(query: str, context: List[str] = None) -> str:
        Prepare the prompt with context for the Gemini model.
    analyze_relevance(query: str, document: str) -> float:
        Analyze the relevance of a document to a query using Google Gemini.
Args:
    query (str)
    context (List[str], optional)
    **kwargs
"""


class GeminiAgent:
    def __init__(self):
        settings = get_settings()
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self.settings = settings

    async def generate_response(
        self, query: str, context: List[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using Google Gemini with email context.
        Args:
            query (str): The user query.
            context (List[str]): List of context strings.
            **kwargs: Additional keyword arguments for generation configuration.
        Returns:
            Dict[str, Any]: The response from the Gemini model.
        """
        try:
            if not query.strip():
                return {
                    "answer": None,
                    "error": "Query cannot be empty",
                    "metadata": {"error_type": "ValueError"},
                }

            # Prepare the prompt with email message as context
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

            if not response or not response.text:
                return {
                    "answer": None,
                    "error": "No response generated from the model",
                    "metadata": {"error_type": "ModelError"},
                }

            return {
                "answer": response.text.strip(),
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
            print(f"Error in generate_response: {str(e)}")
            return {
                "error": str(e),
                "answer": None,
                "metadata": {"error_type": type(e).__name__},
            }

    def _prepare_prompt(self, query: str, context: List[str] = None) -> str:
        """
        Prepare the prompt with context for the Gemini model.
        Args:
            query (str): The user query.
            context (List[str]): List of context strings.
        Returns:
            str: The prepared prompt.
        """
        if not context:
            return query

        context_str = "\n".join(context)
        prompt = f"""Context information:
        {context_str} Based on the above context,
        please answer the following question:
        {query}
        """
        return prompt

    async def analyze_relevance(self, query: str, document: str) -> float:
        """
        Analyze the relevance of a document to a query using Google Gemini.
        Returns a score between 0 and 1, where 1 indicates perfect relevance.
        Args:
            query (str): The query to analyze.
            document (str): The document to analyze.
        Returns:
            float: The relevance score between 0 and 1.
        """
        try:
            prompt = f"""On a scale of 0 to 1, how relevant is this document to the query?
            Query: {query}
            Document: {document}

            Provide only the numerical score without any explanation.
            """

            response = await self.model.generate_content_async(prompt)
            score = float(response.text.strip())
            return min(max(score, 0), 1)  # Ensure score is between 0 and 1

        except Exception:
            return 0.0

import asyncio

async def main():
    query="Analyze this email and determine if it is spam. Consider the sender, subject, content, and any suspicious patterns or red flags. Provide a clear explanation of why it is or isn't spam."
    context=[
        "email_content: Dear Sir/Madam, I am a prince from Nigeria and I need your assistance...",
        "email_metadata: {\"from\": \"        \"email_metadata: {\"from\": \"EMAIL\", \"subject\": \"Urgent Business Proposal\", \"date\": \"2024-01-13T00:00:00Z\"}"
    ]
    client = GeminiAgent()
    messages = await client.generate_response(query=query, context=context)
    print(messages)

asyncio.run(main())