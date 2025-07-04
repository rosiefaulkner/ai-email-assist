from typing import List, Union

import google.generativeai as genai
import numpy as np

from ..config import get_settings


class EmbeddingUtil:
    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel("embedding-001")

    async def get_embedding(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text using Google's Embedding API.

        Args:
            text: Single text string or list of text strings

        Returns:
            List of embeddings or list of lists for batch processing
        """
        try:
            # Handle single text input
            if isinstance(text, str):
                embedding = self.model.embed_content(
                    model=self.settings.EMBEDDING_MODEL,
                    content=text,
                )
                return embedding.values.tolist()

            # Handle batch processing
            embeddings = []
            for t in text:
                embedding = self.model.embed_content(
                    model=self.settings.EMBEDDING_MODEL,
                    content=t,
                )
                embeddings.append(embedding.values.tolist())
            return embeddings

        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 768  # Standard embedding dimension

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )

            # Ensure result is between 0 and 1
            return float(max(0, min(1, similarity)))

        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0

    async def batch_get_embeddings(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches.

        Args:
            texts: List of text strings
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = await self.get_embedding(batch)
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            print(f"Error in batch embedding generation: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]
