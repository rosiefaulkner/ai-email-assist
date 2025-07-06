from typing import List, Union

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.decomposition import PCA

from ..config import get_settings


class EmbeddingUtil:
    def __init__(self):
        self.settings = get_settings()
        self.model = GoogleGenerativeAIEmbeddings(
            google_api_key=self.settings.GOOGLE_API_KEY,
            model="models/embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
        )
        self.pca = PCA(n_components=768)
        self.pca_fitted = False

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
                embedding = await self.model.aembed_query(text)
                # Reshape for PCA
                embedding_array = np.array(embedding).reshape(1, -1)
                if not self.pca_fitted:
                    self.pca.fit(embedding_array)
                    self.pca_fitted = True
                reduced_embedding = self.pca.transform(embedding_array)[0].tolist()
                return reduced_embedding

            # Handle batch processing
            embeddings = await self.model.aembed_documents(text)
            embeddings_array = np.array(embeddings)
            if not self.pca_fitted:
                self.pca.fit(embeddings_array)
                self.pca_fitted = True
            reduced_embeddings = self.pca.transform(embeddings_array).tolist()
            return reduced_embeddings

        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * 768  # Gemini embedding-001 dimension

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
        try:
            # Process all texts at once since LangChain handles batching internally
            embeddings = await self.model.aembed_documents(texts)
            return embeddings

        except Exception as e:
            print(f"Error in batch embedding generation: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]  # Gemini embedding-001 dimension
