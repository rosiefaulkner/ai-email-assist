from typing import List, Union
import google.generativeai as genai
import numpy as np
import time
from tenacity import retry, wait_exponential, stop_after_attempt

from ..config import get_settings

class EmbeddingUtil:
    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.GOOGLE_API_KEY)

    @retry(wait=wait_exponential(multiplier=2, min=10, max=30), stop=stop_after_attempt(5))
    async def get_embedding(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        try:
            # Handle single text input
            if isinstance(text, str):
                result = genai.embed_content(model="embedding-001", content=text)
                time.sleep(3)  # Add longer delay between requests
                return result['embedding']

            # Handle batch processing
            embeddings = []
            for t in text:
                result = genai.embed_content(model="embedding-001", content=t)
                time.sleep(3)  # Add longer delay between requests
                embeddings.append(result['embedding'])
            return embeddings

        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            dim = 768  # embedding-001 dimension
            return [0.0] * dim if isinstance(text, str) else [[0.0] * dim]

    @retry(wait=wait_exponential(multiplier=2, min=10, max=30), stop=stop_after_attempt(5))
    async def batch_get_embeddings(
        self, texts: List[str], batch_size: int = 5
    ) -> List[List[float]]:
        try:
            if not texts:
                return []
                
            # Process texts in batches
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                for text in batch:
                    result = genai.embed_content(model="embedding-001", content=text)
                    time.sleep(3)  # Add longer delay between requests
                    batch_embeddings.append(result['embedding'])
                time.sleep(5)  # Add delay between batches
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings

        except Exception as e:
            print(f"Error in batch embedding generation: {str(e)}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]  # embedding-001 dimension

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
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
