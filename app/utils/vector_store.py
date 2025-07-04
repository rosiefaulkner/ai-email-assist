import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import get_settings


class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.client = chromadb.PersistentClient(
            path=self.settings.VECTOR_STORE_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store.

        Args:
            documents: List of documents with 'content', 'embedding', and 'metadata'
        """
        try:
            # Prepare document batches
            ids = [
                str(i)
                for i in range(
                    self.collection.count(), self.collection.count() + len(documents)
                )
            ]

            embeddings = [doc["embedding"] for doc in documents]
            contents = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            # Add documents to collection
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=contents,
                    metadatas=metadatas,
                ),
            )

            return True
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            return False

    async def similarity_search(
        self, query: str, k: int = 5, filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using embeddings.

        Args:
            query: Query embedding
            k: Number of results to return
            filter: Optional metadata filter
        """
        try:
            # Get results from collection
            results = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self.collection.query(
                    query_embeddings=[query], n_results=k, where=filter
                ),
            )

            # Format results
            documents = []
            for i in range(len(results["documents"][0])):
                documents.append(
                    {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

            return documents
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                self._executor, lambda: self.collection.delete(ids=ids)
            )
            return True
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            return False

    async def update_document(
        self,
        id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """Update a document in the vector store.

        Args:
            id: Document ID to update
            content: New document content
            embedding: New document embedding
            metadata: New document metadata
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                lambda: self.collection.update(
                    ids=[id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata] if metadata else None,
                ),
            )
            return True
        except Exception as e:
            print(f"Error updating document: {str(e)}")
            return False
