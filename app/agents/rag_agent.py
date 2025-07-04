from typing import Any, Dict, List

from ..config import get_settings
from ..utils.embeddings import EmbeddingUtil
from ..utils.vector_store import VectorStore
from .gemini_agent import GeminiAgent


class RAGAgent:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.embedding_util = EmbeddingUtil()
        self.llm = GeminiAgent()

    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query using RAG methodology."""
        try:
            # Get relevant documents
            relevant_docs = await self.retrieve_relevant_documents(query)

            # If no relevant documents found
            if not relevant_docs:
                return await self.llm.generate_response(query)

            # Extract and format context from relevant documents
            context = self._prepare_context(relevant_docs)

            # Generate response using LLM with context
            response = await self.llm.generate_response(
                query=query, context=context, **kwargs
            )

            # Add source information to response
            response["sources"] = [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "relevance_score": doc["score"],
                }
                for doc in relevant_docs
            ]

            return response

        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "sources": [],
                "metadata": {"error_type": type(e).__name__},
            }

    async def retrieve_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query."""
        # Get initial candidates from vector store
        candidates = await self.vector_store.similarity_search(
            query=query, k=self.settings.MAX_DOCUMENTS
        )

        # Analyze relevance of each document
        relevant_docs = []
        for doc in candidates:
            relevance_score = await self.llm.analyze_relevance(query, doc["content"])

            if relevance_score >= self.settings.SIMILARITY_THRESHOLD:
                doc["score"] = relevance_score
                relevant_docs.append(doc)

        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x["score"], reverse=True)

        return relevant_docs

    def _prepare_context(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Prepare context from relevant documents."""
        context = []
        for doc in documents:
            # Format the document content with metadata
            source_info = f"Source: {doc['metadata'].get('source', 'Unknown')}"
            context_entry = f"{doc['content']}\n{source_info}"
            context.append(context_entry)

        return context

    async def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a new document to the vector store."""
        try:
            # Generate embeddings for the document
            embedding = await self.embedding_util.get_embedding(content)

            # Add to vector store
            await self.vector_store.add_documents(
                [
                    {
                        "content": content,
                        "embedding": embedding,
                        "metadata": metadata or {},
                    }
                ]
            )

            return True
        except Exception:
            return False
