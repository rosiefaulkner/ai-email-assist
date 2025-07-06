from typing import Any, Dict, List

from ..config import get_settings
from ..utils.embeddings import EmbeddingUtil
from ..utils.vector_store import VectorStore
from .gemini_agent import GeminiAgent

"""
RAGAgent class for processing queries using RAG methodology.
Args:
    settings (Settings)
    vector_store (VectorStore)
    embedding_util (EmbeddingUtil)
    llm (LLM)
    query (str)
    **kwargs (_type_)
"""


class RAGAgent:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = VectorStore()
        self.embedding_util = EmbeddingUtil()
        self.llm = GeminiAgent()

    async def process_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Process a query using RAG methodology.
        Args:
            query (str): The query to process
            **kwargs: Additional keyword arguments for LLM
        Returns:
            Dict[str, Any]: The response from the LLM
        """
        try:
            # Get relevant documents
            relevant_docs = await self.retrieve_relevant_documents(query)

            # Extract and format context from relevant documents
            context = self._prepare_context(relevant_docs) if relevant_docs else []

            # Generate response using LLM with context
            response = await self.llm.generate_response(
                query=query, context=context, **kwargs
            )

            if response.get("error"):
                print(f"Error generating response: {response['error']}")
                return response

            # Add source information to response
            response["sources"] = (
                [
                    {
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "relevance_score": doc["score"],
                    }
                    for doc in relevant_docs
                ]
                if relevant_docs
                else []
            )

            # Ensure we have an answer even if no relevant docs found
            if not response.get("answer"):
                response["answer"] = (
                    "I couldn't find any relevant information in your emails to answer this question."
                )

            return response

        except Exception as e:
            return {
                "error": str(e),
                "answer": None,
                "sources": [],
                "metadata": {"error_type": type(e).__name__},
            }

    async def retrieve_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for the query.
        Args:
            query (str): The query to search for
        Returns:
            List[Dict[str, Any]]: List of relevant documents with 'content', 'metadata', and'relevance_score'
        """
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
        """
        Prepare the context for the LLM by formatting the documents.
        Args:
            documents (List[Dict[str, Any]]): List of documents with 'content' and 'metadata'
        Returns:
            List[str]: Formatted context
        """
        context = []
        for doc in documents:
            # Format the document content with metadata
            source_info = f"Source: {doc['metadata'].get('source', 'Unknown')}"
            context_entry = f"{doc['content']}\n{source_info}"
            context.append(context_entry)

        return context

    async def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a new document to the vector store.
        Args:
            content (str): The content of the document
            metadata (Dict[str, Any], optional): Additional metadata for the document
        Returns:
            bool: True if document was added successfully, False otherwise
        """
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
