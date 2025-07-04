from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    """Schema for user query requests."""

    query: str = Field(..., description="The user's question or query")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context for the query"
    )


class Source(BaseModel):
    """Schema for source documents used in RAG."""

    content: str = Field(..., description="The content of the source document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the source document"
    )
    relevance_score: float = Field(
        ..., description="Relevance score of the document to the query", ge=0, le=1
    )


class Response(BaseModel):
    """Schema for API responses."""

    answer: Optional[str] = Field(
        None, description="The generated response to the query"
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="List of source documents used for the response",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )
    error: Optional[str] = Field(
        None, description="Error message if something went wrong"
    )


class DocumentInput(BaseModel):
    """Schema for adding new documents to the RAG system."""

    content: str = Field(..., description="The content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata about the document"
    )


class ValidationResponse(BaseModel):
    """Schema for validation responses."""

    is_valid: bool = Field(..., description="Whether the response is valid")
    quality_score: float = Field(
        ..., description="Quality score of the response", ge=0, le=1
    )
    feedback: Optional[str] = Field(
        None, description="Feedback about the response quality"
    )
