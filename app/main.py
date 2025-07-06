import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import Settings
from .graph.workflow import RAGWorkflow
from .models.schemas import Response, UserQuery
from .services.email_sync import EmailSyncService

app = FastAPI(title="LangGraph RAG API")
settings = Settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
workflow = RAGWorkflow()
email_sync_service = EmailSyncService()


@app.on_event("startup")
async def startup_event():
    """Start background services when the app starts."""
    # Start email sync service in the background
    asyncio.create_task(email_sync_service.start())


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup when the app shuts down."""
    email_sync_service.stop()


@app.post("/query", response_model=Response)
async def process_query(request: UserQuery):
    """Process a user query using RAG."""
    try:
        if not request.query.strip():
            return Response(
                answer=None,
                sources=[],
                metadata={"error_type": "ValueError"},
                error="Query cannot be empty",
            )

        result = await workflow.run(
            {"query": request.query, "context": request.context}
        )

        if not result or not result.get("answer"):
            return Response(
                answer=None,
                sources=result.get("sources", []),
                metadata={"error_type": "NoResponseError"},
                error="Failed to generate a response",
            )

        return Response(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {}),
        )
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return Response(
            answer=None,
            sources=[],
            metadata={"error_type": type(e).__name__},
            error=str(e),
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
