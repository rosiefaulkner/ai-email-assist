import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel
from app.tools.gmail import GmailClient
from app.config import Settings
from app.graph.workflow import RAGWorkflow
from app.services.email_sync import EmailSyncService
from app.models.schemas import Response, UserQuery


app = FastAPI(title="LangGraph RAG API")
settings = Settings()

# Initialize Langfuse
import os
from langfuse import get_client

# Langfuse environment variables
os.environ['LANGFUSE_PUBLIC_KEY'] = settings.LANGFUSE_PUBLIC_KEY
os.environ['LANGFUSE_SECRET_KEY'] = settings.LANGFUSE_SECRET_KEY
os.environ['LANGFUSE_HOST'] = settings.LANGFUSE_HOST

# Langfuse client
langfuse = get_client()
langfuse_handler = CallbackHandler()

# Configure middleware
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
    asyncio.create_task(email_sync_service.start())


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup when the app shuts down."""
    email_sync_service.stop()


@app.post("/query", response_model=Response)
async def process_query(request: UserQuery):
    """Process a user query using RAG."""
    trace = langfuse.trace(name="process_query")
    context = GmailClient()
    last_email = await context.get_last_email()
    print(f"Last email details: {context}")

    request = UserQuery(
        query="Analyze this email and determine if it is spam. Consider the sender, subject, content, and any suspicious patterns or red flags. Provide a clear explanation of why it is or is not spam.",
        context={
            "email_content": last_email['snippet'],
            "email_metadata": {
                f"from: {last_email['from']}",
                f"subject: {last_email['subject']}",
                f"date: {last_email['date']}",
            }
        }
        
    )
    try:
        if not request.query.strip():
            langfuse.api.observations.create(trace_id=trace.id, type="error", name="empty-query", error={"message": "Empty query"})
            return Response(
                answer=None,
                sources=[],
                metadata={"error_type": "ValueError"},
                error="Query cannot be empty",
            )

        langfuse.api.observations.create(
            trace_id=trace.id,
            type="span",
            name="input-processing",
            input={"query": request.query, "context": request.context}
        )

        result = await workflow.run(
            {"query": request.query, "context": request.context}
        )

        if not result or not result.get("answer"):
            langfuse.api.observations.create(trace_id=trace.id, type="error", name="no-response", error={"message": "No response generated"})
            return Response(
                answer=None,
                sources=result.get("sources", []),
                metadata={"error_type": "NoResponseError"},
                error="Failed to generate a response",
            )

        # Span for output processing
        langfuse.api.observations.create(
            trace_id=trace.id,
            type="span",
            name="output-processing",
            output={
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "metadata": result.get("metadata", {})
            }
        )
        print(f"hello: {result}")
        chain.invoke({"input": request.query}, config={"callbacks": [langfuse_handler]})

        return Response(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        error_msg = str(e)
        langfuse.api.observations.create(trace_id=trace.id, type="error", name="processing-error", error={"message": error_msg})
        print(f"Error processing query: {error_msg}")
        return Response(
            answer=None,
            sources=[],
            metadata={"error_type": type(e).__name__},
            error=error_msg,
        )
    finally:
    # Update trace status to complete
        langfuse.api.traces.update(trace.id, status="success")

async def main():
    context = GmailClient()
    last_email = await context.get_last_email()
    print(f"Last email details: {context}")
    request = {
        "query": "Analyze this email and determine if it is spam. Consider the sender, subject, content, and any suspicious patterns or red flags. Provide a clear explanation of why it is or is not spam.",
        "context":{
            f"email_content: last_email['snippet']",
            f"'from': {last_email['from']}, 'subject': {last_email['subject']}, 'snippet': {last_email['snippet']}",
        }
    }
    await process_query(request=request)
if __name__ == "__main__":
    asyncio.run(main()) 

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
