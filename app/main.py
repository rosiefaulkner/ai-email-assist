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
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST
)

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
async def process_query():
    """Process a user query using RAG."""
    # trace = langfuse.trace(name="process_query")
    context = GmailClient()
    last_email = await context.get_last_email()
    print(f"Last email details: {context}")

    request = UserQuery(
        query: "Analyze this email and determine if it is spam. Consider the sender, subject, content, and any suspicious patterns or red flags. Provide a clear explanation of why it is or is not spam.",
        context: {
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
            # trace.error("Empty query")
            return Response(
                answer=None,
                sources=[],
                metadata={"error_type": "ValueError"},
                error="Query cannot be empty",
            )

        # trace.span(
        #     name="input",
        #     input={"query": request.query, "context": request.context}
        # )

        result = await workflow.run(
            {"query": request.query, "context": request.context}
        )

        if not result or not result.get("answer"):
            # trace.error("No response generated")
            return Response(
                answer=None,
                sources=result.get("sources", []),
                metadata={"error_type": "NoResponseError"},
                error="Failed to generate a response",
            )

        # trace.span(
        #     name="output",
        #     output={
        #         "answer": result["answer"],
        #         "sources": result.get("sources", []),
        #         "metadata": result.get("metadata", {})
        #     }
        # )
        print(f"hello: {result}")
        workflow.invoke({"input": request.query}, config={"callbacks": [langfuse_handler]})

        return Response(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {}),
        )

    except Exception as e:
        error_msg = str(e)
        # trace.error(error_msg)
        print(f"Error processing query: {error_msg}")
        return Response(
            answer=None,
            sources=[],
            metadata={"error_type": type(e).__name__},
            error=error_msg,
        )
    # finally:
    #     await trace.end()

async def main():
    await process_query()
if __name__ == "__main__":
    asyncio.run(main()) 

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
