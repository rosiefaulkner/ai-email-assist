from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import Settings
from .graph.workflow import RAGWorkflow
from .models.schemas import Response, UserQuery

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

# Initialize workflow
workflow = RAGWorkflow()


class QueryRequest(BaseModel):
    query: str
    context: Dict[str, Any] = {}


@app.post("/query", response_model=Response)
async def process_query(request: QueryRequest):
    try:
        # Process the query through the LangGraph workflow
        result = workflow.run({"query": request.query, "context": request.context})

        return Response(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
