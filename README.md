# Gmail Agentic AI Spam Filter with Knowledge Graph & RAG
This email inbox helper implements a Knowledge Graph built on top of a tailored RAG (Retrieval Augmented Generation) system using FastAPI, LangGraph and Gemini to filter your Gmail inbox. It's an ambient agent running in the background, tossing spam, scams and ads in the trash. It learns your preferences for what to keep and what to throw away and accesses this knowledge through vector retrieval. It doesn't require a prompt or triggers. 

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── gemini_agent.py  # Google Gemini agent implementation
│   │   └── rag_agent.py     # RAG retrieval agent
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py      # LangGraph workflow definition
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py    # Embedding utilities
│       └── vector_store.py  # Vector store operations
├── data/
│   └── documents/          # Store documents for RAG
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   └── test_workflow.py
├── .env                    # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with:
```
GOOGLE_API_KEY=your_api_key_here
```

## Models
Switch models by running the below to list available models:
```python
python3 -m app.tools.list_models
```

## Usage

Run the FastAPI application:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Features

- Integration with Google Gemini AI for advanced language processing
- RAG implementation for knowledge retrieval
- LangGraph for orchestrating the conversation flow
- FastAPI backend for API endpoints
- Chromadb for vector storage

## License

MIT
