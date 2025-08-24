# AI Email Assistant: Gmail Spam Filter with Knowledge Graph & RAG

This intelligent email inbox helper implements a Knowledge Graph built on top of a tailored RAG (Retrieval Augmented Generation) system using FastAPI, LangGraph and Google Gemini to filter your Gmail inbox. It's an ambient agent running in the background, analyzing emails to identify spam, scams, and unwanted advertisements. The system learns your preferences for what to keep and what to discard, accessing this knowledge through vector retrieval.

## 🌟 Features

- **Email Analysis**: Automatically analyzes emails to determine if they are spam, scams, or legitimate communications
- **Background Processing**: Runs as an ambient agent that continuously monitors your inbox
- **RAG Implementation**: Uses Retrieval Augmented Generation to provide context-aware responses
- **LangGraph Workflow**: Orchestrates the conversation flow with a state-based graph architecture
- **Vector Storage**: Stores email embeddings for efficient similarity search using ChromaDB
- **Observability**: Integrated with Langfuse for tracing and monitoring AI interactions

## 🛠️ Technology Stack

- **FastAPI**: Modern, high-performance web framework for building APIs
- **Google Gemini AI**: Advanced language models for text generation and analysis
- **LangGraph**: Framework for building stateful, multi-step AI workflows
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **Gmail API**: For accessing and processing email content
- **Langfuse**: For tracing and monitoring AI interactions

## 📁 Project Structure

```
├── app/
│   ├── __init__.py
│   ├── agents/              # AI agents implementation
│   │   ├── __init__.py
│   │   ├── agent.py         # Base agent functionality
│   │   ├── gemini_agent.py  # Google Gemini agent
│   │   └── rag_agent.py     # RAG retrieval agent
│   ├── config.py            # Application configuration
│   ├── graph/
│   │   ├── __init__.py
│   │   └── workflow.py      # LangGraph workflow definition
│   ├── main.py              # FastAPI application entry point
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── email_sync.py    # Email synchronization service
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── gmail.py         # Gmail API client
│   │   ├── list_models.py   # Utility to list available models
│   │   └── named_entity_recognition.py # NER tool
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py    # Embedding utilities
│       └── vector_store.py  # Vector store operations
├── data/
│   ├── documents/           # Store documents for RAG
│   └── vector_store/        # ChromaDB storage
├── tests/                   # Test suite
│   ├── agents/
│   │   ├── test_gemini_agent.py
│   │   └── test_rag_agent.py
│   ├── graph/
│   │   └── test_workflow.py
│   ├── services/
│   │   └── test_email_sync.py
│   ├── tools/
│   │   └── test_gmail.py
│   └── utils/
│       ├── test_embeddings.py
│       └── test_vector_store.py
├── .env                     # Environment variables
├── .env.example             # Example environment variables
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 📋 Prerequisites

- Python 3.9 or higher (but less than 3.12)
- Gmail account with API access
- Google API credentials
- Google Gemini API key

## 🚀 Getting Started

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rosiefaulkner/ai-email-assist.git
   cd ai-email-assist
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. Create a `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```

2. Fill in the required environment variables in the `.env` file:
   - `GOOGLE_API_KEY`: Your Google API key for Gemini models
   - `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`: For Langfuse tracing (optional)
   - Other configuration options as needed

3. Set up Gmail API credentials:
   - Create a project in the [Google Cloud Console](https://console.cloud.google.com/)
   - Enable the Gmail API
   - Create OAuth credentials and download as `credentials.json`
   - Place the `credentials.json` file in the project root directory

## 🔍 Available Models

To list available Gemini models:

```bash
python -m app.tools.list_models
```

You can change the model by updating the `GEMINI_MODEL` setting in your `.env` file.

## 🏃‍♂️ Running the Application

### Start the API Server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Available Endpoints

- `POST /query`: Process a user query using RAG
- `GET /health`: Check if the API is running

## 📊 Email Synchronization

The application automatically synchronizes emails from your Gmail inbox at regular intervals (default: every 5 minutes). These emails are processed, embedded, and stored in the vector database for retrieval during query processing.

To manually run the email synchronization:

```bash
python -m app.services.email_sync
```

## 🧪 Testing

Run tests using pytest:

```bash
pytest
```

## 👥 Contributors

- [rosiefaulkner](mailto:faulknerproject@gmail.com)

## License

MIT
