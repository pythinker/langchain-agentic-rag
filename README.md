# LangChain Agentic RAG

A complete implementation of the Agentic RAG pattern using LangChain, replacing the n8n workflow from the original project.

![Agentic RAG Architecture](docs/agentic-rag.png)

## Overview

This project is a LangChain-based implementation of the Agentic RAG (Retrieval Augmented Generation) system that was originally implemented using n8n. It provides the same functionality but uses LangChain components directly instead of n8n nodes.

The system:

- Processes PDF documents to create vector embeddings
- Stores embeddings in Qdrant for similarity search
- Provides a chat interface with persistent conversation history in PostgreSQL
- Uses an agent with tools to intelligently answer questions about the documents
- Exposes API endpoints that match the original n8n webhooks

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd langchain-agentic-rag
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy the environment template and edit as needed:
```bash
cp env.template .env
```

4. Make sure the following services are running:
- Ollama (http://localhost:11434)
- Qdrant (http://localhost:6333)
- PostgreSQL (localhost:5432)

You can use the docker-compose.yml file from the original project to start these services:
```bash
docker compose up -d
```

5. Set up the PostgreSQL database for chat history:
```bash
python agent_rag.py setup_db
```

## Usage

### Command Line Interface

The application provides a command-line interface for interacting with the system:

1. Create embeddings from PDF files in the shared directory:
```bash
python agent_rag.py create_embeddings
```

2. Chat with the agent:
```bash
python agent_rag.py chat "What are the ingredients of Apple Berry Crisp?"
```

3. Continue a conversation with a specific session ID:
```bash
python agent_rag.py chat "How do I make it?" your-session-id
```

### Web API

The application also provides a web API that mimics the original n8n webhooks:

1. Start the API server:
```bash
python api_server.py
```

2. Create embeddings:
```bash
curl -X GET http://localhost:5678/webhook/create_source_embeddings
```

3. Chat with the agent:
```bash
curl -X POST http://localhost:5678/webhook/invoke_n8n_agent \
     -H "Content-Type: application/json" \
     -d '{"chatInput": "What are the ingredients of Apple Berry Crisp?", "sessionId": "c324038d8b2944a0855c2e40441038e3"}'
```

## Implementation Details

### Components

1. **Document Processing**: PDF loading and text splitting using LangChain document loaders
2. **Embedding Generation**: Using Ollama's nomic-embed-text model
3. **Vector Storage**: Using Qdrant for similarity search
4. **Chat History**: Using PostgreSQL for persistent conversation storage
5. **Agent**: Using LangChain tools and Ollama for LLM capabilities
6. **API Server**: Flask-based server that replicates the n8n webhooks

### Key Files

- `agent_rag.py`: The main implementation of the LangChain-based Agentic RAG system
- `api_server.py`: A Flask server that provides API endpoints matching the original n8n webhooks
- `agentic_rag.py`: Basic implementation for document embedding (kept for compatibility)
- `query_rag.py`: Simple query script for document retrieval (kept for compatibility)

## Requirements

- Python 3.8+
- Ollama running locally or accessible via URL
- Qdrant running locally or accessible via URL
- PostgreSQL running locally or accessible via URL

## Warning
Note that whenever you create embeddings, the previous embeddings and the related collection on Qdrant are deleted
