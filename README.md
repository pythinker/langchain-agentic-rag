# LangChain Agentic RAG

A complete implementation of the Agentic RAG pattern using LangChain, with n8n for visualization and diagnosis.

![Agentic RAG Architecture](docs/agentic-rag.png)

## Overview

This project implements an Agentic RAG (Retrieval Augmented Generation) system using LangChain with a FastAPI server that provides all the functionality, while n8n is used for visualization and workflow diagnosis using HTTP Request nodes.

The system:

- Processes PDF documents to create vector embeddings
- Stores embeddings in Qdrant for similarity search
- Provides a chat interface with persistent conversation history in PostgreSQL
- Uses an agent with tools to intelligently answer questions about the documents
- Exposes API endpoints that are called by n8n's HTTP Request nodes

## Architecture

The project uses a decoupled architecture:

1. **FastAPI Server**: Provides all the RAG functionality through RESTful endpoints
2. **n8n**: Visualizes the workflow and provides a user interface, but only uses HTTP Request nodes to call the API server
3. **Qdrant**: Vector database for storing document embeddings
4. **PostgreSQL**: Relational database for storing chat history
5. **Ollama**: Local LLM for text generation and embeddings

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd langchain-agentic-rag
```

2. Copy the environment template and edit as needed:
```bash
cp env.template .env
```

3. Start all services with Docker Compose:
```bash
docker compose up -d
```

This will start:
- PostgreSQL database
- pgAdmin web interface
- Qdrant vector database
- FastAPI server (our custom implementation)
- n8n (for visualization)

4. Access the services:
- n8n: http://localhost:5678
- API Server: http://localhost:8000
- pgAdmin: http://localhost:5050
- Qdrant: http://localhost:6333

## Using the System

### Through n8n

1. In n8n, import the workflow file `agentic_rag_workflow_http.json` which uses HTTP Request nodes
2. Trigger the "Create Embeddings" workflow to process PDF files in the shared directory
3. Use the chat interface in n8n to interact with the agent

### Directly via API

The FastAPI server provides these main endpoints:

1. Create embeddings:
```bash
curl -X GET http://localhost:8000/webhook/create_source_embeddings
```

2. Chat with the agent:
```bash
curl -X POST http://localhost:8000/webhook/invoke_n8n_agent \
     -H "Content-Type: application/json" \
     -d '{"chatInput": "What are the ingredients of Apple Berry Crisp?", "sessionId": "c324038d8b2944a0855c2e40441038e3"}'
```

3. Access the API documentation:
```
http://localhost:8000/docs
```

## Implementation Details

### Components

1. **API Server** (`api_server.py`): FastAPI server that provides all RAG functionality
2. **Agent Logic** (`agent_rag.py`): Core LangChain implementation of the Agentic RAG system
3. **n8n Workflow** (`agentic_rag_workflow_http.json`): n8n workflow using HTTP Request nodes
4. **Docker Configuration**: Multi-container setup with services for API, n8n, databases

### Project Structure

- `api_server.py`: FastAPI server implementation
- `agent_rag.py`: LangChain agent implementation
- `agentic_rag_workflow_http.json`: n8n workflow with HTTP Request nodes
- `docker-compose.yml`: Docker Compose configuration for all services
- `Dockerfile.api`: Dockerfile for the API server
- `shared/`: Directory for PDF files to be processed

## Dependencies

The project uses specific versions of the following key packages:
- langchain==0.3.25
- langchain-ollama==0.3.3
- langchain-qdrant==0.2.0
- langchain-postgres==0.0.14
- qdrant-client==1.14.2
- fastapi==0.115.12
- uvicorn==0.34.2
- psycopg2-binary==2.9.10

See requirements.txt for the complete list of dependencies with exact versions.

## Requirements

- Docker and Docker Compose
- Ollama running locally or accessible via URL

## Warning
Note that whenever you create embeddings, the previous embeddings and the related collection on Qdrant are deleted
