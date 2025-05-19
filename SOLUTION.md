# Agentic RAG Solution with LangChain and n8n Integration

## Solution Overview

I've implemented a decoupled architecture where:

1. A FastAPI server provides all the LangChain-based RAG functionality through RESTful API endpoints
2. n8n is used purely for visualization and workflow control, using only HTTP Request nodes to interact with the API server

This approach allows us to:
- Keep all the LangChain implementation logic in a pure Python server
- Use n8n as a visualization and orchestration layer
- Maintain a clean separation of concerns

## Key Components

### 1. Enhanced API Server (api_server.py)

The API server now provides endpoints for all functionality previously handled by n8n native nodes:

- `/webhook/create_source_embeddings` - Process PDFs and create embeddings
- `/webhook/invoke_n8n_agent` - Main agent chat endpoint
- `/api/embeddings/ollama` - Generate embeddings with Ollama
- `/api/vector_store/search` - Search the vector store
- `/api/chat/ollama` - Chat with Ollama model
- `/api/llm/ollama` - Generate text with Ollama model
- `/api/chat/memory/{session_id}` - Get chat history
- `/api/chat/memory/{session_id}/add` - Add to chat history

### 2. n8n Workflow with HTTP Request Nodes (agentic_rag_workflow_http.json)

Created a new n8n workflow that:
- Uses HTTP Request nodes to call the API server endpoints
- Preserves the same visual layout and flow as the original workflow
- Maintains the same functional capabilities

### 3. Docker Integration (docker-compose.yml, Dockerfile.api)

- Added an `api-server` service to the Docker Compose file
- Created a dedicated Dockerfile for the API server
- Configured networking between services

### 4. Documentation (README.md)

- Updated documentation to explain the new architecture
- Provided instructions for using both n8n and direct API access
- Explained the relationship between components

## Benefits of this Approach

1. **Separation of Concerns**: Core RAG logic is in pure Python code, while n8n handles visualization
2. **Maintainability**: Python code can be developed and tested independently of n8n
3. **Performance**: Direct API calls can be more efficient than n8n node execution
4. **Flexibility**: API can be accessed by any client, not just n8n
5. **Debuggability**: API server logs can provide detailed information about execution

## How to Use

1. Start all services with Docker Compose:
   ```
   docker-compose up -d
   ```

2. Access n8n and import/activate the new workflow:
   ```
   http://localhost:5678
   ```

3. Place PDF files in the shared directory to be processed

4. Use n8n workflow to:
   - Create embeddings from PDF files
   - Chat with the agent about the document content

5. Alternatively, interact directly with the API:
   ```
   http://localhost:8000/docs
   ```

## Technical Details

- FastAPI is running on port 8000
- n8n is running on port 5678
- The API server connects to Postgres and Qdrant running in other containers
- The system uses Ollama for LLM capabilities and embeddings
- Chat history is stored in PostgreSQL for persistence across sessions 