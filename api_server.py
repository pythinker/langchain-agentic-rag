import os
from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from agent_rag import create_embeddings_from_pdfs, chat_with_agent, setup_database, get_vector_store, search_documents
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Constants
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b"
POSTGRES_CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING", 
    "postgresql://postgres:postgres@postgres:5432/postgres"
)
DB_TABLE_NAME = "message_store"

# FastAPI app with full WebSocket support
# To ensure WebSocket support, make sure 'websockets' is installed:
# pip install websockets==15.0.1
app = FastAPI(
    title="Agentic RAG API",
    description="API endpoints for Agentic RAG using LangChain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    """Chat input model"""
    chatInput: str = Field(..., description="The user message to process")
    sessionId: Optional[str] = Field(None, description="The session ID for conversation tracking")

class VectorStoreQuery(BaseModel):
    """Vector store query model"""
    query: str = Field(..., description="The query for similarity search")
    topK: int = Field(3, description="Number of top results to return")

class OllamaModelRequest(BaseModel):
    """Request for Ollama LLM"""
    prompt: str = Field(..., description="The prompt to send to the model")
    temperature: float = Field(0.0, description="Temperature for text generation")

@app.get("/webhook/create_source_embeddings", 
         summary="Create embeddings from PDF files", 
         description="Creates vector embeddings from PDF files in the shared directory")
async def create_embeddings():
    """
    Endpoint to create embeddings from PDF files in the shared directory
    This replicates the n8n webhook for creating embeddings
    """
    try:
        create_embeddings_from_pdfs()
        return {"response": "All embeddings are created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/invoke_n8n_agent", 
          summary="Chat with the agent", 
          description="Invoke the agent with a chat input and get a response")
async def invoke_agent(chat_request: ChatInput):
    """
    Endpoint to invoke the agent with a chat input
    This replicates the n8n webhook for chatting with the agent
    """
    try:
        chat_input = chat_request.chatInput
        session_id = chat_request.sessionId or str(uuid.uuid4())
        
        result = chat_with_agent(chat_input, session_id)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/setup_db", 
         summary="Set up the database", 
         description="Initialize the PostgreSQL database for chat history")
async def setup_db():
    """
    Endpoint to set up the database
    """
    try:
        success = setup_database()
        if success:
            return {"response": "Database setup completed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Database setup failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embeddings/ollama", 
          summary="Generate embeddings", 
          description="Generate embeddings using Ollama model")
async def generate_embeddings(texts: List[str] = Body(...)):
    """
    Generate embeddings using Ollama
    Replacement for n8n Embeddings Ollama node
    """
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        result = embeddings.embed_documents(texts)
        return {"embeddings": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vector_store/search", 
          summary="Search vector store", 
          description="Search for similar documents in the vector store")
async def search_vector_store(request: VectorStoreQuery):
    """
    Search for similar documents in the vector store
    Replacement for n8n Vector Store Tool node
    """
    try:
        results = search_documents(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/ollama", 
          summary="Chat with Ollama model", 
          description="Send a message to the Ollama chat model")
async def chat_with_ollama(request: OllamaModelRequest):
    """
    Chat with Ollama model
    Replacement for n8n Ollama Chat Model node
    """
    try:
        chat_model = ChatOllama(model=CHAT_MODEL, temperature=request.temperature, base_url=OLLAMA_BASE_URL)
        response = chat_model.invoke(request.prompt)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/ollama", 
          summary="Generate text with Ollama model", 
          description="Generate text using the Ollama LLM")
async def generate_with_ollama(request: OllamaModelRequest):
    """
    Generate text with Ollama model
    Replacement for n8n Ollama Model node
    """
    try:
        llm = ChatOllama(model=CHAT_MODEL, temperature=request.temperature, base_url=OLLAMA_BASE_URL)
        response = llm.invoke(request.prompt)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/memory/{session_id}", 
          summary="Get chat history", 
          description="Get the chat history for a session")
async def get_chat_memory(session_id: str):
    """
    Get chat history for a session
    Replacement for n8n Postgres Chat Memory node
    """
    try:
        import psycopg
        conn = psycopg.connect(POSTGRES_CONNECTION_STRING)
        
        message_history = PostgresChatMessageHistory(
            DB_TABLE_NAME,
            session_id,
            sync_connection=conn
        )
        
        messages = message_history.messages
        # Convert to dictionaries for JSON response
        message_dicts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                message_dicts.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                message_dicts.append({"role": "ai", "content": msg.content})
        
        return {"session_id": session_id, "messages": message_dicts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/memory/{session_id}/add", 
          summary="Add message to chat history", 
          description="Add a message to the chat history for a session")
async def add_to_chat_memory(
    session_id: str, 
    role: str = Body(...), 
    content: str = Body(...)
):
    """
    Add message to chat history
    Auxiliary function for the chat memory
    """
    try:
        import psycopg
        conn = psycopg.connect(POSTGRES_CONNECTION_STRING)
        
        message_history = PostgresChatMessageHistory(
            DB_TABLE_NAME,
            session_id,
            sync_connection=conn
        )
        
        if role.lower() == "human":
            message_history.add_user_message(content)
        elif role.lower() == "ai":
            message_history.add_ai_message(content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported role: {role}")
        
        return {"success": True, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", 
         summary="Health check", 
         description="Check if the API server is running")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "ok"}

if __name__ == '__main__':
    # Set up the database before starting the server
    setup_database()
    
    # Run the API server
    port = int(os.environ.get('PORT', 8000))  # Different port from n8n
    uvicorn.run(app, host="0.0.0.0", port=port) 