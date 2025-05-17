import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from agent_rag import create_embeddings_from_pdfs, chat_with_agent, setup_database
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# FastAPI app with full WebSocket support
# To ensure WebSocket support, make sure 'websockets' is installed:
# pip install websockets==15.0.1
app = FastAPI(
    title="Agentic RAG API",
    description="API endpoints for Agentic RAG using LangChain",
    version="1.0.0"
)

class ChatInput(BaseModel):
    """Chat input model"""
    chatInput: str = Field(..., description="The user message to process")
    sessionId: Optional[str] = Field(None, description="The session ID for conversation tracking")

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
        session_id = chat_request.sessionId
        
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
    port = int(os.environ.get('PORT', 5678))  # Match n8n port in the original implementation
    uvicorn.run(app, host="0.0.0.0", port=port) 