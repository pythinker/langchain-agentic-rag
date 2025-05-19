import os
import sys
import glob
import uuid
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_qdrant import QdrantVectorStore
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_postgres.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
import psycopg

# Load environment variables
load_dotenv()

# Constants
PDF_DIRECTORY = "shared"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"  # Matches what was used in n8n workflow
CHAT_MODEL = "qwen3:4b"  # Matches what was used in n8n workflow
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Use environment variable or default
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Use environment variable or default
POSTGRES_CONNECTION_STRING = os.getenv(
    "POSTGRES_CONNECTION_STRING", 
    "postgresql://postgres:postgres@localhost:5432/postgres"
)
DB_TABLE_NAME = "message_store"

# Database connection
db_conn = None

def get_db_connection():
    """Get or create a database connection"""
    global db_conn
    if db_conn is None:
        db_conn = psycopg.connect(POSTGRES_CONNECTION_STRING)
    return db_conn

def create_embeddings_from_pdfs() -> Optional[QdrantVectorStore]:
    """
    Read PDF files from the shared directory, create embeddings using Ollama,
    and store them in Qdrant.
    """
    print(f"Looking for PDF files in {PDF_DIRECTORY}")
    
    # Find all PDF files in the directory
    pdf_files = glob.glob(f"{PDF_DIRECTORY}/*.pdf")
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIRECTORY}")
        return None
    
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    
    # Initialize embeddings model with base_url parameter
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Initialize Qdrant client for collection management
    client = QdrantClient(url=QDRANT_URL)
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Process each PDF file
    all_documents = []
    
    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file}")
        try:
            # Load the PDF
            loader = PyPDFLoader(pdf_file, mode="single")
            documents = loader.load()
            
            # Split the document into chunks
            split_documents = text_splitter.split_documents(documents)
            
            print(f"  - Extracted {len(split_documents)} chunks from {pdf_file}")
            
            # Add to our collection
            all_documents.extend(split_documents)
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
    
    # Store documents in Qdrant
    if all_documents:
        print(f"Creating vector store with {len(all_documents)} document chunks")
        
        # Delete collection if it exists to start fresh
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # Collection doesn't exist yet
        
        # Create texts and metadata for embedding
        texts = [doc.page_content for doc in all_documents]
        metadatas = [doc.metadata for doc in all_documents]
        
        # Use Qdrant's from_texts method
        vector_store = QdrantVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=True
        )
        
        print(f"Successfully created embeddings and stored them in Qdrant collection '{COLLECTION_NAME}'")
        return vector_store
    else:
        print("No documents were processed.")
        return None

def get_vector_store() -> QdrantVectorStore:
    """
    Connect to the existing Qdrant vector store
    """
    # Initialize embeddings model with base_url parameter
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL)
    
    # Connect to the existing vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    return vector_store

@tool
def search_documents(query: str) -> str:
    """
    Search the document collection for information related to the query.
    Use this tool to find information from the documents to answer the user's questions.
    Return relevant document content that helps answer the question.
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant information found in the documents."
    
    # Format the results
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown source')
        page = doc.metadata.get('page', 'Unknown page')
        content = doc.page_content
        
        result = f"Document {i+1} (Source: {source}, Page: {page}):\n{content}\n"
        results.append(result)
    
    return "\n".join(results)

def create_agent():
    """
    Create an agentic RAG system with chat capabilities and document search
    """
    # Initialize the chat model with base_url parameter
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.0, base_url=OLLAMA_BASE_URL)
    
    # Define tools
    tools = [search_documents]
    
    # Define the system prompt
    system_prompt = """You are a helpful AI assistant that has access to documents about recipes. 
    You can use the search_documents tool to look for information in these documents.
    
    When asked a question, first determine if you need to search for information. 
    If the question is about recipe ingredients, instructions, or other specific details,
    use the search_documents tool to find relevant information before answering.
    
    Always be helpful, accurate, and base your answers on the information in the documents when available.
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

def get_chat_history(session_id: str):
    """Get the chat history for a session"""
    try:
        conn = get_db_connection()
        
        # Initialize chat history
        message_history = PostgresChatMessageHistory(
            DB_TABLE_NAME,
            session_id,
            sync_connection=conn
        )
        
        # Return formatted chat history
        return message_history.messages
    except Exception as e:
        print(f"Error retrieving chat history: {str(e)}")
        return []

def chat_with_agent(chat_input: str, session_id: str = None):
    """Chat with the agent using the provided input"""
    # Create or get session ID
    if not session_id:
        session_id = str(uuid.uuid4())
    
    print(f"Session ID: {session_id}")
    
    try:
        conn = get_db_connection()
        
        # Initialize chat history
        message_history = PostgresChatMessageHistory(
            DB_TABLE_NAME,
            session_id,
            sync_connection=conn
        )
        
        # Get chat history
        chat_history = get_chat_history(session_id)
        
        # Create the agent
        agent_executor = create_agent()
        
        # Add the message to history
        message_history.add_user_message(chat_input)
        
        # Run the agent
        result = agent_executor.invoke({
            "input": chat_input,
            "chat_history": chat_history
        })
        
        # Add the AI response to history
        message_history.add_ai_message(result["output"])
        
        return {
            "response": result["output"],
            "session_id": session_id
        }
    except Exception as e:
        print(f"Error in chat_with_agent: {str(e)}")
        # Try to continue without storage
        agent_executor = create_agent()
        result = agent_executor.invoke({
            "input": chat_input,
            "chat_history": []
        })
        return {
            "response": result["output"],
            "session_id": session_id,
            "error": str(e)
        }

def setup_database():
    """Set up the PostgreSQL database for chat history"""
    try:
        conn = psycopg.connect(POSTGRES_CONNECTION_STRING)
        
        # Create the table
        PostgresChatMessageHistory.create_tables(conn, DB_TABLE_NAME)
        
        print("Database setup complete.")
        return True
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        return False

def main():
    """
    Main function to process command-line arguments and run the appropriate function
    """
    if len(sys.argv) < 2:
        print("Usage: python agent_rag.py [create_embeddings|chat|setup_db]")
        print("  create_embeddings - Create embeddings from PDFs in the shared directory")
        print("  chat \"your question\" [session_id] - Chat with the agentic RAG system")
        print("  setup_db - Set up the PostgreSQL database for chat history")
        return
    
    command = sys.argv[1]
    
    if command == "create_embeddings":
        create_embeddings_from_pdfs()
    
    elif command == "chat":
        if len(sys.argv) < 3:
            print("Please provide a question to ask")
            return
        
        question = sys.argv[2]
        session_id = sys.argv[3] if len(sys.argv) > 3 else None
        
        result = chat_with_agent(question, session_id)
        print(f"\nResponse: {result['response']}")
        print(f"Session ID: {result['session_id']}")
    
    elif command == "setup_db":
        setup_database()
    
    else:
        print(f"Unknown command: {command}")
        print("Usage: python agent_rag.py [create_embeddings|chat|setup_db]")

if __name__ == "__main__":
    main() 