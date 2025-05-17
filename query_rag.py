import sys
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Constants
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"  # Default for local Qdrant

def query_vector_store(query):
    """Query the vector store with the given query"""
    # Initialize embeddings model
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Initialize Qdrant client
    client = QdrantClient(url=QDRANT_URL)
    
    # Connect to the existing vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    # Perform the query
    docs = vector_store.similarity_search(query, k=3)
    
    # Print results
    print(f"Query: '{query}'")
    print(f"Retrieved {len(docs)} documents\n")
    
    for i, doc in enumerate(docs):
        print(f"Result {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content: {doc.page_content[:300]}...\n")
    
    return docs

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use the command line argument as the query
        query = " ".join(sys.argv[1:])
    else:
        # Default query
        query = "What are the ingredients of Apple Berry Crisp?"
    
    query_vector_store(query) 