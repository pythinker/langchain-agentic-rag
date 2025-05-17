import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Constants
PDF_DIRECTORY = "shared"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "nomic-embed-text"  # Matches what was used in n8n workflow
QDRANT_URL = "http://localhost:6333"  # Default for local Qdrant

def create_embeddings_from_pdfs():
    """
    Read PDF files from the shared directory, create embeddings using Ollama,
    and store them in Qdrant.
    """
    print(f"Looking for PDF files in {PDF_DIRECTORY}")
    
    # Find all PDF files in the directory
    pdf_files = glob.glob(f"{PDF_DIRECTORY}/*.pdf")
    
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIRECTORY}")
        return
    
    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    
    # Initialize embeddings model
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
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
            loader = PyPDFLoader(pdf_file)
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

def test_retrieval(vector_store, query="What are the ingredients of Apple Berry Crisp?"):
    """Test retrieval from the vector store"""
    if not vector_store:
        print("No vector store available for retrieval test.")
        return
    
    print(f"\nTesting retrieval with query: '{query}'")
    
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=3)
    
    print(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        print(f"\nResult {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content snippet: {doc.page_content[:150]}...")

if __name__ == "__main__":
    print("Starting Agentic RAG test using LangChain")
    
    # Create embeddings from PDFs and store in Qdrant
    vector_store = create_embeddings_from_pdfs()
    
    # Test retrieval if we have a vector store
    if vector_store:
        test_retrieval(vector_store)
    
    print("\nTest completed.") 