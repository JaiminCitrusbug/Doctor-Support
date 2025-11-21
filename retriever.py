import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB setup
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical-vectors")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def get_embedding(text, model=None):
    """Generate embedding for query."""
    if model is None:
        model = EMBEDDING_MODEL
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def retrieve_similar_chunks(query, top_k=3):
    """Retrieve top-k similar chunks for the given query using ChromaDB."""
    # Check if ChromaDB path exists
    if not os.path.exists(CHROMA_DB_PATH):
        raise ValueError(f"ChromaDB path not found: {CHROMA_DB_PATH}. Please run vector_store.py first to create the database.")
    
    query_embedding = get_embedding(query)
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get collection
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    except Exception as e:
        raise ValueError(f"Collection '{CHROMA_COLLECTION_NAME}' not found. Please run vector_store.py first to create the collection. Error: {e}")
    
    # Query ChromaDB for similar vectors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results to match the original structure
    chunks = []
    
    # ChromaDB returns results in a nested structure
    if results["documents"] and len(results["documents"]) > 0:
        documents = results["documents"][0]  # First query result
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
        
        for i, doc in enumerate(documents):
            # ChromaDB uses distance (lower is better), convert to similarity (higher is better)
            # For cosine distance: similarity = 1 - distance
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - distance  # Convert distance to similarity
            
            # Get text from document or metadata
            text = doc
            if not text and metadatas and i < len(metadatas):
                # Fallback to metadata text if document is empty
                text = metadatas[i].get("text", "")
            
            chunks.append({
                "text": text,
                "similarity": float(similarity)
            })
    
    return chunks
