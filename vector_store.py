# vector_store.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ChromaDB setup
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical-vectors")

INPUT_JSON = os.getenv("INPUT_JSON", "new_data.json")

def build_text_for_embedding(doc: dict) -> str:
    """
    Construct a single text blob from the Wockhardt product record.
    Uses available fields safely (some keys may not exist).
    """
    fields_in_order = [
        "product_name", "brand_name", "therapeutic_class", "strength",
        "dosage_form", "pack_size", "composition", "indication_summary",
        "extracted_text"
    ]
    parts = []
    for k in fields_in_order:
        v = doc.get(k)
        if v:
            parts.append(f"{k.replace('_',' ').title()}: {v}")
    # Fallback if nothing present
    if not parts:
        parts.append(doc.get("id", ""))
    text_blob = "\n".join(parts).strip()
    # keep it within a reasonable size for embeddings
    return text_blob[:6000]

def create_embedding(text: str):
    """Generate an embedding vector for a given text."""
    if not text:
        text = " "  # avoid empty input to embeddings API
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def clean_metadata(metadata: dict) -> dict:
    """Remove None values from metadata dict. ChromaDB doesn't accept None values."""
    return {k: v for k, v in metadata.items() if v is not None and v != ""}

def store_embeddings():
    """Store embeddings in ChromaDB collection."""
    # Get embedding dimension from OpenAI model
    print(f"Getting embedding dimension for model: {EMBEDDING_MODEL}")
    sample_embedding = create_embedding("sample")
    embedding_dimension = len(sample_embedding)
    print(f"✅ Embedding dimension: {embedding_dimension}")
    
    # Initialize ChromaDB client (persistent mode)
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Check if collection exists, get or create
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✅ Collection '{CHROMA_COLLECTION_NAME}' already exists")
        
        # Check if collection is empty or needs update
        count = collection.count()
        print(f"   Current document count: {count}")
        
        # Option: Clear existing collection if you want to rebuild
        # Uncomment the next 2 lines if you want to rebuild from scratch
        # chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        # collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)
        
    except Exception:
        # Collection doesn't exist, create it
        print(f"Creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        collection = chroma_client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"✅ Collection {CHROMA_COLLECTION_NAME} created successfully!")
    
    # Load JSON
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        docs = json.load(f)
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    print(f"\nProcessing {len(docs)} documents...")
    
    for i, doc in enumerate(docs):
        uid = doc.get("id", f"vec_{i}")
        title = doc.get("product_name") or doc.get("brand_name") or uid
        source_id = doc.get("source_url") or ""  # Ensure it's never None
        text = build_text_for_embedding(doc)
        embedding = create_embedding(text)
        
        # ChromaDB metadata - only include non-None, non-empty values
        metadata = {
            "title": title or uid,  # Ensure title is never None/empty
            "chunk_index": "0",
        }
        
        # Add source_id only if it exists
        if source_id:
            metadata["source_id"] = source_id
        
        # Add optional fields only if they exist and are not None/empty
        if doc.get("product_name"):
            metadata["product_name"] = doc.get("product_name")
        if doc.get("therapeutic_class"):
            metadata["therapeutic_class"] = doc.get("therapeutic_class")
        if doc.get("strength"):
            metadata["strength"] = doc.get("strength")
        if doc.get("dosage_form"):
            metadata["dosage_form"] = doc.get("dosage_form")
        
        # Clean metadata to remove any None or empty values
        metadata = clean_metadata(metadata)
        
        ids.append(uid)
        embeddings.append(embedding)
        metadatas.append(metadata)
        documents.append(text)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(docs)} documents...")
    
    # Batch add to ChromaDB (ChromaDB handles batching internally, but we can do it manually for large datasets)
    batch_size = 100
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"\nAdding documents to ChromaDB in {total_batches} batches...")
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
        
        batch_num = i // batch_size + 1
        print(f"   Added batch {batch_num}/{total_batches} ({len(batch_ids)} documents)")
    
    # Verify final count
    final_count = collection.count()
    print(f"\n✅ All embeddings stored successfully in ChromaDB!")
    print(f"   Total documents in collection: {final_count}")
    print(f"   Collection path: {os.path.abspath(CHROMA_DB_PATH)}")

if __name__ == "__main__":
    store_embeddings()
