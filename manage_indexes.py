"""
Helper script to manage ChromaDB collections.
Use this to list, describe, or delete collections.
"""
import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

if not os.path.exists(CHROMA_DB_PATH):
    print(f"⚠️  ChromaDB path not found: {CHROMA_DB_PATH}")
    print("   The database will be created when you first run vector_store.py")
    print("   Or create the directory manually if needed.")
    exit(1)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)

def list_collections():
    """List all collections."""
    print("📋 Available ChromaDB Collections:\n")
    try:
        collections = chroma_client.list_collections()
        
        if not collections:
            print("   No collections found.")
            return
        
        for collection in collections:
            try:
                count = collection.count()
                print(f"   Name: {collection.name}")
                print(f"   Document Count: {count}")
                print()
            except Exception as e:
                print(f"   Name: {collection.name}")
                print(f"   Error getting details: {e}")
                print()
    except Exception as e:
        print(f"❌ Error listing collections: {e}")

def describe_collection(collection_name: str):
    """Describe a specific collection."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"📊 Collection Details for '{collection_name}':\n")
        
        count = collection.count()
        print(f"   Document Count: {count}")
        
        # Get metadata info
        try:
            # Sample a few documents to show metadata structure
            sample_results = collection.peek(limit=1)
            if sample_results["metadatas"] and len(sample_results["metadatas"]) > 0:
                print(f"\n   Sample Metadata Keys: {list(sample_results['metadatas'][0].keys())}")
        except Exception as e:
            print(f"   Could not get metadata info: {e}")
        
        print(f"\n   Database Path: {os.path.abspath(CHROMA_DB_PATH)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def delete_collection(collection_name: str, confirm: bool = False):
    """Delete a collection."""
    if not confirm:
        print(f"⚠️  To delete collection '{collection_name}', run with confirm=True")
        print(f"   Example: delete_collection('{collection_name}', confirm=True)")
        return
    
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"✅ Collection '{collection_name}' deleted successfully!")
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_collections()
        elif command == "describe" and len(sys.argv) > 2:
            describe_collection(sys.argv[2])
        elif command == "delete" and len(sys.argv) > 2:
            collection_name = sys.argv[2]
            confirm = len(sys.argv) > 3 and sys.argv[3].lower() == "confirm"
            delete_collection(collection_name, confirm)
        else:
            print("Usage:")
            print("  python manage_indexes.py list")
            print("  python manage_indexes.py describe <collection_name>")
            print("  python manage_indexes.py delete <collection_name> confirm")
    else:
        list_collections()
        print("\nUsage:")
        print("  python manage_indexes.py list")
        print("  python manage_indexes.py describe <collection_name>")
        print("  python manage_indexes.py delete <collection_name> confirm")
