import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "chroma_db"


client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = client.get_or_create_collection(
    name="documents"
)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def store_chunks(chunks: list[str], source: str = "unknown"):
    embeddings = embedding_model.encode(chunks).tolist()
    existing_count = collection.count()
    ids = [f"{source}_chunk_{existing_count + i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

def get_collection_count():
    """
    Returns the number of stored chunks in ChromaDB.
    """

    return collection.count()
def clear_collection():
    """
    Remove all documents from the Chroma collection.
    """
    ids = collection.get()["ids"]

    if ids:
        collection.delete(ids=ids)