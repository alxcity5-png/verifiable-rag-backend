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


def store_chunks(chunks: list[str]):
    """
    Generate embeddings and store chunks in ChromaDB.
    """

    embeddings = embedding_model.encode(chunks).tolist()

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return len(chunks)


def get_collection_count():
    """
    Returns the number of stored chunks in ChromaDB.
    """

    return collection.count()