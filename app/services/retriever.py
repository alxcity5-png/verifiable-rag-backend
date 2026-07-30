import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DB_PATH


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)


collection = client.get_collection(
    name="documents"
)


def retrieve_chunks(question: str, top_k: int = 3):
    """
    Retrieve relevant document chunks with metadata and scores.
    """

    # Create embedding for question
    question_embedding = model.encode(
        question
    ).tolist()


    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )


    retrieved_results = []


    for i in range(len(results["documents"][0])):
        retrieved_results.append(
            {
                "chunk": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
        )


    return retrieved_results