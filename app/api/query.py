from fastapi import APIRouter

from app.models.query import QueryRequest
from app.services.retriever import retrieve_chunks

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/")
def query_documents(request: QueryRequest):
    chunks = retrieve_chunks(
        question=request.question,
        top_k=request.top_k,
    )

    return {
        "question": request.question,
        "retrieved_chunks": chunks,
    }