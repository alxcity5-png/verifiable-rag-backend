from fastapi import APIRouter

from app.models.query import QueryRequest
from app.services.rag_pipeline import answer_question

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/")
def query_documents(request: QueryRequest):
    result = answer_question(
        question=request.question,
        top_k=request.top_k,
    )

    return result