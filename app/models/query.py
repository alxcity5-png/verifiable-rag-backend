from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    document_name: str
    top_k: int = 3