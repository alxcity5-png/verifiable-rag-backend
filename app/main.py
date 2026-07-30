from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)


@app.get("/")
def root():
    return {
        "message": "Verifiable RAG Backend Running"
    }