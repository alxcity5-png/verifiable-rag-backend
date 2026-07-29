from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)


app.include_router(health_router)


@app.get("/")
def root():
    return {
        "message": "Verifiable RAG Backend Running"
    }