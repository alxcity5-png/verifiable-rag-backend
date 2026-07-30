from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME = "Verifiable RAG Engine"
    VERSION = "1.0.0"

    # Paths
    UPLOAD_DIR = "uploads"
    CHROMA_DB_PATH = "chroma_db"

    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"


settings = Settings()

# Convenience constants
UPLOAD_DIR = settings.UPLOAD_DIR
CHROMA_DB_PATH = settings.CHROMA_DB_PATH
EMBEDDING_MODEL = settings.EMBEDDING_MODEL