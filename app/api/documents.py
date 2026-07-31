from fastapi import APIRouter, UploadFile, File
import shutil
import os

from pypdf import PdfReader

from app.services.ingestion import (
    extract_text_from_pdf,
    split_text_into_chunks
)

from app.services.vector_store import (
    store_chunks,
)


router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/documents")
async def upload_document(file: UploadFile = File(...)):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = extract_text_from_pdf(file_path)

    chunks = split_text_into_chunks(extracted_text)

    stored_chunks = store_chunks(
    chunks,
    source=file.filename
)

    reader = PdfReader(file_path)
    pages = len(reader.pages)

    file_size_kb = round(
        os.path.getsize(file_path) / 1024,
        2
    )

    return {
        "message": "Document stored successfully",
        "filename": file.filename,
        "pages": pages,
        "sizeKb": file_size_kb,
        "characters_extracted": len(extracted_text),
        "chunks_created": stored_chunks
    }