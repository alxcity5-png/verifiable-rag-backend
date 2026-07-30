from pypdf import PdfReader
import re

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap_sentences: int = 1
):
    """
    Split text into chunks along sentence boundaries so a fact is
    never cut mid-sentence. Chunks target ~chunk_size characters but
    always end on a full sentence; the last `overlap_sentences`
    sentences of each chunk are repeated at the start of the next
    chunk for context continuity.
    """

    # Basic sentence splitter: splits after ., !, or ? followed by
    # whitespace, but avoids breaking on common abbreviations like
    # "Rs." or single-letter initials.
    sentence_pattern = r'(?<!\bRs)(?<!\bNo)(?<=[.!?])\s+(?=[A-Z(])'
    sentences = re.split(sentence_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        current.append(sentence)
        current_len += len(sentence) + 1

        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            # carry the last N sentences forward for overlap
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s) + 1 for s in current)

    if current:
        chunks.append(" ".join(current))

    return chunks
    