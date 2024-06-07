from io import BytesIO
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from pypdf import PdfReader


def load_document(uploaded_file: BytesIO) -> str:
    """Load and extract text from a PDF file.

    Args:
        uploaded_file (BytesIO): The uploaded PDF file.

    Returns:
        str: The extracted raw text from the PDF.
    """
    reader = PdfReader(uploaded_file)
    raw_text = "\n".join(p.extract_text() for p in reader.pages)
    return raw_text


def load_and_split_document(uploaded_file: BytesIO) -> List[str]:
    """Load and split the document into chunks.

    Args:
        uploaded_file (BytesIO): The uploaded PDF file.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    raw_text = load_document(uploaded_file)
    chunks = text_splitter.split_text(raw_text)
    return chunks
