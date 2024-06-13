import os
import PyPDF2.errors
import pdfplumber
from typing import Callable, List, Tuple, Dict
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import re
from langchain.embeddings import HuggingFaceBgeEmbeddings
import PyPDF2

model_name = "BAAI/bge-base-en"
encode_kwargs = {"normalize_embeddings": True}  # set True to compute cosine similarity
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, encode_kwargs=encode_kwargs
)

# Specify the path to the root directory
root_directory = "D:/personal/projects/legal_assistant_chatbot/Union acts"

database = []


def extract_metadata_from_pdf(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)  # Change this line
            docmetadata = reader.metadata
            return {"title": str(docmetadata.title)}
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file: {e}")


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    """
    Extracts the text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A list of tuples containing the page number and the extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdfplumber.open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Check if extracted text is not empty
                pages.append((page_num + 1, text))
    return pages


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    """
    Extracts the title and text from each page of the PDF.

    :param file_path: The path to the PDF file.
    :return: A tuple containing the title and a list of tuples with page numbers and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(
    pages: List[Tuple[int, str]], cleaning_functions: List[Callable[[str], str]]
) -> List[Tuple[int, str]]:
    cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_num, text))
    return cleaned_pages


def text_to_docs(text, metadata: Dict[str, str]) -> List[Document]:
    """Converts list of strings to a list of Documents with metadata."""
    doc_chunks = []
    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


for root, directories, files in os.walk(root_directory):
    for filename in files:
        # Construct the full path to the current file
        file_path = os.path.join(root, filename[0:])
        file_path = file_path.replace("\\", "/")
        print(file_path)

        # Step 1: Parse PDF

        raw_pages, metadata = parse_pdf(file_path)

        # Step 2: Create text chunks
        cleaning_functions = [
            merge_hyphenated_words,
            fix_newlines,
            remove_multiple_newlines,
        ]
        cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
        document_chunks = text_to_docs(cleaned_text_pdf, metadata)
        database.extend(document_chunks)

        # Optional: Reduce embedding cost by only using the first 23 pages
        if len(database) > 2000:

            vector_store = Chroma.from_documents(
                database,
                embeddings,
                persist_directory="D:/personal/projects/legal_assistant_chatbot/vectordb2",
            )
            vector_store.persist()

            database = []
