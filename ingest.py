"""Load resume/linkedin/summary into a persisted Chroma vector store."""
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "digital_twin"

SOURCES = [
    (DATA_DIR / "resume.pdf", PyPDFLoader),
    (DATA_DIR / "linkedin.pdf", PyPDFLoader),
    (DATA_DIR / "summary.txt", TextLoader),
]


def main() -> None:
    docs = []
    for path, loader_cls in SOURCES:
        docs.extend(loader_cls(str(path)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Ingested {len(chunks)} chunks from {len(SOURCES)} files into {PERSIST_DIR}")


if __name__ == "__main__":
    main()
