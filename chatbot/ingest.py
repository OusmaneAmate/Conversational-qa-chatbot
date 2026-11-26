"""Load PDFs from documents/, split them into chunks and store them in Pinecone.

Run from the project root:  python -m chatbot.ingest
Chunk IDs are deterministic, so re-running updates vectors instead of duplicating them.
"""
import hashlib
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file (e.g. VS Code Run button): make `chatbot` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from chatbot import config


def load_documents(directory: Path = config.DOCUMENTS_DIR):
    return PyPDFDirectoryLoader(str(directory)).load()


def split_documents(documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def get_embeddings():
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)


def ensure_index(index_name: str = config.INDEX_NAME) -> None:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def _chunk_id(chunk, position: int) -> str:
    key = f"{Path(chunk.metadata.get('source', '')).name}|{chunk.metadata.get('page')}|{position}"
    return hashlib.sha1(key.encode()).hexdigest()


def ingest(index_name: str = config.INDEX_NAME) -> int:
    config.require_keys()
    chunks = split_documents(load_documents())
    ensure_index(index_name)
    store = PineconeVectorStore(index_name=index_name, embedding=get_embeddings(),
                                pinecone_api_key=config.PINECONE_API_KEY)
    store.add_documents(chunks, ids=[_chunk_id(c, i) for i, c in enumerate(chunks)])
    return len(chunks)


if __name__ == "__main__":
    print(f"Stored {ingest()} chunks in Pinecone index '{config.INDEX_NAME}'.")
