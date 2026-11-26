"""Answer questions using the documents stored in Pinecone.

Run from the project root:
    python -m chatbot.qa "What are the main priorities of this budget?"
    python -m chatbot.qa            # interactive mode
"""
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a file (e.g. VS Code Run button): make `chatbot` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from chatbot import config
from chatbot.ingest import get_embeddings

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def get_vector_store(index_name: str = config.INDEX_NAME) -> PineconeVectorStore:
    return PineconeVectorStore(index_name=index_name, embedding=get_embeddings(),
                               pinecone_api_key=config.PINECONE_API_KEY)


def retrieve(query: str, k: int = 2, store: PineconeVectorStore | None = None):
    return (store or get_vector_store()).similarity_search(query, k=k)


def answer(query: str, k: int = 2, store: PineconeVectorStore | None = None) -> str:
    config.require_keys()
    docs = retrieve(query, k=k, store=store)
    context = "\n\n".join(d.page_content for d in docs)
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0.5, api_key=config.OPENAI_API_KEY)
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


def main() -> None:
    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:])))
        return
    store = get_vector_store()
    print("Ask a question about your documents (empty line to quit).")
    while question := input("\n> ").strip():
        print(answer(question, store=store))


if __name__ == "__main__":
    main()
