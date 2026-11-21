from pathlib import Path
from hashlib import sha256

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embeddings import get_embedding_model


BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "docuquery_documents"


def get_vector_store() -> Chroma:
    """Create or connect to the persistent ChromaDB collection."""

    embedding_model = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DB_PATH),
    )

    return vector_store


def store_documents(chunks: list[Document]) -> Chroma:
    """Store document chunks and their embeddings in ChromaDB."""

    if not chunks:
        raise ValueError("No document chunks provided.")

    vector_store = get_vector_store()

    # Generate stable IDs to prevent duplicate chunks
    # when the same documents are added again.
    document_ids = []

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)

        unique_text = f"{source}-{page}-{index}-{chunk.page_content}"

        document_id = sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        document_ids.append(document_id)

    vector_store.add_documents(
        documents=chunks,
        ids=document_ids,
    )

    return vector_store


def search_documents(
    query: str,
    k: int = 3,
) -> list[Document]:
    """Retrieve the most relevant chunks for a question."""

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("Number of results must be greater than zero.")

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k,
    )

    return results


if __name__ == "__main__":
    from app.document_loader import load_pdf
    from app.text_splitter import split_documents

    print("Loading PDF...")
    documents = load_pdf("data/sample.pdf")

    print("Splitting documents...")
    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("\nStoring chunks in ChromaDB...")
    store_documents(chunks)

    print("Documents stored successfully!")

    question = "What information is available in this document?"

    print(f"\nSearching for: {question}")

    results = search_documents(question, k=3)

    print(f"\nRetrieved {len(results)} relevant chunks.")

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(f"Source: {result.metadata.get('source')}")
        print(f"Page: {result.metadata.get('page')}")
        print(result.page_content)
