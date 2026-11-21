from langchain_core.documents import Document

from app.vector_store import get_vector_store


def get_retriever(k: int = 3):
    """Create a retriever to search relevant document chunks."""

    if k <= 0:
        raise ValueError("Number of results must be greater than zero.")

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return retriever


def retrieve_documents(query: str, k: int = 3) -> list[Document]:
    """Retrieve relevant documents for a user question."""

    if not query.strip():
        raise ValueError("Question cannot be empty.")

    retriever = get_retriever(k=k)

    results = retriever.invoke(query)

    return results


if __name__ == "__main__":
    question = "What information is available in this document?"

    print(f"Searching for: {question}")

    try:
        documents = retrieve_documents(question, k=3)

        print(f"\nRetrieved {len(documents)} chunks.")

        for index, document in enumerate(documents, start=1):
            print(f"\n--- Result {index} ---")
            print(f"Source: {document.metadata.get('source')}")
            print(f"Page: {document.metadata.get('page')}")
            print(document.page_content)

    except Exception as error:
        print(f"\nRetrieval error: {error}")
