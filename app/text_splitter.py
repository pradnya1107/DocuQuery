
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split loaded documents into smaller text chunks.
    """

    # Validate chunk settings
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    # Create the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split the documents
    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    # Sample document for testing
    sample_documents = [
        Document(
            page_content=(
                "Artificial Intelligence enables machines to perform "
                "tasks that normally require human intelligence. "
                "Machine Learning is a subset of Artificial Intelligence. "
                "Deep Learning uses neural networks to learn from data. "
                "RAG combines retrieval with language generation. "
                "Embeddings represent text as numerical vectors."
            ),
            metadata={
                "source": "sample.pdf",
                "page": 1,
            },
        )
    ]

    print("Starting text splitting...")

    chunks = split_documents(
        sample_documents,
        chunk_size=150,
        chunk_overlap=30,
    )

    print(f"\nTotal chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- Chunk {index} ---")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Page: {chunk.metadata['page']}")
        print(f"Characters: {len(chunk.page_content)}")
        print(chunk.page_content)
