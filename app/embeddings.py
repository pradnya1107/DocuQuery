
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Load environment variables from .env
load_dotenv()


def get_embedding_model():
    """
    Initialize and return the Gemini embedding model.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please configure it in your .env file."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )

    return embeddings


def generate_document_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple text documents.
    """

    if not texts:
        raise ValueError("No text provided for embedding generation.")

    model = get_embedding_model()

    vectors = model.embed_documents(texts)

    return vectors


def generate_query_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single user query.
    """

    if not text.strip():
        raise ValueError("Query text cannot be empty.")

    model = get_embedding_model()

    vector = model.embed_query(text)

    return vector


if __name__ == "__main__":

    sample_texts = [
        "Artificial Intelligence enables machines to perform intelligent tasks.",
        "Machine Learning allows computers to learn from data.",
        "Deep Learning uses neural networks to learn complex patterns.",
    ]

    print("Generating document embeddings...")

    try:

        document_vectors = generate_document_embeddings(sample_texts)

        print("\nEmbeddings generated successfully!")
        print(f"Total embeddings: {len(document_vectors)}")
        print(f"Vector dimensions: {len(document_vectors[0])}")

        print("\nFirst embedding preview:")
        print(document_vectors[0][:10])

        query = "How do computers learn from information?"

        print("\nGenerating query embedding...")

        query_vector = generate_query_embedding(query)

        print("Query embedding generated successfully!")
        print(f"Query vector dimensions: {len(query_vector)}")
        print("Query vector preview:")
        print(query_vector[:10])

    except Exception as error:
        print(f"\nError generating embeddings: {error}")
