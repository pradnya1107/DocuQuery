
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Default PDF file location
PDF_PATH = BASE_DIR / "data" / "sample.pdf"


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a PDF file and convert each page into a LangChain Document.
    """

    pdf_path = Path(file_path)

    # Check whether the PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Check the file extension
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Please provide a valid PDF file.")

    # Read the PDF
    reader = PdfReader(str(pdf_path))

    documents = []

    # Extract text from each page
    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        # Create a LangChain Document
        document = Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "page": page_number
            }
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    print("Loading PDF document...")

    try:
        docs = load_pdf(PDF_PATH)

        print("\nPDF loaded successfully!")
        print(f"Total pages: {len(docs)}")

        for doc in docs:
            print(f"\nPage: {doc.metadata['page']}")
            print(f"Source: {doc.metadata['source']}")
            print("Extracted text:")
            print(doc.page_content[:300])

    except Exception as error:
        print(f"\nError loading PDF: {error}")
