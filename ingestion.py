import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from google import genai

load_dotenv()

CONTRACTS_DIR = Path("data/contracts")
EMBEDDINGS_DIR = Path("data/embeddings")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chroma_client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
collection = chroma_client.get_or_create_collection(name="contracts")


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page": i + 1,
                "source": pdf_path.name
            })
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    """Get embedding from Gemini."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def make_chunk_id(source: str, page: int, chunk_idx: int) -> str:
    """Create a unique, stable ID for each chunk."""
    raw = f"{source}_p{page}_c{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_contract(pdf_path: Path):
    """Full pipeline: PDF → chunks → embeddings → ChromaDB."""
    print(f"\nIngesting: {pdf_path.name}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    all_chunks, all_ids, all_embeddings, all_metadata = [], [], [], []

    for page_data in pages:
        chunks = chunk_text(page_data["text"])
        for idx, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(page_data["source"], page_data["page"], idx)

            # Skip if already indexed
            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue

            embedding = get_embedding(chunk)
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_embeddings.append(embedding)
            all_metadata.append({
                "source": page_data["source"],
                "page": page_data["page"],
                "chunk_idx": idx
            })

    if all_chunks:
        collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadata
        )
        print(f"  Indexed {len(all_chunks)} new chunks")
    else:
        print(f"  Already indexed — skipped")


def ingest_all():
    """Ingest all PDFs in the contracts directory."""
    pdfs = list(CONTRACTS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {CONTRACTS_DIR}")
        return
    print(f"Found {len(pdfs)} contracts to ingest")
    for pdf_path in sorted(pdfs):
        ingest_contract(pdf_path)
    total = collection.count()
    print(f"\nDone. Total chunks in vector store: {total}")


def test_retrieval(query: str, n_results: int = 3):
    """Test that retrieval works after ingestion."""
    print(f"\nTest query: '{query}'")
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n  Result {i+1} | {meta['source']} p.{meta['page']} | score: {1-dist:.3f}")
        print(f"  {doc[:200]}...")


def ingest_single_file(file_bytes: bytes, filename: str) -> int:
    """Ingest a PDF from raw bytes into ChromaDB. Returns number of new chunks indexed."""
    import io
    reader = PdfReader(io.BytesIO(file_bytes))

    candidate_chunks, candidate_ids, candidate_metadata = [], [], []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        page_num = i + 1
        for idx, chunk in enumerate(chunk_text(text.strip())):
            chunk_id = make_chunk_id(filename, page_num, idx)
            candidate_chunks.append(chunk)
            candidate_ids.append(chunk_id)
            candidate_metadata.append({"source": filename, "page": page_num, "chunk_idx": idx})

    if not candidate_ids:
        return 0

    existing_ids = set(collection.get(ids=candidate_ids)["ids"])
    new_indices = [i for i, cid in enumerate(candidate_ids) if cid not in existing_ids]

    if not new_indices:
        return 0

    all_chunks = [candidate_chunks[i] for i in new_indices]
    all_ids = [candidate_ids[i] for i in new_indices]
    all_metadata = [candidate_metadata[i] for i in new_indices]
    all_embeddings = [get_embedding(chunk) for chunk in all_chunks]

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        ids=all_ids,
        metadatas=all_metadata,
    )

    return len(all_chunks)


if __name__ == "__main__":
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    ingest_all()
    test_retrieval("auto-renewal clause termination notice period")
    test_retrieval("payment terms penalty interest rate")
    test_retrieval("intellectual property ownership")
