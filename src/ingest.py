"""
EngiKnow ingestion: read /data/*.txt, chunk, embed, store in local ChromaDB.
Run once: python src/ingest.py
"""
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def chunk_text(text, max_chars=700, overlap=100):
    """Simple char-based chunking (fast + good enough for short docs)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    # fresh collection each run so re-running ingest is idempotent
    try:
        client.delete_collection("engiknow")
    except Exception:
        pass
    collection = client.create_collection("engiknow")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    print(f"Found {len(files)} docs")

    ids, docs, metadatas = [], [], []
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath, "r") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{fname}::{i}")
            docs.append(chunk)
            metadatas.append({"source": fname})

    print(f"Embedding {len(docs)} chunks...")
    embeddings = model.encode(docs, show_progress_bar=True).tolist()

    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    print(f"Stored {len(docs)} chunks in ChromaDB at {DB_DIR}")

if __name__ == "__main__":
    main()
