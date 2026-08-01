"""
Retrieval accuracy check: does the right source doc show up in top-3 for each test query?
Run: python src/eval.py
"""
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("engiknow")

    with open(os.path.join(DATA_DIR, "_eval_queries.json")) as f:
        queries = json.load(f)

    hits = 0
    print(f"{'Query':<60} {'Top-3 hit?':<10}")
    print("-" * 75)
    for q in queries:
        emb = model.encode([q["query"]]).tolist()
        res = collection.query(query_embeddings=emb, n_results=3)
        sources = [m["source"] for m in res["metadatas"][0]]
        hit = q["expected_file"] in sources
        hits += hit
        print(f"{q['query'][:58]:<60} {'YES' if hit else 'NO':<10}")

    accuracy = hits / len(queries) * 100
    print("-" * 75)
    print(f"Top-3 retrieval accuracy: {hits}/{len(queries)} ({accuracy:.1f}%)")

if __name__ == "__main__":
    main()
