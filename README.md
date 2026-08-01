
# EngiKnow — RAG Chatbot over Engineering Docs

A retrieval-augmented generation (RAG) chatbot that answers questions over a knowledge base of
engineering RCA reports, maintenance logs, and lessons-learned entries — with source citations.

## Problem to Approach

Engineering teams accumulate valuable failure/RCA knowledge in scattered documents that's hard
to search. EngiKnow embeds those documents into a vector store and uses retrieval plus an LLM to
answer natural-language questions with citations back to the source doc, so answers are
verifiable rather than hallucinated.

## Architecture

```
 data/*.txt  ->  chunk (700 chars, 100 overlap)  ->  MiniLM-L6-v2 embeddings
              ->  ChromaDB (local vector store)
              ->  query embedding -> top-3 similarity search
              ->  context + question -> Groq Llama-3.1-8B -> cited answer
              ->  Streamlit chat UI (answer + sidebar source snippets)
```

## Dataset

35 synthetic engineering knowledge-base entries (`data/*.txt`) across categories: conveyor
systems, hydraulics, CNC machining, robotics, PLC/controls, pneumatics, cooling systems,
bearings/gearboxes, welding, EV battery/motor systems, and quality/assembly. Each is a realistic
RCA/maintenance-log style entry with a title, category, date, and root-cause narrative.
`data/_eval_queries.json` has 15 test questions with the expected source file, used for the
retrieval accuracy check below.

## How to run

```bash
pip install -r requirements.txt

# 1. Build the vector store (run once, or whenever data/ changes)
python src/ingest.py

# 2. Check retrieval accuracy (top-3 hit rate against 15 test queries)
python src/eval.py

# 3. Get a free Groq API key: https://console.groq.com/keys
export GROQ_API_KEY=your_key_here  # Windows: set GROQ_API_KEY=your_key_here
# 4. Launch the chat app
streamlit run app.py
```

## Retrieval evaluation

`src/eval.py` runs 15 held-out test questions and checks whether the correct source document
appears in the top-3 retrieved chunks, a simple and transparent proxy for retrieval quality
that's easy to explain in an interview.

## Deploy

Push to GitHub, then deploy free on Streamlit Community Cloud, point it at `app.py`, add
`GROQ_API_KEY` as a secret, done.

## Possible extensions

- Swap Groq for an on-prem/self-hosted model for data-sensitive deployments
- Add metadata filtering (search by category or date range)
- Add re-ranking (cross-encoder) on top of the initial vector search for higher precision
=======
# engiknow-rag-chatbot
df9552907eb030aa731863bbb694da71bf6e6470

echo chroma_db/ > .gitignore