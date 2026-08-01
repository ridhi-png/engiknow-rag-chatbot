# 🔧 EngiKnow — RAG Chatbot over Engineering Docs

**🚀 Live demo: [engiknow-rag-chatbot-6erlna5nzgdfgwr6nhfxmo.streamlit.app](https://engiknow-rag-chatbot-6erlna5nzgdfgwr6nhfxmo.streamlit.app/)**

A retrieval-augmented generation (RAG) chatbot that answers natural-language questions over a
knowledge base of engineering RCA reports, maintenance logs, and lessons-learned entries — with
every answer backed by a cited source document.

> Built for the "AI Chatbot Development for Engineering Knowledge Management" problem statement:
> new engineers spend significant time hunting for past project information, lessons learned, and
> technical documents. EngiKnow lets them just ask.

## Demo

Ask it things like:
- *"Why do our conveyor motor bearings keep failing?"*
- *"What caused the hydraulic press to leak?"*
- *"Why is the CNC spindle overheating?"*

Each answer comes with a `[source: filename.txt]` citation, and the sidebar shows the exact
retrieved snippets — so answers are verifiable, not just plausible-sounding.

## Architecture

```
 data/*.txt  ->  chunk (700 chars, 100 overlap)  ->  MiniLM-L6-v2 embeddings
              ->  ChromaDB (local vector store)
              ->  query embedding -> top-3 similarity search
              ->  context + question -> Groq Llama-3.1-8B -> cited answer
              ->  Streamlit chat UI (answer + sidebar source snippets)
```

| Component | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Fast, free, runs locally, no API key needed |
| Vector store | ChromaDB | Zero-config, file-based, no server to manage |
| Generation | Llama 3.1 8B via Groq | Free tier, very low latency |
| UI | Streamlit | Fast to build, easy to deploy |

The architecture is model-agnostic — the generation step can be swapped for an on-prem or
self-hosted model in a deployment where data privacy matters more than latency.

## Dataset

35 synthetic engineering knowledge-base entries (`data/*.txt`) spanning conveyor systems,
hydraulics, CNC machining, robotics, PLC/controls, pneumatics, cooling systems,
bearings/gearboxes, welding, EV battery/motor systems, and quality/assembly. Each entry reads like
a real RCA/maintenance-log record: a title, category, date, and root-cause narrative.
`data/_eval_queries.json` holds 15 test questions with their expected source file, used for the
retrieval accuracy check below.

## How to run locally

```bash
pip install -r requirements.txt

# 1. Build the vector store (run once, or whenever data/ changes)
python src/ingest.py

# 2. Check retrieval accuracy (top-3 hit rate against 15 test queries)
python src/eval.py

# 3. Get a free Groq API key: https://console.groq.com/keys
export GROQ_API_KEY=your_key_here     # Windows: set GROQ_API_KEY=your_key_here

# 4. Launch the chat app
streamlit run app.py
```

## Retrieval evaluation

`src/eval.py` runs 15 held-out test questions and checks whether the correct source document
appears in the top-3 retrieved chunks — a simple, transparent proxy for retrieval quality that's
easy to explain and defend in an interview or presentation.

## Deployed with

Streamlit Community Cloud (free tier), pulling directly from this repo's `main` branch. Secrets
(`GROQ_API_KEY`) are managed via Streamlit Cloud's secrets manager, never committed to the repo.

## Possible extensions

- Swap Groq for an on-prem/self-hosted model for data-sensitive enterprise deployments
- Add metadata filtering (search by category or date range)
- Add re-ranking (cross-encoder) on top of the initial vector search for higher precision
- Add a feedback mechanism (thumbs up/down per answer) to track real-world accuracy over time

---

Built by [Ridhi Arora](https://github.com/ridhi-png) — [LinkedIn](https://linkedin.com/in/ridhi-arora)
