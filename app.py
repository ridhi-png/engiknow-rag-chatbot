"""
EngiKnow — RAG chatbot over engineering "lessons learned" docs.
Run: streamlit run app.py
Requires env var GROQ_API_KEY (free tier: https://console.groq.com/keys)
"""
import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

st.set_page_config(page_title="EngiKnow", page_icon="🔧", layout="wide")

@st.cache_resource
def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("engiknow")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return embed_model, collection, groq_client

embed_model, collection, groq_client = load_resources()

st.title("🔧 EngiKnow")
st.caption("Ask questions about engineering equipment failures, RCA reports, and maintenance lessons learned.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

with st.sidebar:
    st.header("Retrieved sources")
    st.caption("Source snippets used for the last answer will appear here.")
    sources_placeholder = st.empty()

query = st.chat_input("e.g. Why do our conveyor motor bearings keep failing?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)
    retrieved_chunks = results["documents"][0]
    retrieved_sources = [m["source"] for m in results["metadatas"][0]]

    context = "\n\n---\n\n".join(
        f"[Source: {src}]\n{chunk}" for src, chunk in zip(retrieved_sources, retrieved_chunks)
    )

    prompt = f"""You are EngiKnow, an engineering knowledge assistant. Answer the question using ONLY the context below.
Always cite the source file(s) you used in your answer, like [source: filename.txt].
If the context does not contain the answer, say so clearly instead of guessing.

Context:
{context}

Question: {query}

Answer:"""

    with st.chat_message("assistant"):
        if not os.environ.get("GROQ_API_KEY"):
            answer = "No GROQ_API_KEY set. Add it as an environment variable to enable answers. (Retrieval still works, see sidebar.)"
        else:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with sources_placeholder.container():
        for src, chunk in zip(retrieved_sources, retrieved_chunks):
            with st.expander(f"Source: {src}"):
                st.write(chunk)
