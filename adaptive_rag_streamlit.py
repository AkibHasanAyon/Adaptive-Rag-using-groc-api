import os
import fitz  # PyMuPDF
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env file")

# Initialize clients
client = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ========== Core Functions ==========
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text("text") for page in doc])

def chunk_text(text, chunk_size=1000, overlap=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add_item(self, text, embedding):
        self.texts.append(text)
        self.vectors.append(np.array(embedding))

    def similarity_search(self, query_embedding, k=5):
        query_vec = np.array(query_embedding)
        sims = [
            (i, np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))
            for i, vec in enumerate(self.vectors)
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [self.texts[i] for i, _ in sims[:k]]

def get_embeddings(text_list):
    return embedding_model.encode(text_list, convert_to_numpy=True, batch_size=16, show_progress_bar=False)

def groq_chat(system_prompt, user_prompt, model="llama3-70b-8192"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def process_document(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    store = SimpleVectorStore()
    for chunk, vec in zip(chunks, embeddings):
        store.add_item(chunk, vec)
    return chunks, store

def classify_query(query):
    system_prompt = """You are an expert at query classification.
Classify the input into one of the following categories:
- Factual
- Analytical
- Opinion
- Contextual
Return ONLY the category name with no explanation."""
    return groq_chat(system_prompt, query)

def factual_retrieval_strategy(query, store, k=4):
    system_prompt = "You are an expert at refining factual search queries."
    enhanced_query = groq_chat(system_prompt, query)
    emb = embedding_model.encode(enhanced_query, convert_to_numpy=True)
    return store.similarity_search(emb, k=k)

def analytical_retrieval_strategy(query, store, k=4):
    system_prompt = """You are an expert at breaking down complex analytical queries.
Generate exactly 3 sub-questions that explore different dimensions of the main query.
Return each sub-question on a new line without numbering or explanation."""
    sub_questions = groq_chat(system_prompt, query).splitlines()
    results = []
    seen = set()
    for sq in sub_questions:
        emb = embedding_model.encode(sq, convert_to_numpy=True)
        for chunk in store.similarity_search(emb, k=2):
            if chunk not in seen:
                results.append(chunk)
                seen.add(chunk)
    return results[:k]

def opinion_retrieval_strategy(query, store, k=4):
    system_prompt = """You are an expert at identifying diverse viewpoints.
Generate exactly 3 different perspectives for the query. Return one per line."""
    perspectives = groq_chat(system_prompt, query).splitlines()
    results = []
    seen = set()
    for p in perspectives:
        q = f"{query} | {p}"
        emb = embedding_model.encode(q, convert_to_numpy=True)
        for chunk in store.similarity_search(emb, k=2):
            if chunk not in seen:
                results.append(chunk)
                seen.add(chunk)
    return results[:k]

def contextual_retrieval_strategy(query, store, k=4):
    ctx_prompt = """You are an expert at identifying implied context.
What context is likely needed to answer the query below? Return 1-2 lines."""
    context = groq_chat(ctx_prompt, query)
    rewrite_prompt = """You are an expert at rewriting queries with context.
Query: {query}
Context: {context}
Rewrite the query to include the context."""
    revised_query = groq_chat(rewrite_prompt, f"Query: {query}\nContext: {context}")
    emb = embedding_model.encode(revised_query, convert_to_numpy=True)
    return store.similarity_search(emb, k=k)

def adaptive_retrieval(query, store, k=4):
    query_type = classify_query(query)
    if query_type.lower() == "factual":
        return query_type, factual_retrieval_strategy(query, store, k)
    elif query_type.lower() == "analytical":
        return query_type, analytical_retrieval_strategy(query, store, k)
    elif query_type.lower() == "opinion":
        return query_type, opinion_retrieval_strategy(query, store, k)
    elif query_type.lower() == "contextual":
        return query_type, contextual_retrieval_strategy(query, store, k)
    else:
        return query_type, factual_retrieval_strategy(query, store, k)

def generate_response(query, chunks, query_type):
    context = "\n\n---\n\n".join(chunks)
    prompt_map = {
        "Factual": "Provide an accurate, fact-based response using only the provided context.",
        "Analytical": "Provide a detailed, multi-angle analysis using the context.",
        "Opinion": "Summarize diverse viewpoints from the context.",
        "Contextual": "Use context-aware understanding to answer the query."
    }
    system_prompt = prompt_map.get(query_type, prompt_map["Factual"])
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return groq_chat(system_prompt, user_prompt)

# ========== Streamlit UI ==========
st.title("📄 Adaptive RAG Chatbot with Groq")

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
user_query = st.text_area("Enter your question")

if st.button("Get Answer") and pdf_file and user_query:
    with st.spinner("Processing document and retrieving answer..."):
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())
        chunks, store = process_document("temp.pdf")
        q_type, top_chunks = adaptive_retrieval(user_query, store)
        answer = generate_response(user_query, top_chunks, q_type)

    st.markdown(f"**Query Type:** {q_type}")
    st.markdown(f"### 💬 Answer\n{answer}")
