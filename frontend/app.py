import streamlit as st
import fitz  # PyMuPDF
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set Streamlit page settings
st.set_page_config(page_title="📄 Chat with Your Document", layout="wide")

# Load environment variables
load_dotenv()

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Initialize session state keys
for key in ["chunks", "chunk_embeddings", "full_text", "last_question", "last_answer", "summary"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["last_question", "last_answer", "summary"] else []

# Clear memory button
if st.button("🧹 Clear Memory"):
    for key in st.session_state.keys():
        st.session_state[key] = None if key in ["last_question", "last_answer", "summary"] else []
    st.success("🧠 Memory cleared. Ready for new uploads.")
    st.stop()

# Title and model selector
st.title("📄 Chat with Your Document")

MODEL_OPTIONS = {
    "Zephyr-7B (HuggingFaceH4/zephyr-7b-beta)": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    "Mistral-7B (mistralai/Mistral-7B-Instruct-v0.1)": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
}
selected_model = st.selectbox("🔁 Choose a model", list(MODEL_OPTIONS.keys()))
st.session_state.model_url = MODEL_OPTIONS[selected_model]

uploaded_files = st.file_uploader("📄 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Reset state if new files uploaded
    st.session_state.chunks = []
    st.session_state.chunk_embeddings = []
    st.session_state.full_text = ""

    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    for uploaded_file in uploaded_files:
        if uploaded_file.type != "application/pdf":
            st.error(f"❌ {uploaded_file.name} is not a valid PDF.")
            continue

        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_text = "".join(page.get_text() for page in doc)
        st.session_state.full_text += file_text + "\n"

        chunks = chunk_text(file_text)
        embeddings = embedder.encode(chunks)

        st.session_state.chunks.extend(chunks)
        st.session_state.chunk_embeddings.extend(embeddings)

        st.success(f"✅ Processed {uploaded_file.name} with {len(chunks)} chunks.")

# Summarization
if st.session_state.full_text and st.button("📝 Summarize the Uploaded PDFs"):
    with st.spinner("Generating summary..."):
        try:
            resp = requests.post(
                "https://ai-powered-pdf-chatbot-z6u3.onrender.com/summarize",
                json={"context": st.session_state.full_text[:2000]},
                headers={"X-Model-URL": st.session_state.model_url},
                timeout=60
            )
            if resp.status_code == 200:
                summary = resp.json().get("summary", "")
                if summary.strip() == "" or summary.strip().lower() == "summary":
                    st.warning("⚠️ The model did not return a valid summary.")
                else:
                    st.session_state.summary = summary
                    st.markdown("### 📝 Summary")
                    st.write(summary)
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            st.error(f"❌ Request failed: {str(e)}")

if st.session_state.summary:
    st.markdown("### 📝 Last Summary")
    st.info(st.session_state.summary)

# Q&A
question = st.text_input("💬 Ask a question about the PDFs")

if question and st.session_state.chunk_embeddings:
    with st.spinner("Thinking..."):
        q_embed = embedder.encode([question])
        similarities = cosine_similarity(q_embed, st.session_state.chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:4]
        context = "\n".join([st.session_state.chunks[i] for i in top_indices])

        try:
            resp = requests.post(
                "https://ai-powered-pdf-chatbot-z6u3.onrender.com/ask",
                json={"question": question, "context": context},
                headers={"X-Model-URL": st.session_state.model_url},
                timeout=60
            )
            if resp.status_code == 200:
                answer = resp.json().get("answer", "")
                st.session_state.last_question = question
                st.session_state.last_answer = answer
                st.markdown(f"**📘 Answer:** {answer}")
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            st.error(f"❌ Request failed: {str(e)}")

# Display last Q&A
if st.session_state.last_question and st.session_state.last_answer:
    st.markdown("### 🧠 Previous Q&A")
    st.markdown(f"**Q:** {st.session_state.last_question}")
    st.markdown(f"**A:** {st.session_state.last_answer}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;'>Created with ❤️ by Gokularaman C</div>", unsafe_allow_html=True)