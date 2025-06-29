import streamlit as st
import fitz  # PyMuPDF
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page settings
st.set_page_config(page_title="📄 Chat with Your Document", layout="wide")
load_dotenv()

# Load model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Init session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.chunk_embeddings = []
    st.session_state.all_text = ""
    st.session_state.uploaded = False
    st.session_state.retry_context = None
    st.session_state.retry_question = None

# Title & model
st.title("📄 Chat with Your Document")
MODEL_OPTIONS = {
    "Zephyr-7B": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    "Mistral-7B": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
}
selected_model = st.selectbox("🔁 Choose a model", list(MODEL_OPTIONS.keys()))
st.session_state.model_url = MODEL_OPTIONS[selected_model]

# Clear and re-upload
if st.button("🔄 Re-upload PDFs"):
    st.session_state.chunks = []
    st.session_state.chunk_embeddings = []
    st.session_state.all_text = ""
    st.session_state.uploaded = False
    st.experimental_rerun()

# Upload PDFs
if not st.session_state.uploaded:
    uploaded_files = st.file_uploader("📄 Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        all_text = ""
        chunks_all = []
        embeddings_all = []

        def chunk_text(text, chunk_size=500, overlap=50):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                start += chunk_size - overlap
            return chunks

        for file in uploaded_files:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            file_text = "".join(page.get_text() for page in doc)
            all_text += file_text + "\n"
            chunks = chunk_text(file_text)
            embeddings = embedder.encode(chunks)

            chunks_all.extend(chunks)
            embeddings_all.extend(embeddings)

        st.session_state.chunks = chunks_all
        st.session_state.chunk_embeddings = embeddings_all
        st.session_state.all_text = all_text
        st.session_state.uploaded = True
        st.success("✅ PDFs processed successfully.")

# Summarize
if st.session_state.uploaded and st.button("📝 Summarize"):
    with st.spinner("Summarizing..."):
        try:
            resp = requests.post(
                "https://ai-powered-pdf-chatbot-z6u3.onrender.com/summarize",
                json={"context": st.session_state.all_text[:2000]},
                headers={"X-Model-URL": st.session_state.model_url},
                timeout=60
            )
            if resp.status_code == 200:
                summary = resp.json().get("summary", "")
                if summary.strip() and summary.strip().lower() != "summary":
                    st.markdown("### 📝 Summary")
                    st.write(summary)
                else:
                    st.warning("⚠️ The model returned an empty or invalid summary.")
            else:
                st.error(f"API Error: {resp.text}")
        except Exception as e:
            st.error(f"❌ Request failed: {str(e)}")

# Ask Question
question = st.text_input("💬 Ask a question")

if question and st.session_state.chunk_embeddings:
    with st.spinner("Thinking..."):
        try:
            q_embed = embedder.encode([question])
            similarities = cosine_similarity(q_embed, st.session_state.chunk_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:4]
            context = "\n".join([st.session_state.chunks[i] for i in top_indices])

            resp = requests.post(
                "https://ai-powered-pdf-chatbot-z6u3.onrender.com/ask",
                json={"question": question, "context": context},
                headers={"X-Model-URL": st.session_state.model_url},
                timeout=60
            )

            if resp.status_code == 200:
                answer = resp.json().get("answer", "")
                st.markdown(f"**📘 Answer:** {answer}")
                st.session_state.retry_context = None
                st.session_state.retry_question = None
            else:
                raise Exception(resp.text)

        except Exception as e:
            st.error(f"❌ Request failed: {str(e)}")
            st.session_state.retry_context = context
            st.session_state.retry_question = question

# Retry on failure
if st.session_state.retry_question:
    if st.button("🔁 Retry Last Question"):
        with st.spinner("Retrying..."):
            try:
                resp = requests.post(
                    "https://ai-powered-pdf-chatbot-z6u3.onrender.com/ask",
                    json={"question": st.session_state.retry_question, "context": st.session_state.retry_context},
                    headers={"X-Model-URL": st.session_state.model_url},
                    timeout=60
                )
                if resp.status_code == 200:
                    answer = resp.json().get("answer", "")
                    st.markdown(f"**📘 Answer:** {answer}")
                    st.session_state.retry_context = None
                    st.session_state.retry_question = None
                else:
                    st.error(f"Retry API Error: {resp.text}")
            except Exception as e:
                st.error(f"❌ Retry failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;'>Created with ❤️ by Gokularaman C</div>", unsafe_allow_html=True)