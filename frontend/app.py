import streamlit as st
import fitz  # PyMuPDF
import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Streamlit page config
st.set_page_config(page_title="📄 Chat with Your Document", layout="wide")

# Load environment variables
load_dotenv()

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Model selection
st.title("📄 Chat with Your Document")
MODEL_OPTIONS = {
    "Zephyr-7B (HuggingFaceH4/zephyr-7b-beta)": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    "Mistral-7B (mistralai/Mistral-7B-Instruct-v0.1)": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
}
selected_model = st.selectbox("🔁 Choose a model", list(MODEL_OPTIONS.keys()))
model_url = MODEL_OPTIONS[selected_model]

# Upload PDFs
uploaded_files = st.file_uploader("📄 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

chunks = []
chunk_embeddings = []
all_text = ""

if uploaded_files:
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
        all_text += file_text + "\n"

        file_chunks = chunk_text(file_text)
        file_embeddings = embedder.encode(file_chunks)

        chunks.extend(file_chunks)
        chunk_embeddings.extend(file_embeddings)

        st.success(f"✅ Processed {uploaded_file.name} with {len(file_chunks)} chunks.")

    if st.button("📝 Summarize the Uploaded PDFs"):
        with st.spinner("Generating summary..."):
            try:
                resp = requests.post(
                    "https://ai-powered-pdf-chatbot-z6u3.onrender.com/summarize",
                    json={"context": all_text[:2000]},
                    headers={"X-Model-URL": model_url},
                    timeout=60
                )
                if resp.status_code == 200:
                    summary = resp.json().get("summary", "")
                    if summary.strip() == "" or summary.strip().lower() == "summary":
                        st.warning("⚠️ The model did not return a valid summary.")
                    else:
                        st.markdown("### 📝 Summary")
                        st.write(summary)
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")

    question = st.text_input("💬 Ask a question about the PDFs")
    if question and chunk_embeddings:
        with st.spinner("Thinking..."):
            q_embed = embedder.encode([question])
            similarities = cosine_similarity(q_embed, chunk_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:4]
            context = "\n".join([chunks[i] for i in top_indices])

            try:
                resp = requests.post(
                    "https://ai-powered-pdf-chatbot-z6u3.onrender.com/ask",
                    json={"question": question, "context": context},
                    headers={"X-Model-URL": model_url},
                    timeout=60
                )
                if resp.status_code == 200:
                    answer = resp.json().get("answer", "")
                    st.markdown(f"**📘 Answer:** {answer}")
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;'>Created with ❤️ by Gokularaman C</div>", unsafe_allow_html=True)