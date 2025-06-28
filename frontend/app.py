import streamlit as st
import fitz  # PyMuPDF
import os
import requests
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Set Streamlit page settings
st.set_page_config(page_title="📄 Chat with Your Document", layout="wide")

# Load environment variables
load_dotenv()

# Load sentence transformer model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Set up ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# Title and model selector
st.title("📄 Chat with Your Document")

MODEL_OPTIONS = {
    "Zephyr-7B (HuggingFaceH4/zephyr-7b-beta)": "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    "Mistral-7B (mistralai/Mistral-7B-Instruct-v0.1)": "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
}
selected_model = st.selectbox("🔁 Choose a model", list(MODEL_OPTIONS.keys()))
st.session_state.model_url = MODEL_OPTIONS[selected_model]

# Upload multiple PDFs
uploaded_files = st.file_uploader("📄 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    all_chunks = []
    all_ids = []
    all_metadata = []

    try:
        collection.delete(where={"source": {"$eq": "pdf"}})
    except Exception as e:
        st.warning(f"⚠️ Could not delete old chunks: {str(e)}")

    def chunk_text(text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    for pdf_index, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.type != "application/pdf":
            st.error(f"❌ {uploaded_file.name} is not a valid PDF.")
            continue

        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_text = "".join(page.get_text() for page in doc)
        all_text += file_text + "\n"

        chunks = chunk_text(file_text)
        all_chunks.extend(chunks)
        all_ids.extend([f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))])
        all_metadata.extend([{"source": "pdf", "filename": uploaded_file.name}] * len(chunks))

        st.success(f"✅ Processed {uploaded_file.name} with {len(chunks)} chunks.")

    # Embed and store chunks
    embeddings = embedder.encode(all_chunks).tolist()
    collection.add(documents=all_chunks, metadatas=all_metadata, ids=all_ids, embeddings=embeddings)

    # Summarize
    if st.button("📝 Summarize the Uploaded PDFs"):
        with st.spinner("Generating summary..."):
            try:
                resp = requests.post(
                    "http://localhost:8000/summarize",
                    json={"context": all_text[:2000]},
                    headers={"X-Model-URL": st.session_state.model_url},
                    timeout=60
                )
                if resp.status_code == 200:
                    summary = resp.json().get("summary", "")
                    if summary.strip() == "" or summary.strip().lower() == "summary":
                        st.warning("⚠️ The model did not return a valid summary. Try rephrasing or re-uploading.")
                    else:
                        st.markdown("### 📝 Summary")
                        st.write(summary)
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")

    # Ask a question
    question = st.text_input("💬 Ask a question about the PDFs")

    if question:
        with st.spinner("Thinking..."):
            q_embed = embedder.encode([question])[0].tolist()
            results = collection.query(query_embeddings=[q_embed], n_results=4)
            context = "\n".join(results["documents"][0])

            try:
                resp = requests.post(
                    "http://localhost:8000/ask",
                    json={"question": question, "context": context},
                    headers={"X-Model-URL": st.session_state.model_url},
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