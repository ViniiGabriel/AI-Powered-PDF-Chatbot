# 🤖 AI-Powered PDF Chatbot

Chat with your PDF documents using powerful LLMs like **Zephyr** or **Mistral** via Hugging Face APIs.  
Upload any PDF, get a summary, and ask questions with accurate context-aware responses.

[🌐 Try the App](https://ai-powered-pdf-chatbot.streamlit.app)

---

## ✨ Features

- 📄 Upload multiple PDF documents  
- 🧠 Summarize large PDFs instantly  
- 💬 Ask questions based on document content  
- 🔁 Retry on API failure for better stability  
- 🔄 Re-upload PDFs anytime using the button  
- 🧠 Contextual memory (in-memory embeddings)  
- 🌐 Powered by Hugging Face Inference API  

---

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: FastAPI (deployed on Render)  
- **Embedding Model**: `all-MiniLM-L6-v2` (via `sentence-transformers`)  
- **Inference Models**: Zephyr-7B, Mistral-7B (via Hugging Face)  
- **Vector Store**: In-memory using cosine similarity (`sklearn`)  

---

## 🛠️ Local Setup

Want to run this project on your own machine? Just follow these steps:

---

### ✅ 1. Clone this repo

```bash
git clone https://github.com/gokularaman-c/AI-Powered-PDF-Chatbot.git
cd AI-Powered-PDF-Chatbot
```

---

### ✅ 2. Install dependencies

Make sure you have Python 3.10+ and pip installed. Then run:

```bash
# For frontend
cd frontend
pip install -r requirements.txt

# For backend (in a separate terminal/tab)
cd ../backend
pip install -r requirements.txt
```

---

### ✅ 3. Add your Hugging Face token

Create a `.env` file in both `frontend/` and `backend/` folders with:

```env
HF_TOKEN=your_huggingface_token
```

🔐 Replace `your_huggingface_token` with your actual token from:  
👉 https://huggingface.co/settings/tokens

---

### ✅ 4. Run the app locally

Use two terminals or terminal tabs:

**Terminal 1 – Run Backend**

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 – Run Frontend**

```bash
cd frontend
streamlit run app.py
```

Then open your browser and go to:  
👉 http://localhost:8501

---

## 🚀 Deployment

- **Frontend**: Deployed using Streamlit Cloud  
  👉 https://ai-powered-pdf-chatbot.streamlit.app  
- **Backend**: Deployed using Render  
  👉 https://ai-powered-pdf-chatbot-z6u3.onrender.com

---

## 📝 License

MIT License  
© 2025 Gokularaman C