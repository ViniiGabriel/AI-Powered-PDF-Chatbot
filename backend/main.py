from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from ai_chat import query_huggingface

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    context: str

class SummaryRequest(BaseModel):
    context: str

@app.post("/ask")
async def ask_question(data: QuestionRequest, request: Request):
    model_url = request.headers.get("X-Model-URL")  # Support model switching
    prompt = f"""[INST] You are an academic assistant helping users understand complex documents.
Only use the provided context to answer the question.
If the context is not enough, respond with: "The provided context is insufficient to answer the question."
Be precise, avoid unnecessary details.

Context:
{data.context}

Question:
{data.question}
[/INST]
"""
    answer = query_huggingface(prompt, model_url=model_url)
    return {"answer": answer}


@app.post("/summarize")
async def summarize_pdf(data: SummaryRequest, request: Request):
    model_url = request.headers.get("X-Model-URL")  # Support model switching
    prompt = f"""[INST] You are a helpful assistant. Summarize the following document clearly and concisely.

Document:
{data.context}
[/INST]
"""
    summary = query_huggingface(prompt, model_url=model_url)
    return {"summary": summary}