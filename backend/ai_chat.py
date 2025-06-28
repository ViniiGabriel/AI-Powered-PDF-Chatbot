import os
import requests
from dotenv import load_dotenv

# Load env variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Dynamic header generator
def get_headers():
    return {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface(prompt: str, model_url: str) -> str:
    payload = {"inputs": prompt}
    try:
        response = requests.post(model_url, headers=get_headers(), json=payload, timeout=60)
        if response.headers.get("content-type") != "application/json":
            return f"❌ Non-JSON response: {response.text}"

        result = response.json()
        raw = result[0]["generated_text"]
        answer = raw.split("[/INST]")[-1].strip()
        return answer.replace("Answer:", "").strip()

    except Exception as e:
        return f"❌ Request failed: {str(e)}"