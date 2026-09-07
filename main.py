import uuid
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
from textblob import TextBlob

# Download required lightweight tokenizers
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = FastAPI(title="Neutral Grammar Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory RAM storage (auto-deletes after 2 hours)
temp_cache = TTLCache(maxsize=10000, ttl=7200)

class TextPayload(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running", "engine": "TextBlob Pure-Python"}

@app.post("/fix")
def fix_text(payload: TextPayload):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Native rule & statistical correction (zero censorship / filters)
    blob = TextBlob(raw_text)
    corrected = str(blob.correct())

    session_id = str(uuid.uuid4())
    temp_cache[session_id] = {"original": raw_text, "corrected": corrected}

    return {"session_id": session_id, "corrected": corrected}
