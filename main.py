import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
from happytransformer import HappyTextToText, TTSettings

app = FastAPI(title="Neutral Grammar & Paraphrase Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a lightweight T5 grammar correction transformer (Pure Python / PyTorch CPU)
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
settings = TTSettings(num_beams=2, min_length=1)

# In-memory RAM storage (auto-deletes after 2 hours)
temp_cache = TTLCache(maxsize=10000, ttl=7200)

class TextPayload(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running", "engine": "T5 Grammar Transformer"}

@app.post("/fix")
def fix_text(payload: TextPayload):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty text")

    # T5 Grammar prefix format
    prompt = f"grammar: {raw_text}"
    result = happy_tt.generate_text(prompt, args=settings)
    corrected = result.text.strip()

    session_id = str(uuid.uuid4())
    temp_cache[session_id] = {"original": raw_text, "corrected": corrected}

    return {"session_id": session_id, "corrected": corrected}
