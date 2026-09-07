import uuid
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache

app = FastAPI(title="Advanced Uncensored Text Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# High-quality open inference endpoints
HF_GRAMMAR_URL = "https://api-inference.huggingface.co/models/pszemraj/flan-t5-large-grammar-synthesis"
HF_PARAPHRASE_URL = "https://api-inference.huggingface.co/models/humarin/chatgpt_paraphraser_on_T5_base"

# Temporary in-memory cache (2-hour TTL)
temp_cache = TTLCache(maxsize=10000, ttl=7200)

class ProcessPayload(BaseModel):
    text: str
    mode: str = "grammar"  # "grammar", "paraphrase", or "fluent"

@app.get("/")
def root():
    return {"status": "running", "modes": ["grammar", "paraphrase", "fluent"]}

async def query_model(url: str, payload: dict) -> str:
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, json=payload)
        data = response.json()
        
        if isinstance(data, dict) and "error" in data:
            if "loading" in data.get("error", "").lower():
                raise HTTPException(
                    status_code=503, 
                    detail="Model is warming up on cloud GPU. Please retry in 10 seconds."
                )
            raise HTTPException(status_code=500, detail=data["error"])
        
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("generated_text", "")
        return ""

@app.post("/process")
async def process_text(payload: ProcessPayload):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        if payload.mode == "paraphrase":
            # Paraphrase mode: Rephrases content while maintaining core semantics
            prompt = f"paraphrase: {raw_text}"
            corrected = await query_model(
                HF_PARAPHRASE_URL, 
                {"inputs": prompt, "parameters": {"max_length": 512, "temperature": 0.7}}
            )
        else:
            # Grammar mode: Pure syntax, tense, spelling, and punctuation correction
            corrected = await query_model(
                HF_GRAMMAR_URL, 
                {"inputs": raw_text, "parameters": {"max_length": 512}}
            )

        if not corrected:
            corrected = raw_text

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {str(e)}")

    session_id = str(uuid.uuid4())
    temp_cache[session_id] = {
        "original": raw_text,
        "corrected": corrected,
        "mode": payload.mode
    }

    return {"session_id": session_id, "corrected": corrected, "mode": payload.mode}
