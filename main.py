import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
from gingerit.gingerit import GingerIt

app = FastAPI(title="Neutral Grammar Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize lightweight context parser (~35 MB RAM footprint)
parser = GingerIt()

# Temporary RAM cache (auto-deletes entries after 2 hours)
temp_cache = TTLCache(maxsize=10000, ttl=7200)

class TextPayload(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running", "engine": "GingerIt Context Engine"}

@app.post("/fix")
def fix_text(payload: TextPayload):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty text provided")

    try:
        result = parser.parse(raw_text)
        corrected = result.get("result", raw_text)
    except Exception:
        corrected = raw_text

    session_id = str(uuid.uuid4())
    temp_cache[session_id] = {"original": raw_text, "corrected": corrected}

    return {"session_id": session_id, "corrected": corrected}
