import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
import language_tool_python

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LanguageTool (downloads rule dictionary automatically)
tool = language_tool_python.LanguageTool('en-US')

# In-memory RAM storage (auto-deletes after 2 hours)
temp_cache = TTLCache(maxsize=10000, ttl=7200)

class TextPayload(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/fix")
def fix_text(payload: TextPayload):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Corrects text with zero censorship
    corrected = tool.correct(raw_text)

    session_id = str(uuid.uuid4())
    temp_cache[session_id] = {"original": raw_text, "corrected": corrected}

    return {"session_id": session_id, "corrected": corrected}
