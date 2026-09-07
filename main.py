import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cachetools import TTLCache
import spacy
from symspellpy import SymSpell, Verbosity
import pkg_resources

app = FastAPI(title="Neutral Grammar Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load lightweight English NLP parser (<100MB RAM)
nlp = spacy.load("en_core_web_sm")

# 2. Initialize SymSpell engine for context & edit-distance corrections
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

# Temporary RAM cache (auto-deletes entries after 2 hours)
temp_cache = TTLCache(maxsize=10000, ttl=7200)

class TextPayload(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "running", "engine": "SymSpell + SpaCy Context Engine"}

@app.post("/fix")
def fix_text(payload: TextPayload):
    raw_text = payload.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty text provided")

    # Pass 1: Multi-word bigram and frequency-based spelling/phrase correction
    suggestions = sym_spell.lookup_compound(raw_text, max_edit_distance=2)
    corrected_base = suggestions[0].term if suggestions else raw_text

    # Pass 2: Sentence capitalization and token boundary cleanup via SpaCy
    doc = nlp(corrected_base)
    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if sent_text:
            # Capitalize first letter of every sentence
            sent_text = sent_text[0].upper() + sent_text[1:]
            sentences.append(sent_text)
    
    final_text = " ".join(sentences) if sentences else corrected_base

    session_id = str(uuid.uuid4())
    temp_cache[session_id] = {"original": raw_text, "corrected": final_text}

    return {"session_id": session_id, "corrected": final_text}
