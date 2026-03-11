"""
DocChat — FastAPI Backend
Deploy on Railway. Set GROQ_API_KEY in Railway → Variables.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("docchat")

# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    port = os.getenv("PORT", "8000")
    key_ok = bool(os.getenv("GROQ_API_KEY"))
    logger.info(f"DocChat API starting  port={port}  groq_key={'SET' if key_ok else 'MISSING'}")
    yield
    logger.info("DocChat API shutting down")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocChat API",
    version="1.0.0",
    description="Secure Groq proxy for DocChat.",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow all origins so Railway health probes and HF Space both work.
# Restrict to your HF Space URL in production by setting HF_SPACE_URL env var.
_hf_url   = os.getenv("HF_SPACE_URL", "")
_origins  = ["*"] if not _hf_url else [
    _hf_url,
    "http://localhost:3000",
    "http://127.0.0.1:5500",
    "null",   # file:// origin for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

ALLOWED_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: Any          # str for text, list for vision

class ChatRequest(BaseModel):
    model:      str       = Field(..., description="Groq model ID")
    messages:   list[Message]
    max_tokens: int       = Field(default=3000, ge=1, le=8000)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "DocChat API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "version": "1.0.0",
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Add it in Railway → Variables.",
        )

    if req.model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' is not in the allowed list.",
        )

    payload = {
        "model":      req.model,
        "messages":   [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                GROQ_ENDPOINT,
                headers={
                    "Authorization":  f"Bearer {GROQ_API_KEY}",
                    "Content-Type":   "application/json",
                },
                json=payload,
            )

        if resp.status_code != 200:
            logger.error("Groq error %s: %s", resp.status_code, resp.text[:300])
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Groq API error: {resp.text[:400]}",
            )

        return JSONResponse(content=resp.json())

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Groq request timed out. Try again.")
    except httpx.RequestError as exc:
        logger.error("Network error reaching Groq: %s", exc)
        raise HTTPException(status_code=502, detail="Could not reach Groq API.")
