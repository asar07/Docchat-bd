"""
DocChat — FastAPI Backend
Deploy on Railway. Set GROQ_API_KEY as an environment variable / secret.
"""

import os
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any
import logging

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docchat")

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="DocChat API",
    version="1.0.0",
    description="Proxy API for DocChat — routes requests to Groq securely.",
    docs_url="/docs",
    redoc_url=None,
)

# ── CORS — allow Hugging Face Space origin ────────────────
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "*")   # set this in Railway env vars

app.add_middleware(
    CORSMiddleware,
    allow_origins=[HF_SPACE_URL, "http://localhost:3000", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── Groq config ───────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

ALLOWED_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

# ── Pydantic models ───────────────────────────────────────
class Message(BaseModel):
    role: str
    content: Any  # str or list (for vision)

class ChatRequest(BaseModel):
    model: str = Field(..., description="Groq model ID")
    messages: list[Message]
    max_tokens: int = Field(default=3000, le=8000)

class HealthResponse(BaseModel):
    status: str
    groq_configured: bool
    version: str

# ── Health check ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "version": "1.0.0",
    }

@app.get("/")
async def root():
    return {"message": "DocChat API is running. POST to /chat to use it."}

# ── Main proxy endpoint ───────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")

    if req.model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{req.model}' is not allowed.")

    payload = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                GROQ_BASE_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code != 200:
            logger.error(f"Groq error {response.status_code}: {response.text[:300]}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq API error: {response.text[:500]}",
            )

        data = response.json()
        return JSONResponse(content=data)

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Groq timed out. Please try again.")
    except httpx.RequestError as e:
        logger.error(f"Network error: {e}")
        raise HTTPException(status_code=502, detail="Could not reach Groq API.")

