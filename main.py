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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("docchat")

@asynccontextmanager
async def lifespan(app: FastAPI):
    port = os.getenv("PORT", "8000")
    key_ok = bool(os.getenv("GROQ_API_KEY"))
    logger.info(f"DocChat starting — port={port}  groq_key={'SET' if key_ok else '*** MISSING ***'}")
    yield
    logger.info("DocChat shutting down")

app = FastAPI(title="DocChat API", version="1.0.0", docs_url="/docs", redoc_url=None, lifespan=lifespan)

# ── CORS: allow ALL origins (safe — API key lives server-side only) ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

ALLOWED_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

class Message(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model:      str = Field(...)
    messages:   list[Message]
    max_tokens: int = Field(default=3000, ge=1, le=8000)

@app.get("/")
async def root():
    return {"status": "ok", "service": "DocChat API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok", "groq_configured": bool(GROQ_API_KEY), "version": "1.0.0"}

@app.post("/chat")
async def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY is not set. Add it in Railway → Variables.")
    if req.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"Model '{req.model}' is not allowed.")

    payload = {
        "model":      req.model,
        "messages":   [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code != 200:
            logger.error("Groq %s: %s", resp.status_code, resp.text[:300])
            raise HTTPException(resp.status_code, f"Groq error: {resp.text[:400]}")
        return JSONResponse(content=resp.json())

    except httpx.TimeoutException:
        raise HTTPException(504, "Groq request timed out.")
    except httpx.RequestError as exc:
        logger.error("Network error: %s", exc)
        raise HTTPException(502, "Could not reach Groq API.")
        
