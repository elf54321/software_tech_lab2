"""Minimal FastAPI backend: streams a response from a local Ollama model
and logs every request to SQLite for research.
Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import json
import sqlite3
import time

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OLLAMA = "http://localhost:11434"
DB = "experiments.db"

app = FastAPI(title="LLM Experimentation Playground")


def init_db():
    with sqlite3.connect(DB) as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, model TEXT, prompt TEXT, response TEXT,
                first_token_s REAL, total_s REAL, chars INTEGER)"""
        )


init_db()


class GenerateRequest(BaseModel):
    model: str = "llama3.2:3b"
    prompt: str


@app.get("/models")
async def models():
    """List models currently pulled in Ollama."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{OLLAMA}/api/tags")
    return [m["name"] for m in r.json()["models"]]


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Stream tokens from one model and log the run."""

    async def stream():
        start = time.time()
        first = None
        text = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA}/api/generate",
                json={"model": req.model, "prompt": req.prompt, "stream": True},
            ) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line).get("response", "")
                    if chunk and first is None:
                        first = time.time() - start
                    text += chunk
                    yield chunk
        with sqlite3.connect(DB) as con:
            con.execute(
                "INSERT INTO runs (ts, model, prompt, response, first_token_s, total_s, chars)"
                " VALUES (?,?,?,?,?,?,?)",
                (time.time(), req.model, req.prompt, text, first,
                 time.time() - start, len(text)),
            )

    return StreamingResponse(stream(), media_type="text/plain")
