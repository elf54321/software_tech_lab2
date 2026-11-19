# run with: uvicorn main:app --reload

import json
import os
import secrets
import sqlite3
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

OLLAMA = "http://localhost:11434"
DB = "experiments.db"
DEFAULT_DAILY_LIMIT = 50
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="LLM Experimentation Playground")


def db():
    return sqlite3.connect(DB)


def init_db():
    with db() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS api_keys "
            "(key TEXT PRIMARY KEY, owner TEXT, daily_limit INTEGER)"
        )
        con.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL, owner TEXT, model TEXT, prompt TEXT, system TEXT,
                temperature REAL, top_p REAL, max_tokens INTEGER, seed INTEGER,
                response TEXT, first_token_s REAL, total_s REAL, chars INTEGER)"""
        )


init_db()


def create_key(owner, daily_limit=DEFAULT_DAILY_LIMIT):
    key = "sk-" + secrets.token_urlsafe(16)
    with db() as con:
        con.execute("INSERT INTO api_keys VALUES (?,?,?)", (key, owner, daily_limit))
    return key


def check_key(key):
    if not key:
        raise HTTPException(401, "Missing API key")
    with db() as con:
        row = con.execute(
            "SELECT owner, daily_limit FROM api_keys WHERE key=?", (key,)
        ).fetchone()
        if not row:
            raise HTTPException(401, "Invalid API key")
        owner, limit = row
        used = con.execute(
            "SELECT COUNT(*) FROM runs WHERE owner=? AND ts>?",
            (owner, time.time() - 86400),
        ).fetchone()[0]
    if used >= limit:
        raise HTTPException(429, "Limit reached for the last 24 hours")
    return owner


class GenerateRequest(BaseModel):
    model: str = "llama3.2:3b"
    prompt: str
    system: Optional[str] = None
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(0.9, ge=0, le=1)
    max_tokens: int = Field(512, ge=16, le=2048)
    seed: Optional[int] = None


@app.get("/")
async def home():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/models")
async def models():
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{OLLAMA}/api/tags")
    return [m["name"] for m in r.json()["models"]]


@app.post("/generate")
async def generate(req: GenerateRequest, x_api_key: Optional[str] = Header(None)):
    owner = check_key(x_api_key)

    options = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "num_predict": req.max_tokens,
    }
    if req.seed is not None:
        options["seed"] = req.seed
    body = {"model": req.model, "prompt": req.prompt, "stream": True, "options": options}
    if req.system:
        body["system"] = req.system

    async def stream():
        start = time.time()
        first = None
        text = ""
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA}/api/generate", json=body) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line).get("response", "")
                    if chunk and first is None:
                        first = time.time() - start
                    text += chunk
                    yield chunk
        # save the finished run
        with db() as con:
            con.execute(
                "INSERT INTO runs (ts, owner, model, prompt, system, temperature, top_p, "
                "max_tokens, seed, response, first_token_s, total_s, chars) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), owner, req.model, req.prompt, req.system, req.temperature,
                 req.top_p, req.max_tokens, req.seed, text, first,
                 time.time() - start, len(text)),
            )

    return StreamingResponse(stream(), media_type="text/plain")
