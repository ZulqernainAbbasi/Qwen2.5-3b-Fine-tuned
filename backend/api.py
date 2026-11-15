"""FastAPI backend: the laptop-side service the Streamlit Cloud app talks to through the tunnel.

Run: uvicorn backend.api:app --port 8000   (needs env CAR_API_KEY)
"""
import hmac
import logging
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import config
from agent.graph import run_agent
from backend import db

load_dotenv(config.ROOT / ".env")
log = logging.getLogger("car-api")
app = FastAPI(title="PakCar Assistant API", version="1.0")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)


def require_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("CAR_API_KEY", "")
    if not expected or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": config.LLM_MODEL}


@app.post("/chat", dependencies=[Depends(require_key)])
def chat(body: ChatIn) -> dict:
    try:
        return run_agent(body.session_id, body.message)
    except Exception as e:  # model server down, etc.
        log.exception("agent failed")
        raise HTTPException(status_code=503, detail=f"assistant unavailable: {type(e).__name__}")


@app.get("/bookings", dependencies=[Depends(require_key)])
def bookings() -> dict:
    return {"bookings": db.list_bookings(), "tickets": db.list_tickets()}
