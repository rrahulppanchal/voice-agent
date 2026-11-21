import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.container import Services
from app.session.session import VoiceSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("voice-agent")

STATIC = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models...")
    app.state.services = await Services.create(settings)
    log.info("Voice agent ready at http://localhost:8000")
    yield
    await app.state.services.close()


app = FastAPI(title="Local Voice Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
async def health():
    svc: Services = app.state.services
    return {"ok": True, "llm": settings.llm_model, "stt": f"{svc.stt.name}:{settings.stt_model}", "tts": svc.tts.name}


@app.websocket("/ws/voice")
async def voice(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")
    try:
        await VoiceSession(ws, ws.app.state.services).run()
    finally:
        log.info("Client disconnected")
