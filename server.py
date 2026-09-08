import os
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler,
    SmallWebRTCRequest,
    SmallWebRTCPatchRequest,
)
from pipecat.runner.types import SmallWebRTCRunnerArguments

from config.settings import settings
import bot

app = FastAPI(title="Hindi Telecalling WebRTC Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure ICE Servers
ice_servers = [
    IceServer(urls=settings.ice_servers)
]

small_webrtc_handler = SmallWebRTCRequestHandler(
    ice_servers=ice_servers,
    esp32_mode=False,
    host=settings.host,
)

# Background tasks reference tracking
_active_sessions = set()

@app.on_event("startup")
async def startup_warmup():
    logger.info("Pre-warming models on GPU during server startup...")
    try:
        bot.get_shared_models()
        logger.info("All AI models pre-warmed successfully!")
    except Exception as e:
        logger.error(f"Error pre-warming models: {e}")

@app.get("/")
async def index():
    """Serves the WebRTC test client UI."""
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "gpu": settings.tts_device}

@app.post("/api/offer")
async def offer(
    request: SmallWebRTCRequest,
    background_tasks: BackgroundTasks,
    session_id: str | None = None,
):
    """
    WebRTC offer endpoint compliant with Pipecat SmallWebRTC transport.
    Invokes the Pipecat bot task upon session initialization.
    """
    resolved_session_id = session_id or str(uuid.uuid4())

    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        runner_args = SmallWebRTCRunnerArguments(
            webrtc_connection=connection,
            body=request.request_data,
            session_id=resolved_session_id,
        )
        logger.info(f"Dispatching bot session for peer connection {connection.pc_id}...")
        background_tasks.add_task(bot.bot, runner_args)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer

@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Handles trickle ICE candidates."""
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}

if __name__ == "__main__":
    logger.info(f"Starting FastAPI WebRTC Server on {settings.host}:{settings.port}...")
    uvicorn.run(app, host=settings.host, port=settings.port)
