"""
FastAPI backend — creates LiveKit rooms and issues join tokens.
"""
import asyncio
import json
import os
import re
import uuid
from pathlib import Path

PERSONAS_DIR = Path(__file__).parent / "personas"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from livekit.api import AccessToken, LiveKitAPI, VideoGrants
from livekit.api.room_service import CreateRoomRequest

from evaluator import evaluate_and_save_in_background

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROVIDERS = [
    {"id": "groq",      "name": "Groq",      "model": "llama-3.3-70b-versatile", "env": "GROQ_API_KEY"},
    {"id": "gemini",    "name": "Gemini",     "model": "gemini-2.0-flash-lite",   "env": "GOOGLE_API_KEY"},
    {"id": "anthropic", "name": "Anthropic",  "model": "claude-sonnet-4-6",       "env": "ANTHROPIC_API_KEY"},
]
EVALUATION_JOBS: dict[str, asyncio.Task] = {}


def transcript_path_for_room(room_name: str) -> Path:
    if not re.fullmatch(r"roleplay-[A-Za-z0-9_-]+", room_name):
        raise HTTPException(status_code=400, detail="Invalid room name")
    return TRANSCRIPTS_DIR / f"{room_name}.json"


def evaluation_error_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".evaluation-error.json")


def start_evaluation(room_name: str, transcript_path: Path) -> None:
    if room_name in EVALUATION_JOBS:
        return

    error_path = evaluation_error_path(transcript_path)
    error_path.unlink(missing_ok=True)
    task = asyncio.create_task(evaluate_and_save_in_background(transcript_path))
    EVALUATION_JOBS[room_name] = task

    def record_result(completed_task: asyncio.Task) -> None:
        EVALUATION_JOBS.pop(room_name, None)
        if completed_task.cancelled():
            return
        error = completed_task.exception()
        if error:
            error_path.write_text(
                json.dumps({"error": "Evaluation could not be completed", "detail": str(error)}),
                encoding="utf-8",
            )

    task.add_done_callback(record_result)


@app.get("/personas")
async def list_personas():
    personas = []
    for f in sorted(PERSONAS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        personas.append({"id": data["id"], "name": data["name"], "role": data["role"], "company_type": data["company_type"]})
    return personas


@app.get("/providers")
async def list_providers():
    # Only return providers whose API key is actually set
    return [p for p in PROVIDERS if os.environ.get(p["env"])]


@app.get("/token")
async def get_token(persona_id: str, provider_id: str = "groq"):
    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    livekit_url = os.environ.get("LIVEKIT_URL")

    if not all([api_key, api_secret, livekit_url]):
        raise HTTPException(status_code=500, detail="LiveKit env vars not set")

    room_name = f"roleplay-{persona_id}-{uuid.uuid4().hex[:8]}"
    # Pack both persona_id and provider_id into metadata as JSON
    metadata = json.dumps({"persona_id": persona_id, "provider_id": provider_id})

    async with LiveKitAPI(url=livekit_url, api_key=api_key, api_secret=api_secret) as lk:
        await lk.room.create_room(
            CreateRoomRequest(name=room_name, metadata=metadata)
        )

    participant_identity = f"rep-{uuid.uuid4().hex[:6]}"
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name("Sales Rep")
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    return {"token": token, "room": room_name, "url": livekit_url}


@app.post("/evaluations/{room_name}", status_code=202)
async def request_evaluation(room_name: str):
    """Start a post-call evaluation without affecting the realtime agent."""
    transcript_path = transcript_path_for_room(room_name)
    if not transcript_path.is_file():
        raise HTTPException(status_code=404, detail="Transcript is not available yet")

    report_path = transcript_path.with_suffix(".evaluation.json")
    if report_path.is_file():
        return {"status": "completed", "evaluation": json.loads(report_path.read_text(encoding="utf-8"))}

    start_evaluation(room_name, transcript_path)
    return {"status": "pending"}


@app.get("/evaluations/{room_name}")
async def get_evaluation(room_name: str):
    """Return a completed evaluation or the current async job status."""
    transcript_path = transcript_path_for_room(room_name)
    report_path = transcript_path.with_suffix(".evaluation.json")
    if report_path.is_file():
        return {"status": "completed", "evaluation": json.loads(report_path.read_text(encoding="utf-8"))}

    error_path = evaluation_error_path(transcript_path)
    if error_path.is_file():
        return {"status": "error", **json.loads(error_path.read_text(encoding="utf-8"))}
    if room_name in EVALUATION_JOBS:
        return {"status": "pending"}
    if not transcript_path.is_file():
        raise HTTPException(status_code=404, detail="Transcript is not available yet")
    return {"status": "not_requested"}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
