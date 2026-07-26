"""
Phase 2-4 — STT -> LLM -> TTS pipeline with transcript saving.
LLM provider selected dynamically from room metadata.
"""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    ConversationItemAddedEvent,
    JobContext,
    WorkerOptions,
)
from livekit.agents.cli import run_app
from livekit.plugins import anthropic, deepgram, google, groq, silero

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSONAS_DIR = Path(__file__).parent / "personas"
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
sys.path.insert(0, str(Path(__file__).parent))


def load_persona(persona_id: str) -> dict:
    path = PERSONAS_DIR / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No persona file: {path}")
    with open(path) as f:
        return json.load(f)


def make_llm(provider_id: str):
    if provider_id == "anthropic":
        return anthropic.LLM(model="claude-sonnet-4-6")
    elif provider_id == "gemini":
        return google.LLM(model="gemini-2.0-flash-lite")
    else:  # default: groq
        return groq.LLM(model="llama-3.3-70b-versatile")


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Metadata is now JSON: {"persona_id": "...", "provider_id": "..."}
    try:
        meta = json.loads(ctx.room.metadata or "{}")
    except json.JSONDecodeError:
        # Backwards compat: plain string persona_id
        meta = {"persona_id": ctx.room.metadata or "skeptical_cfo_manufacturing"}

    persona_id  = meta.get("persona_id", "skeptical_cfo_manufacturing")
    provider_id = meta.get("provider_id", "groq")
    logger.info(f"persona_id={persona_id!r} provider_id={provider_id!r}")

    try:
        persona = load_persona(persona_id)
    except FileNotFoundError:
        logger.warning(f"Persona '{persona_id}' not found, using fallback")
        persona = load_persona("skeptical_cfo_manufacturing")

    from personas.prompt_template import render_system_prompt
    system_prompt = render_system_prompt(persona)
    logger.info(f"Loaded persona: {persona['name']} | provider: {provider_id}")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPTS_DIR / f"{ctx.room.name}.json"
    turns = []
    call_start = time.time()

    def save_transcript():
        data = {
            "room": ctx.room.name,
            "persona_id": persona_id,
            "persona_name": persona["name"],
            "provider": provider_id,
            "started_at": call_start,
            "turns": turns,
        }
        transcript_path.write_text(json.dumps(data, indent=2))

    # Create a reportable file even if the caller ends before a reply is generated.
    save_transcript()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=make_llm(provider_id),
        # Deepgram provides both STT and TTS through the existing DEEPGRAM_API_KEY.
        tts=deepgram.TTS(model="aura-2-andromeda-en"),
    )

    @session.on("conversation_item_added")
    def on_turn(ev: ConversationItemAddedEvent):
        item = ev.item
        if not hasattr(item, "role") or item.role not in ("user", "assistant"):
            return
        text = " ".join(c for c in item.content if isinstance(c, str)).strip()
        if not text:
            return
        turns.append({
            "role": item.role,
            "text": text,
            "timestamp": round(ev.created_at - call_start, 2),
        })
        logger.info(f"[{item.role}] {text[:80]}")
        save_transcript()

    agent = Agent(instructions=system_prompt)
    await session.start(agent, room=ctx.room)
    await session.generate_reply(
        instructions=f"Start the call. Introduce yourself as {persona['name']} briefly and wait for them to speak."
    )

    await asyncio.Event().wait()


if __name__ == "__main__":
    run_app(WorkerOptions(entrypoint_fnc=entrypoint))
