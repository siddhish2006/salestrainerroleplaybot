"""
Phase 1 — echo agent.
Joins any new room, echoes participant audio back.
No STT/LLM/TTS — proves LiveKit audio plumbing only.
"""
import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions
from livekit.agents.cli import run_app

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext):
    logger.info(f"Agent joined room: {ctx.room.name} | metadata: {ctx.room.metadata!r}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Outbound audio source — we push received frames into this
    audio_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
    echo_track = rtc.LocalAudioTrack.create_audio_track("echo", audio_source)
    await ctx.room.local_participant.publish_track(
        echo_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    logger.info("Echo track published")

    async def forward(track: rtc.RemoteAudioTrack):
        stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
        async for event in stream:
            await audio_source.capture_frame(event.frame)

    @ctx.room.on("track_subscribed")
    def on_track(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to audio: {track.sid} — starting echo")
            asyncio.ensure_future(forward(track))

    logger.info("Phase 1 echo agent ready")
    await asyncio.Event().wait()


if __name__ == "__main__":
    run_app(WorkerOptions(entrypoint_fnc=entrypoint))
