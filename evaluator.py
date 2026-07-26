"""Post-call sales-performance evaluation, kept separate from the realtime agent."""

import asyncio
import json
import os
from pathlib import Path
from typing import Literal

import aiohttp
from pydantic import BaseModel, Field


PERSONAS_DIR = Path(__file__).parent / "personas"

PROVIDER_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash-lite",
    "anthropic": "claude-sonnet-4-6",
}
PROVIDER_ENV_VARS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


class CategoryScores(BaseModel):
    opening_rapport: int = Field(ge=0, le=10)
    discovery_questioning: int = Field(ge=0, le=10)
    listening_adaptability: int = Field(ge=0, le=10)
    value_proposition: int = Field(ge=0, le=10)
    objection_handling: int = Field(ge=0, le=10)
    persuasion: int = Field(ge=0, le=10)
    product_communication: int = Field(ge=0, le=10)
    customer_engagement: int = Field(ge=0, le=10)
    closing_next_step: int = Field(ge=0, le=10)
    sales_judgment: int = Field(ge=0, le=10)


class Moment(BaseModel):
    evidence: str
    explanation: str


class EvaluationReport(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    scores: CategoryScores
    customer_start_state: Literal[
        "HOSTILE", "SKEPTICAL", "NEUTRAL", "INTERESTED", "HIGHLY_INTERESTED"
    ]
    customer_end_state: Literal[
        "HOSTILE", "SKEPTICAL", "NEUTRAL", "INTERESTED", "HIGHLY_INTERESTED"
    ]
    conversion_outcome: Literal[
        "FAILED",
        "NO_PROGRESS",
        "SLIGHT_PROGRESS",
        "STRONG_PROGRESS",
        "NEXT_STEP_SECURED",
        "SALE_SECURED",
    ]
    customer_movement_explanation: str
    top_strengths: list[str] = Field(max_length=3)
    critical_mistakes: list[str] = Field(max_length=3)
    missed_opportunities: list[str] = Field(max_length=3)
    objections_handled_well: list[str]
    objections_handled_poorly: list[str]
    best_moment: Moment
    worst_moment: Moment
    better_response: str
    coaching_plan: list[str] = Field(min_length=3, max_length=3)
    final_verdict: str


def _load_persona(persona_id: str) -> dict:
    persona_path = PERSONAS_DIR / f"{persona_id}.json"
    if not persona_path.is_file():
        raise ValueError(f"Persona not found: {persona_id}")
    return json.loads(persona_path.read_text(encoding="utf-8"))


def _transcript_text(turns: list[dict]) -> str:
    lines = []
    for turn in turns:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        speaker = "SALESPERSON" if turn.get("role") == "user" else "CUSTOMER"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines) or "No usable conversation turns were recorded."


def _evaluation_prompt(transcript: dict, persona: dict) -> str:
    persona_context = {
        "name": persona.get("name"),
        "role": persona.get("role"),
        "company_type": persona.get("company_type"),
        "personality": persona.get("personality"),
        "pain_points": persona.get("pain_points", []),
        "objections": persona.get("objections", []),
        "yield_conditions": persona.get("yield_conditions", []),
    }
    transcript_text = _transcript_text(transcript.get("turns", []))

    return f"""You are a demanding but fair sales manager evaluating a roleplay call.

Evaluate the SALESPERSON only, never the simulated customer. Judge the salesperson against the customer persona, its pain points, objections, and yield conditions. Do not reward the salesperson merely because the customer became cooperative; identify the salesperson actions that caused meaningful progress. Do not invent events, product details, or evidence absent from the transcript. If product/service or difficulty is not supplied, state that limitation where relevant instead of guessing.

CUSTOMER PERSONA:
{json.dumps(persona_context, ensure_ascii=False)}

PRODUCT/SERVICE: Not provided
SCENARIO/DIFFICULTY: Not provided

TRANSCRIPT:
{transcript_text}

Return valid JSON only. No markdown, commentary, or code fences. Use exactly this structure:
{{
  "overall_score": 0,
  "scores": {{
    "opening_rapport": 0,
    "discovery_questioning": 0,
    "listening_adaptability": 0,
    "value_proposition": 0,
    "objection_handling": 0,
    "persuasion": 0,
    "product_communication": 0,
    "customer_engagement": 0,
    "closing_next_step": 0,
    "sales_judgment": 0
  }},
  "customer_start_state": "HOSTILE|SKEPTICAL|NEUTRAL|INTERESTED|HIGHLY_INTERESTED",
  "customer_end_state": "HOSTILE|SKEPTICAL|NEUTRAL|INTERESTED|HIGHLY_INTERESTED",
  "conversion_outcome": "FAILED|NO_PROGRESS|SLIGHT_PROGRESS|STRONG_PROGRESS|NEXT_STEP_SECURED|SALE_SECURED",
  "customer_movement_explanation": "",
  "top_strengths": ["maximum 3, each grounded in evidence"],
  "critical_mistakes": ["maximum 3, each grounded in evidence"],
  "missed_opportunities": ["maximum 3, each grounded in evidence"],
  "objections_handled_well": ["list only objections handled effectively"],
  "objections_handled_poorly": ["list ignored, mishandled, or inadequately answered objections"],
  "best_moment": {{"evidence": "short quote or precise paraphrase", "explanation": ""}},
  "worst_moment": {{"evidence": "short quote or precise paraphrase", "explanation": ""}},
  "better_response": "A stronger exact response to the worst moment.",
  "coaching_plan": ["exactly 3 concrete practices"],
  "final_verdict": "A demanding but fair 2-4 sentence manager assessment."
}}

Use 0-10 integers for every category and 0-100 integer for overall_score. Score these categories: opening and rapport; discovery and questioning; listening and adaptability; value proposition; objection handling; persuasion; product/business communication; customer engagement; closing/next step; and overall sales judgment. Every material praise or criticism must cite a short excerpt or precise paraphrase from the salesperson's words."""


def _json_from_model_text(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


async def _request_groq(session: aiohttp.ClientSession, prompt: str, api_key: str) -> str:
    payload = {
        "model": PROVIDER_MODELS["groq"],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only valid JSON that follows the user's required schema."},
            {"role": "user", "content": prompt},
        ],
    }
    async with session.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    ) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"Groq evaluation request failed with HTTP {response.status}")
    return json.loads(body)["choices"][0]["message"]["content"]


async def _request_gemini(session: aiohttp.ClientSession, prompt: str, api_key: str) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{PROVIDER_MODELS['gemini']}:generateContent?key={api_key}"
    )
    async with session.post(url, json=payload) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"Gemini evaluation request failed with HTTP {response.status}")
    parts = json.loads(body)["candidates"][0]["content"]["parts"]
    return "".join(part.get("text", "") for part in parts)


async def _request_anthropic(session: aiohttp.ClientSession, prompt: str, api_key: str) -> str:
    payload = {
        "model": PROVIDER_MODELS["anthropic"],
        "max_tokens": 4096,
        "temperature": 0.2,
        "system": "Return only valid JSON that follows the user's required schema.",
        "messages": [{"role": "user", "content": prompt}],
    }
    async with session.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
    ) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"Anthropic evaluation request failed with HTTP {response.status}")
    blocks = json.loads(body)["content"]
    return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")


async def evaluate_transcript(transcript_path: Path) -> EvaluationReport:
    """Evaluate one saved transcript using the LLM used for that roleplay."""
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    provider_id = transcript.get("provider", "groq")
    if provider_id not in PROVIDER_MODELS:
        raise ValueError(f"Unsupported evaluation provider: {provider_id}")

    api_key = os.getenv(PROVIDER_ENV_VARS[provider_id])
    if not api_key:
        raise RuntimeError(f"{PROVIDER_ENV_VARS[provider_id]} is not configured")

    persona = _load_persona(transcript.get("persona_id", ""))
    prompt = _evaluation_prompt(transcript, persona)
    timeout = aiohttp.ClientTimeout(total=90)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        if provider_id == "groq":
            content = await _request_groq(session, prompt, api_key)
        elif provider_id == "gemini":
            content = await _request_gemini(session, prompt, api_key)
        else:
            content = await _request_anthropic(session, prompt, api_key)

    return EvaluationReport.model_validate(_json_from_model_text(content))


async def evaluate_and_save(transcript_path: Path) -> Path:
    """Run after a call ends and persist the report beside its transcript."""
    report = await evaluate_transcript(transcript_path)
    output_path = transcript_path.with_suffix(".evaluation.json")
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output_path


async def evaluate_and_save_in_background(transcript_path: Path) -> Path:
    """Yield once so API callers receive their pending response before evaluation starts."""
    await asyncio.sleep(0)
    return await evaluate_and_save(transcript_path)
