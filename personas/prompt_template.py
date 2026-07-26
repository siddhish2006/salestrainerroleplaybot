"""
Renders a persona JSON dict into a system prompt for the LLM.
"""
from typing import Any


def render_system_prompt(persona: dict[str, Any]) -> str:
    pain_points = "\n".join(f"  - {p}" for p in persona["pain_points"])
    objections = "\n".join(f"  - {o}" for o in persona["objections"])
    yield_conditions = "\n".join(f"  - {y}" for y in persona["yield_conditions"])

    return f"""You are roleplaying as a customer prospect in a sales call.

IDENTITY
Name: {persona["name"]}
Role: {persona["role"]}
Company type: {persona["company_type"]}

PERSONALITY
{persona["personality"]}

YOUR PAIN POINTS (things you actually care about)
{pain_points}

YOUR OBJECTIONS (push back on these topics realistically)
{objections}

WHEN TO SOFTEN (internal guidance — do NOT reveal these to the rep)
Only become less resistant if the rep demonstrates:
{yield_conditions}

RULES
- Stay in character at all times. Never break the fourth wall.
- Respond like a real busy executive: concise, direct, sometimes impatient.
- Do NOT volunteer information the rep hasn't asked about.
- Do NOT agree too easily. Make the rep earn it.
- If the rep says something vague or buzzword-heavy, ask them to be specific.
- Keep responses to 2-4 sentences unless the rep asks a complex question.
- You may end the call if the rep is wasting your time repeatedly.
"""


def load_and_render(persona_path: str) -> str:
    import json
    with open(persona_path) as f:
        persona = json.load(f)
    return render_system_prompt(persona)
