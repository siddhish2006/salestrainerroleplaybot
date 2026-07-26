# Persona JSON Schema

Each persona file is a JSON object with these fields:

## Required Fields

```json
{
  "id": "string — unique snake_case identifier, used in room metadata",
  "name": "string — display name shown in UI dropdown",
  "role": "string — job title of the prospect",
  "company_type": "string — type/industry of company",
  "personality": "string — 1-2 sentence description of disposition and communication style",
  "pain_points": ["array of strings — top 2-4 business problems they care about"],
  "objections": ["array of strings — typical objections they raise"],
  "yield_conditions": ["array of strings — what would actually move them; shown to LLM as signals to soften stance"],
  "voice_id": "string — Cartesia voice ID to use for this persona",
  "tts_emotion": "string — emotion hint passed to Cartesia (e.g. 'neutral', 'curious', 'annoyed')"
}
```

## Notes

- `id` must match the filename (e.g. `skeptical_cfo_manufacturing.json` → `"id": "skeptical_cfo_manufacturing"`)
- `yield_conditions` are NOT things the persona volunteers — they are internal LLM guidance on when to become slightly less resistant
- Keep `objections` grounded in the industry; generic objections ("too expensive") are less useful than specific ones ("our ERP migration is locked in for 18 months, we can't add integrations right now")
- `voice_id` and `tts_emotion` are looked up at agent startup; invalid IDs will cause Cartesia to error at call time
