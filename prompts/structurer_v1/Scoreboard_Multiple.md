Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - map obvious answer labels such as "final answer", "scoreboard", or "result" into `text`
- If the raw model output contains reasoning plus a final answer, extract the final answer.
- If multiple candidate scoreboard answers appear, prefer the last one presented as the final answer.
- Do not invent unreadable or missing scoreboard content.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
