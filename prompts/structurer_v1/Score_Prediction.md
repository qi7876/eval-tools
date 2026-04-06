Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - map obvious answer labels such as "final answer", "answer", "prediction", "result", "conclusion", or "decision" into `text`
- If the raw model output contains reasoning plus a final answer, extract the final answer.
- If multiple candidate answers appear, prefer the last one presented as the final answer.
- Do not invent or complete missing content.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{
  "text": ""
}
