Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - normalize explicit interval expressions such as "frames 2-4", "2 to 4", or "[2,4]" into `start_sampled` and `end_sampled`
  - map obvious field aliases such as "segments" or "actions" into the canonical fields
- If the raw model output contains reasoning plus a final structured answer, extract the final answer.
- If multiple candidate segment results appear, prefer the last one presented as the final answer.
- Do not infer unseen segments or frame indices.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
