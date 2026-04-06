Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - normalize explicit interval expressions such as "frames 2-4", "2 to 4", or "[2,4]" into `time_window_sampled`
  - map obvious field aliases such as "window" or "time span" into `time_window_sampled`
- If the raw model output contains reasoning plus a final structured answer, extract the final answer.
- If multiple candidate windows appear, prefer the last one presented as the final answer.
- Do not infer a time window that is not explicitly given.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{
  "time_window_sampled": []
}
