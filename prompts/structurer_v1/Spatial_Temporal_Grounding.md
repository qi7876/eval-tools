Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - normalize explicit interval expressions such as "frames 2-4", "2 to 4", or "[2,4]" into `time_window_sampled`
  - normalize explicit tracking rows or coordinate strings into `frame_sampled` and `bbox_mot`
  - map obvious field aliases such as "window", "time span", "track", or "boxes" into the canonical fields
- If the raw model output contains reasoning plus a final structured answer, extract the final answer.
- If multiple candidate windows or tracking results appear, prefer the last ones presented as the final answer.
- Do not infer a time window or boxes that are not explicitly given.
- `bbox_mot` must be formatted as normalized_1000 `[left, top, width, height]`.
- In this coordinate system, `(0, 0)` is the top-left corner and `(1000, 1000)` is the bottom-right corner.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{
  "time_window_sampled": [],
  "tracking": []
}
