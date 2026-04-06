Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - map obvious answer labels such as "final answer", "score", or "result" into `text`
  - if the raw model output contains an explicit valid scoreboard bbox, extract it as `bbox = [x1, y1, x2, y2]`
  - map obvious field aliases such as `box`, `bbox`, or `scoreboard_box` into `bbox`
- If the raw model output contains reasoning plus a final answer, extract the final answer.
- If multiple candidate boxes or text answers appear, prefer the last ones presented as the final answer.
- Coordinates must stay in the normalized_1000 coordinate system, where `(0, 0)` is the top-left corner and `(1000, 1000)` is the bottom-right corner.
- If the raw model output does not contain an explicit valid scoreboard bbox, output `bbox = [-1, -1, -1, -1]`.
- Do not infer a scoreboard bbox beyond what is explicitly given in the raw model output.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{
  "text": "",
  "bbox": []
}
