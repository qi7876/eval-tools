Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - map obvious field aliases such as `objects`, `boxes`, or `detections` into `objects`
  - convert explicit coordinate strings, tuples, or lists into `[xtl, ytl, xbr, ybr]`
  - map obvious answer labels into `text`
- If the raw model output contains reasoning plus a final answer, extract the final answer.
- If multiple candidate object lists appear, prefer the last one presented as the final answer.
- Each object must contain `label` and `bbox`.
- Use exactly these object labels: {{required_object_labels_json}}.
- Match boxes by explicit label, not by first/second position.
- Do not infer boxes that are not explicitly given.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
