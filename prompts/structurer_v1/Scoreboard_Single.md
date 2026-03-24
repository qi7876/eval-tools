Convert the raw model output into the benchmark's canonical JSON for {{task_name}}.

Task-specific extraction rules:
- Produce exactly one JSON object matching the schema.
- Use the question only to understand what fields this task expects. Do not copy answer content from the question into the prediction.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - map obvious answer labels such as "final answer", "score", "result", or similar wording into the canonical `text` field
  - convert explicit coordinate strings, tuples, or lists into `bbox = [xtl, ytl, xbr, ybr]`
  - map obvious field aliases such as `box`, `bbox`, `scoreboard_box`, or similar labels into `bbox`
- If the raw model output contains reasoning plus a final answer, extract the final answer.
- If multiple candidate boxes or text answers appear, prefer the last ones presented as the final answer.
- Do not infer a bbox that is not explicitly given.
- Extract the final scoreboard answer text into `text`.

Question:
{{question}}

Number of sampled frames: {{num_sampled_frames}}
Valid sampled frame indices: {{sampled_index_range}}

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
