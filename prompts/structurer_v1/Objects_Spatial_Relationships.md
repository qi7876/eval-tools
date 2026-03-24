Convert the raw model output into the benchmark's canonical JSON for {{task_name}}.

Task-specific extraction rules:
- Produce exactly one JSON object matching the schema.
- Use the question only to understand what fields this task expects. Do not copy answer content from the question into the prediction.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - map obvious field aliases such as `box_a`, `box_b`, `bbox1`, `bbox2`, `first box`, `second box`, or similar labels into `bbox_a` and `bbox_b`
  - convert explicit coordinate strings, tuples, or lists into `[xtl, ytl, xbr, ybr]`
  - map obvious answer labels into the canonical `text` field
- If the raw model output contains reasoning plus a final answer, extract the final answer.
- If multiple candidate box pairs appear, prefer the last pair presented as the final answer.
- Preserve the original box order from the raw model output. Do not swap `bbox_a` and `bbox_b` on your own.
- Do not infer boxes that are not explicitly given.
- Extract the spatial relation text into `text`.

Question:
{{question}}

Number of sampled frames: {{num_sampled_frames}}
Valid sampled frame indices: {{sampled_index_range}}

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
