Convert the raw model output into the benchmark's canonical JSON for {{task_name}}.

Task-specific extraction rules:
- Produce exactly one JSON object matching the schema.
- Use the question only to understand what fields this task expects. Do not copy answer content from the question into the prediction.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - normalize explicit interval expressions such as "frames 2-4", "2 to 4", "[2,4]" into `time_window_sampled`
  - normalize explicit tracking rows or coordinate strings into `frame_sampled` and `bbox_mot`
  - map obvious field aliases such as "window", "time span", "track", "boxes", or similar labels into the canonical fields
- If the raw model output contains reasoning plus a final structured answer, extract the final answer.
- If multiple candidate windows or tracking results appear, prefer the last ones presented as the final answer.
- Do not infer a time window or boxes that are not explicitly given.
- `time_window_sampled` must be a two-value sampled-frame interval.
- `bbox_mot` must be formatted as `[left, top, width, height]`.

Question:
{{question}}

Number of sampled frames: {{num_sampled_frames}}
Valid sampled frame indices: {{sampled_index_range}}

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
