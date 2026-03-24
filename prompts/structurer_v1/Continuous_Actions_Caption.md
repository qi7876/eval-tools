Convert the raw model output into the benchmark's canonical JSON for {{task_name}}.

Task-specific extraction rules:
- Produce exactly one JSON object matching the schema.
- Use the question only to understand what fields this task expects. Do not copy answer content from the question into the prediction.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - normalize explicit interval expressions such as "frames 2-4", "2 to 4", "[2,4]" into `start_sampled` and `end_sampled`
  - normalize explicit tracking rows or coordinate strings into `frame_sampled` and `bbox_mot`
  - normalize obvious field aliases such as "track", "boxes", "trajectory", "segments", or "actions" into the canonical fields
- If the raw model output contains reasoning plus a final structured answer, extract the final answer.
- If multiple candidate segment or tracking results appear, prefer the last one presented as the final answer.
- Do not infer unseen segments, frame indices, or boxes.
- `segments` should contain action descriptions with sampled-frame intervals.
- `tracking` should contain only explicit sampled-frame box rows.
- `bbox_mot` must be formatted as `[left, top, width, height]`.

Question:
{{question}}

Number of sampled frames: {{num_sampled_frames}}
Valid sampled frame indices: {{sampled_index_range}}

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
