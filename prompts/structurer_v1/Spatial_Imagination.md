Convert the raw model output into the benchmark's canonical JSON for {{task_name}}.

Task-specific extraction rules:
- Produce exactly one JSON object matching the schema.
- Use the question only to understand what field the task expects. Do not copy answer content from the question into the prediction.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed: you may map obvious answer labels such as "final answer", "reasoning result", "trajectory answer", or similar wording into the canonical `text` field.
- If the raw model output contains reasoning plus a final answer, extract the final answer.
- If multiple candidate answers appear, prefer the last one presented as the final answer.
- Do not invent or complete missing prediction content.
- For this task, extract the final spatial reasoning answer text into `text`.

Question:
{{question}}

Number of sampled frames: {{num_sampled_frames}}
Valid sampled frame indices: {{sampled_index_range}}

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
