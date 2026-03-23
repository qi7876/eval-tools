# system
You convert a raw model answer into the benchmark's canonical JSON output for {{task_name}}.
Use only information that explicitly appears in the raw answer.

# user
Question:
{{question}}

Number of sampled frames: {{num_sampled_frames}}
Valid sampled frame indices: {{sampled_index_range}}

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{{output_schema}}
