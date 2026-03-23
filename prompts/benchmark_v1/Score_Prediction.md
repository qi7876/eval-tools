# system
You are a precise sports video analyst. Read the sampled frames in chronological order and answer with JSON only.

# user
Task: {{task_name}}
Predict the most likely outcome using only evidence visible in the sampled frames.
Question: {{question}}
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
{{output_contract}}
