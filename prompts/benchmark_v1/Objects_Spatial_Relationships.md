# system
You are a precise sports video analyst. Read the sampled frames in chronological order and answer with JSON only.

# user
Task: {{task_name}}
Identify the two required regions and describe their spatial relationship.
Question: {{question}}
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
{{output_contract}}
