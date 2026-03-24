Task: {{task_name}}
Task level: {{task_level}}. Protocol: {{protocol_id}}.
Read the sampled frames in chronological order and describe the events over time.
Question: {{question}}
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
In each segment, `start_sampled` and `end_sampled` are sampled-frame indices.
Return exactly one JSON object and nothing else.
{{output_contract}}
