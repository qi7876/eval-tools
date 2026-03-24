Task: {{task_name}}
Task level: {{task_level}}. Protocol: {{protocol_id}}.
Read the sampled frames in chronological order and describe the athlete actions over time.
Question: {{question}}
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
In each segment, `start_sampled` and `end_sampled` are sampled-frame indices.
In each tracking row, `frame_sampled` is a sampled-frame index and `bbox_mot` uses `[left, top, width, height]`.
Return exactly one JSON object and nothing else.
{{output_contract}}
