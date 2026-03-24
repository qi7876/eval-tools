Task: {{task_name}}
Task level: {{task_level}}. Protocol: {{protocol_id}}.
Ground the target action to the correct sampled time window and tracking boxes.
Question: {{question}}
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
`time_window_sampled` uses sampled-frame indices.
Each tracking row uses `frame_sampled` as a sampled-frame index and `bbox_mot` as `[left, top, width, height]`.
Oracle track enabled for this run: {{oracle_track_enabled}}.
Return exactly one JSON object and nothing else.
{{output_contract}}
