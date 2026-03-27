Task: {{task_name}}
Task level: {{task_level}}. Protocol: {{protocol_id}}.
Ground the target action to the correct sampled time window.
Question: {{question}}
OracleTrack input for {{oracle_tracking_subject}}:
- {{oracle_tracking_explanation}}
- The GT tracking rows below refer to the question subject.
```json
{{oracle_tracking_json}}
```
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
`time_window_sampled` uses sampled-frame indices.
The GT tracking above already identifies the subject. Do not output tracking boxes.
Return exactly one JSON object and nothing else.
{{output_contract}}
