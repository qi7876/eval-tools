Task: {{task_name}}
Task level: {{task_level}}. Protocol: {{protocol_id}}.
Read the sampled frames in chronological order and describe the athlete actions over time.
Question: {{question}}
OracleTrack input for {{oracle_tracking_subject}}:
- {{oracle_tracking_explanation}}
- The GT tracking rows below refer to the question subject.
```json
{{oracle_tracking_json}}
```
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
In each segment, `start_sampled` and `end_sampled` are sampled-frame indices.
The GT tracking above already identifies the subject. Do not output tracking boxes.
Return exactly one JSON object and nothing else.
{{output_contract}}
