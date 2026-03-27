Task: {{task_name}}
Task level: {{task_level}}. Protocol: {{protocol_id}}.
Identify the two required regions and describe their spatial relationship.
Question: {{question}}
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
Each object must be returned as `{"label": "...", "bbox": [xtl, ytl, xbr, ybr]}`.
Use exactly these labels in `objects[].label`: {{required_object_labels_json}}.
The order of entries in `objects` does not matter.
Return exactly one JSON object and nothing else.
{{output_contract}}
