Task: {{task_name}}
Task level: {{task_level}}. Protocol: {{protocol_id}}.
Read the sampled frames in chronological order and answer the current reasoning question.
You can also reference the previous question-answer messages already provided in the conversation history.
Question: {{question}}
Sampled frame count: {{num_sampled_frames}}. Valid sampled frame indices: {{sampled_index_range}}.
Return exactly one JSON object and nothing else.
{{output_contract}}
