You will receive sampled video frames in chronological order.

Describe the target athlete's actions over time and provide the target athlete's tracking boxes.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- Each segment should describe what the target athlete is doing during that time span.
- In each segment, `start_sampled` and `end_sampled` must be sampled-frame indices.
- Each tracking row should localize that same target athlete.
- In each tracking row, `frame_sampled` must be a sampled-frame index.
- In each tracking row, `bbox_mot` must use `[left, top, width, height]`.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
