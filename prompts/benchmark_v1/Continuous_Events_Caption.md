You will receive sampled video frames in chronological order.

Describe the key events relevant to the question over time.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- Each segment should describe an event and when it happens.
- In each segment, `start_sampled` and `end_sampled` must be sampled-frame indices.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
