You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Describe the target athlete's actions over time.

Question:
{{question}}

{{oracle_visual_explanation}}

You do not need to output tracking boxes.

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- Each segment should describe what the target athlete is doing during that time span.
- In each segment, `start_sampled` and `end_sampled` must be sampled-frame indices.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
