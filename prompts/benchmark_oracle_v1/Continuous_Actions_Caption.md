You will receive sampled video frames in chronological order.

Describe the target athlete's actions over time.

Question:
{{question}}

The target athlete's position is already known in some sampled frames. Use the known positions below only to identify the correct target athlete across frames. The task is still to describe that target athlete's actions over time:

{{oracle_tracking_explanation}}

```json
{{oracle_tracking_json}}
```

You do not need to output tracking boxes.

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- Each segment should describe what the target athlete is doing during that time span.
- In each segment, `start_sampled` and `end_sampled` must be sampled-frame indices.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
