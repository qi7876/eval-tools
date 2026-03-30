You will receive sampled video frames in chronological order.

You will also receive a target description that specifies an action or event involving a particular subject.
Use the frames to find when the described action or event happens.

Target Description:
{{question}}

The target subject's position is already known in some sampled frames. Use the known positions below only to identify the subject referred to by the target description across frames:

{{oracle_tracking_explanation}}

```json
{{oracle_tracking_json}}
```

You do not need to output tracking boxes.

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `time_window_sampled` should cover when the described action or event happens.
- `time_window_sampled` must use sampled-frame indices.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
