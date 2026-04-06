You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

You will also receive a target description that specifies an action or event involving a particular subject.
Use the sampled inputs to find when the described action or event happens.

Target Description:
{{question}}

The target subject's position is already known in some sampled frames. Use the known positions below only to identify the subject referred to by the target description across frames:

Each row already gives the target subject's known location in the normalized_1000 coordinate system, where `(0, 0)` is the top-left corner of the frame and `(1000, 1000)` is the bottom-right corner. Each row uses `frame_sampled` as a sampled-frame index and `bbox_mot` as `[left, top, width, height]`. Use these known boxes only to identify the target subject. You only need to predict `time_window_sampled`.

```json
{{oracle_tracking_json}}
```

You do not need to output tracking boxes.

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"time_window_sampled": [0, 4]}
- `time_window_sampled` should cover when the described action or event happens.
- `time_window_sampled` must use sampled-frame indices.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
