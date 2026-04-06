You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

You will also receive a target description that specifies an action or event involving a particular subject.
Use the sampled inputs to find when the described action or event happens, and track the subject referred to by that target description.

Target Description:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"time_window_sampled": [0, 4], "tracking": [{"frame_sampled": 0, "bbox_mot": [left, top, width, height]}]}
- `time_window_sampled` should cover when the described action or event happens.
- `time_window_sampled` must use sampled-frame indices.
- Each tracking row should localize the subject referred to by the target description.
- Each tracking row must use `frame_sampled` as a sampled-frame index.
- Each tracking row must use normalized_1000 `bbox_mot = [left, top, width, height]`.
- In this coordinate system, the top-left corner of the frame is `(0, 0)` and the bottom-right corner is `(1000, 1000)`.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
