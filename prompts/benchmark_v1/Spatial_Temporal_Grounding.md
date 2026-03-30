You will receive sampled video frames in chronological order.

You will also receive a target description that specifies an action or event involving a particular subject.
Use the frames to find when the described action or event happens, and track the subject referred to by that target description.

Target Description:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `time_window_sampled` should cover when the described action or event happens.
- `time_window_sampled` must use sampled-frame indices.
- Each tracking row should localize the subject referred to by the target description.
- Each tracking row must use `frame_sampled` as a sampled-frame index.
- Each tracking row must use `bbox_mot` as `[left, top, width, height]`.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
