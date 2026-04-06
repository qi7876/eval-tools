You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

You will also receive a target description that specifies an action or event involving a particular subject.
Use the sampled inputs to find when the described action or event happens.

Target Description:
{{question}}

The target subject has already been highlighted with GT tracking boxes directly on the sampled inputs. Use the highlighted boxes only to identify the target subject. You only need to predict `time_window_sampled`.

You do not need to output tracking boxes.

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"time_window_sampled": [0, 4]}
- `time_window_sampled` should cover when the described action or event happens.
- `time_window_sampled` must use sampled-frame indices.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
