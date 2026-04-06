You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Describe the target athlete's actions over time and provide the target athlete's tracking boxes.
The sampled inputs may include earlier history outside the question-relevant interval.
The question-relevant interval in sampled-frame indices is {{question_relevant_sampled_interval}}.
Focus the action description on that interval. Frames outside it are background/history context only.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"segments": [{"start_sampled": 0, "end_sampled": 3, "text": "..."}], "tracking": [{"frame_sampled": 0, "bbox_mot": [left, top, width, height]}]}
- Each segment should describe what the target athlete is doing during that time span.
- In each segment, `start_sampled` and `end_sampled` must be sampled-frame indices.
- Each tracking row should localize that same target athlete.
- In each tracking row, `frame_sampled` must be a sampled-frame index.
- In each tracking row, `bbox_mot` must use normalized_1000 coordinates `[left, top, width, height]`.
- Any `start_sampled`, `end_sampled`, and `frame_sampled` you output must use the full sampled sequence index space, not an index local to the question-relevant interval.
- In this coordinate system, the top-left corner of the frame is `(0, 0)` and the bottom-right corner is `(1000, 1000)`.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
