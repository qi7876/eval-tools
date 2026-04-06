You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Describe the target athlete's actions over time.
The sampled inputs may include earlier history outside the question-relevant interval.
The question-relevant interval in sampled-frame indices is {{question_relevant_sampled_interval}}.
Focus the action description on that interval. Frames outside it are background/history context only.

Question:
{{question}}

The target athlete's position is already known in some sampled frames. Use the known positions below only to identify the correct target athlete across frames. The task is still to describe that target athlete's actions over time:

Each row already gives the target athlete's known location in the normalized_1000 coordinate system, where `(0, 0)` is the top-left corner of the frame and `(1000, 1000)` is the bottom-right corner. Each row uses `frame_sampled` as a sampled-frame index and `bbox_mot` as `[left, top, width, height]`. Use these known boxes only to identify the target athlete. You only need to describe action segments.

```json
{{oracle_tracking_json}}
```

The target athlete has already been highlighted with GT tracking boxes directly on the sampled inputs. Use the highlighted boxes only to identify the target athlete. You only need to describe action segments.

You do not need to output tracking boxes.

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"segments": [{"start_sampled": 0, "end_sampled": 3, "text": "..."}]}
- Each segment should describe what the target athlete is doing during that time span.
- In each segment, `start_sampled` and `end_sampled` must be sampled-frame indices.
- Any `start_sampled` and `end_sampled` you output must use the full sampled sequence index space, not an index local to the question-relevant interval.
- There are {{num_sampled_frames}} sampled frames in total.
- Valid sampled frame indices are {{sampled_index_range}}.
