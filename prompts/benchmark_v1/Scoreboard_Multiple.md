You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Use the scoreboard shown in the sampled inputs to answer every scoreboard detail requested in the question.
The sampled inputs may include earlier history outside the question-relevant interval.
The question-relevant interval in sampled-frame indices is {{question_relevant_sampled_interval}}.
Focus your answer on that interval. Frames outside it are background/history context only.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"text": "..."}
- `text` should include all scoreboard information requested by the question.
