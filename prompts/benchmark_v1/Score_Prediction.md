You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Use the ranking, score, and other visible game-state information in the sampled inputs to answer the question.
Depending on the question, the answer may involve ranking, score, or another game-status conclusion.
The sampled inputs may include earlier history outside the question-relevant interval.
The question-relevant interval in sampled-frame indices is {{question_relevant_sampled_interval}}.
Focus your answer on that interval. Frames outside it are background/history context only.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"text": "..."}
- `text` should answer the question using the visible game-state information.
