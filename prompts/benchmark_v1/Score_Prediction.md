You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Use the ranking, score, and other visible game-state information in the sampled inputs to answer the question.
Depending on the question, the answer may involve ranking, score, or another game-status conclusion.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `text` should answer the question using the visible game-state information.
