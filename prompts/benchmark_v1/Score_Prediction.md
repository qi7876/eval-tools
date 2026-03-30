You will receive sampled video frames in chronological order.

Use the ranking, score, and other visible game-state information in the frames to answer the question.
Depending on the question, the answer may involve ranking, score, or another game-status conclusion.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `text` should answer the question using the visible game-state information.
