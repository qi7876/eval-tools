You will receive one sampled video frame.

Use the scoreboard shown in that frame to answer the question.
Also output one bounding box for the entire scoreboard. The box should cover the full scoreboard, not only the small region that contains the needed text.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `text` should answer the question using the scoreboard content.
- `bbox` must cover the whole scoreboard.
- `bbox` must use `[xtl, ytl, xbr, ybr]`.
