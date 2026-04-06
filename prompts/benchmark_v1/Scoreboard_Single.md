You will receive one sampled video frame.

Use the scoreboard shown in that frame to answer the question.
Also output one bounding box for the entire scoreboard. The box should cover the full scoreboard, not only the small region that contains the needed text.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"text": "...", "bbox": [x1, y1, x2, y2]}
- `text` should answer the question using the scoreboard content.
- `bbox` must cover the whole scoreboard.
- `bbox` must use normalized_1000 corner coordinates `[x1, y1, x2, y2]`.
- In this coordinate system, the top-left corner of the frame is `(0, 0)` and the bottom-right corner is `(1000, 1000)`.
