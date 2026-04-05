You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Use the scoreboard shown in the sampled inputs to answer every scoreboard detail requested in the question.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `text` should include all scoreboard information requested by the question.
