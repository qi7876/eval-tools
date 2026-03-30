You will receive sampled video frames in chronological order.

Use the frames to answer why the asked result happened.
Depending on the question, the result may be a win, loss, ranking, lead change, failure, or another competition outcome.
Your answer should identify the main cause shown in the video, not merely restate the result.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `text` should directly state the main cause of the asked result.
- `text` may summarize the key causal chain, but the main cause must be clear.
