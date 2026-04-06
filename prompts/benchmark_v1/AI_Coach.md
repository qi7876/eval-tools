You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.

Identify the player's mistakes relevant to the question using what is visible in the sampled inputs.
Point out the actual mistakes. Do not turn the answer into general advice or improvement suggestions.
The sampled inputs may include earlier history outside the question-relevant interval.
The question-relevant interval in sampled-frame indices is {{question_relevant_sampled_interval}}.
Focus your answer on that interval. Frames outside it are background/history context only.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"text": "..."}
- `text` should state the player's mistakes asked about in the question.
