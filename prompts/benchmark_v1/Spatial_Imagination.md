You will receive sampled visual inputs in chronological order.
These sampled inputs correspond to approximately {{sampled_video_fps}} fps.
You will also receive previous question-answer messages in the conversation history.

Use the sampled inputs and the previous question-answer messages as reference context to answer the current question.
The sampled inputs may include earlier history outside the question-relevant interval.
The question-relevant interval in sampled-frame indices is {{question_relevant_sampled_interval}}.
Focus the current answer on that interval. Frames outside it are only background/history context.
The current question asks for the spatial conclusion required by that question, often from a specified viewpoint, observer position, or imagined camera angle.
Depending on the question, your answer may need to describe the target's position, movement trajectory, spatial relation, formation, or how the motion appears from that viewpoint.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {"text": "..."}
- `text` should directly answer the current spatial question from the viewpoint requested in the question.
