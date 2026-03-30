You will receive sampled video frames in chronological order.
You will also receive previous question-answer messages in the conversation history.

Use the frames and the previous question-answer messages as reference context to answer the current question.
The current question asks for the spatial conclusion required by that question, often from a specified viewpoint, observer position, or imagined camera angle.
Depending on the question, your answer may need to describe the target's position, movement trajectory, spatial relation, formation, or how the motion appears from that viewpoint.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- `text` should directly answer the current spatial question from the viewpoint requested in the question.
