You will receive one sampled video frame.

Use that frame to find the two objects named in the question, answer their spatial relationship, and output one bounding box for each of those two objects.

Question:
{{question}}

Output requirements:
- Return exactly one JSON object and nothing else.
- Use this format exactly:
  {{output_contract}}
- In `text`, describe the spatial relationship from the camera viewpoint and prefer these terms:
  - `left` / `right`: left or right in the image plane
  - `above` / `below`: clearly higher or lower in the 2D image plane
  - `over`: A is completely above B
  - `in front of` / `behind`: closer to or farther from the camera; this is about depth to the camera, not who is leading in the play
- If one object occludes the other, the occluding object is in front of the occluded object.
- Each object must be written as `{"label": "...", "bbox": [x1, y1, x2, y2]}` in the normalized_1000 coordinate system.
- In this coordinate system, the top-left corner of the frame is `(0, 0)` and the bottom-right corner is `(1000, 1000)`.
- Use exactly these labels in `objects[].label`: {{required_object_labels_json}}.
- Each label must refer to the corresponding object asked about in the question.
- The order of entries in `objects` does not matter.
