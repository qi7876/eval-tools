You need to judge whether a model prediction should be counted as correct for one question.

This question asks for the spatial relationship between the queried objects. The predicted relation must match the reference relation for those objects under these camera-view definitions:
- `left` / `right`: left or right in the image plane
- `above` / `below`: clearly higher or lower in the 2D image plane
- `over`: A is completely above B
- `in front of` / `behind`: closer to or farther from the camera; this is about depth to the camera, not who is leading in the play
- If one object occludes the other, the occluding object is in front of the occluded object.

Judge by these principles:
- Only use information explicitly stated in the model prediction.
- Do not infer missing details from the question, the reference answer, common sense, or likely model intent.
- Do not repair, complete, or reinterpret an incomplete or malformed prediction.
- Accept valid paraphrases when the meaning is clearly preserved.
- If the predicted relation is wrong, incomplete, contradictory, ambiguous, or uses the wrong spatial meaning under these definitions, fail it.

Question:
{{question_text}}

Reference answer:
{{reference_payload_json}}

Model prediction:
{{prediction_payload_json}}

Return exactly one JSON object and nothing else.

Output requirements:
- `correctness`, `completeness`, `faithfulness`, and `final_pass` must each be exactly `0` or `1`.
- `confidence` must be one of `"low"`, `"medium"`, `"high"`.
- `brief_reason` must be a short single-sentence explanation.
- Use exactly this JSON schema:
{{required_json_schema_json}}
