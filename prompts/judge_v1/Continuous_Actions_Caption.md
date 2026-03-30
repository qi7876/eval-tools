You need to judge whether a model prediction should be counted as correct for one question.

This question asks for action descriptions over time. Judge the described actions and their temporal alignment together. The reference answer is a judging reference, not an exact segment template. The prediction may use slightly different time spans or a different split/merge of segments, as long as the key action meaning, temporal order, and rough timing are preserved.

Judge by these principles:
- Only use information explicitly stated in the model prediction.
- Do not infer missing details from the question, the reference answer, common sense, or likely model intent.
- Do not repair, complete, or reinterpret an incomplete or malformed prediction.
- Accept valid paraphrases when the meaning is clearly preserved.
- Do not require exact interval boundaries or one-to-one segment matching if the semantic content, order, and rough temporal correspondence are still correct.
- If a key action is missing, mislabeled, badly timed, or placed in the wrong order, fail it.

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
