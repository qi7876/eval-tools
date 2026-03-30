You need to judge whether a model prediction should be counted as correct for one question.

This question asks the model to identify the player's mistakes. The reference answer defines the mistake points that must be recognized. The prediction does not need to match the reference wording, but it must clearly point out the same underlying mistakes.

Pass the prediction if it satisfies all of the following:
- it identifies all core mistake points stated in the reference answer,
- it describes the same mistakes even if the wording is different,
- it does not introduce additional mistake points that are not in the reference answer.

Judge by these principles:
- Only use information explicitly stated in the model prediction.
- Do not infer missing details from the question, the reference answer, common sense, or likely model intent.
- Do not repair, complete, or reinterpret an incomplete or malformed prediction.
- Accept valid paraphrases when they point to the same underlying mistake.
- Do not treat a vague symptom, consequence, or general weakness as correct unless it clearly identifies the referenced mistake itself.
- If the prediction also gives improvement advice, ignore that advice unless it conflicts with or replaces the mistake identification.
- Do not reward verbosity, tone, or confidence.
- Fail the prediction if it misses any referenced mistake point, adds extra mistake points, stays too vague to map to the referenced mistake, or conflicts with the reference answer.

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
