You need to judge whether a model prediction should be counted as correct for one question.

This question asks for information read from a scoreboard. The predicted team or player identity, score, ranking, or other queried scoreboard information must match the reference.

Judge by these principles:
- Only use information explicitly stated in the model prediction.
- Do not infer missing details from the question, the reference answer, common sense, or likely model intent.
- Do not repair, complete, or reinterpret an incomplete or malformed prediction.
- Accept valid paraphrases when the meaning is clearly preserved.
- If any core queried fact is incorrect, missing, contradictory, or ambiguous, fail it.

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
{
  "correctness": 1,
  "completeness": 1,
  "faithfulness": 1,
  "final_pass": 1,
  "confidence": "high",
  "brief_reason": "Prediction matches the reference answer with acceptable paraphrasing."
}
