You need to judge whether a model prediction should be counted as correct for one question.

This question asks why a result happened. The asked result may be a win, loss, ranking, lead change, failure, or another competition outcome. Pass only if the predicted main cause matches the reference. A side effect, consequence, or loosely related event must not be accepted as the main cause.

Judge by these principles:
- Only use information explicitly stated in the model prediction.
- Do not infer missing details from the question, the reference answer, common sense, or likely model intent.
- Do not repair, complete, or reinterpret an incomplete or malformed prediction.
- Accept valid paraphrases when the meaning is clearly preserved.
- If the main cause is wrong, replaced by a consequence, missing, contradictory, or ambiguous, fail it.
- If the prediction only restates the result without giving its cause, fail it.

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
