Convert the raw model output into the canonical JSON schema.

Extraction rules:
- Produce exactly one JSON object matching the schema.
- Use only information that explicitly appears in the raw model output.
- Light normalization is allowed:
  - normalize explicit interval expressions such as "frames 2-4", "2 to 4", or "[2,4]" into `start_sampled` and `end_sampled`
  - normalize explicit tracking rows or coordinate strings into `frame_sampled` and `bbox_mot`
  - map obvious field aliases such as "track", "boxes", "trajectory", "segments", or "actions" into the canonical fields
  - within each segment, map obvious field aliases such as "action", "caption", "description", or "text" into `text`
- If the raw model output contains reasoning plus a final structured answer, extract the final answer.
- If multiple candidate segment or tracking results appear, prefer the last one presented as the final answer.
- Do not infer unseen segments, frame indices, or boxes.
- Each segment must contain `start_sampled`, `end_sampled`, and `text`.
- Each tracking row must contain `frame_sampled` and `bbox_mot`.
- `bbox_mot` must be formatted as normalized_1000 `[left, top, width, height]`.
- In this coordinate system, `(0, 0)` is the top-left corner and `(1000, 1000)` is the bottom-right corner.

Raw model output:
{{raw_output}}

Return JSON only. Use this schema exactly:
{
  "segments": [
    {
      "start_sampled": 0,
      "end_sampled": 3,
      "text": ""
    }
  ],
  "tracking": [
    {
      "frame_sampled": 0,
      "bbox_mot": [0, 0, 100, 100]
    }
  ]
}
