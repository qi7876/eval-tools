import sys
from types import SimpleNamespace

from omnichain_eval import experiments
from omnichain_eval.schema import EvaluationRecord


def _record(
    sample_id: str,
    *,
    candidate: str | None,
    reference: str | None,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task_name="Temporal_Causal",
        video_key="video/sample",
        protocol_id="main",
        structured_prediction={"text": candidate} if candidate else None,
        structuring_errors=[],
        structuring_warnings=[],
        component_metrics={},
        component_pass={},
        task_pass=0,
        bertscore_candidate=candidate,
        bertscore_reference=reference,
    )


def test_resolve_bertscore_tokenizer_max_length_clips_huge_tokenizer_limit():
    assert (
        experiments._resolve_bertscore_tokenizer_max_length(
            1000000000000000019884624838656,
            512,
        )
        == 512
    )
    assert experiments._resolve_bertscore_tokenizer_max_length(256, 512) == 256
    assert experiments._resolve_bertscore_tokenizer_max_length(None, 512) == 512
    assert experiments._resolve_bertscore_tokenizer_max_length(None, None) is None


def test_get_bertscore_scorer_clips_tokenizer_model_max_length(monkeypatch):
    class FakeTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 1000000000000000019884624838656
            self.init_kwargs: dict[str, int] = {}

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=512)

    class FakeBERTScorer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self._tokenizer = FakeTokenizer()
            self._model = FakeModel()

    experiments._get_bertscore_scorer.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "bert_score",
        SimpleNamespace(BERTScorer=FakeBERTScorer),
    )

    scorer = experiments._get_bertscore_scorer()

    assert scorer._tokenizer.model_max_length == 512
    assert scorer._tokenizer.init_kwargs["model_max_length"] == 512
    assert scorer.kwargs == {
        "model_type": experiments.BERTSCORE_MODEL,
        "lang": "en",
        "rescale_with_baseline": True,
        "idf": False,
        "use_fast_tokenizer": False,
    }

    experiments._get_bertscore_scorer.cache_clear()


def test_compute_bertscore_populates_f1_for_eligible_records(monkeypatch):
    class FakeTensor:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    class FakeScorer:
        def __init__(self) -> None:
            self.calls: list[dict[str, list[str]]] = []

        def score(self, *, cands: list[str], refs: list[str]):
            self.calls.append({"cands": cands, "refs": refs})
            return None, None, FakeTensor([0.25, 0.75])

    scorer = FakeScorer()
    monkeypatch.setattr(experiments, "_get_bertscore_scorer", lambda: scorer)

    eligible_a = _record("sample-a", candidate="prediction a", reference="reference a")
    ineligible = _record("sample-b", candidate=None, reference="reference b")
    eligible_c = _record("sample-c", candidate="prediction c", reference="reference c")

    experiments.compute_bertscore([eligible_a, ineligible, eligible_c])

    assert scorer.calls == [
        {
            "cands": ["prediction a", "prediction c"],
            "refs": ["reference a", "reference c"],
        }
    ]
    assert eligible_a.component_metrics["bertscore_f1"] == 0.25
    assert "bertscore_f1" not in ineligible.component_metrics
    assert eligible_c.component_metrics["bertscore_f1"] == 0.75
