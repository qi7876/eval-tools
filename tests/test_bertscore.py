import sys
from types import SimpleNamespace
import warnings

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


def test_quiet_bertscore_model_load_restores_transformers_verbosity(monkeypatch):
    from transformers.utils import logging as hf_logging

    calls: list[object] = []
    monkeypatch.setattr(hf_logging, "get_verbosity", lambda: 30)
    monkeypatch.setattr(hf_logging, "set_verbosity_error", lambda: calls.append("error"))
    monkeypatch.setattr(hf_logging, "set_verbosity", lambda value: calls.append(("restore", value)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with experiments._quiet_bertscore_model_load():
            calls.append("inside")
            warnings.warn(
                "Some weights of FakeModel were not used when initializing FakeEncoder",
                UserWarning,
            )
            warnings.warn("keep this warning", UserWarning)

    assert calls == ["error", "inside", ("restore", 30)]
    assert [str(item.message) for item in caught] == ["keep this warning"]


def test_preload_bertscore_baseline_uses_writable_copy(monkeypatch, tmp_path):
    copy_arguments: list[bool] = []

    class FakeRow:
        def __init__(self, values):
            self._values = values

        def to_numpy(self, *, copy: bool = False):
            copy_arguments.append(copy)
            return list(self._values)

    class FakeILoc:
        def __getitem__(self, index: int):
            if index != 1:  # pragma: no cover - defensive
                raise AssertionError(f"unexpected row index {index}")
            return FakeRow([1.0, 0.3, 0.4])

    class FakeFrame:
        iloc = FakeILoc()

        def to_numpy(self, *, copy: bool = False):
            copy_arguments.append(copy)
            return [[0.0, 0.1, 0.2], [1.0, 0.3, 0.4]]

    class FakeTensor:
        def __init__(self, values, dtype) -> None:
            self.values = values
            self.dtype = dtype
            self.unsqueeze_dim: int | None = None

        def unsqueeze(self, dim: int):
            self.unsqueeze_dim = dim
            return self

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def tensor(values, dtype=None):
            return FakeTensor(values, dtype)

    baseline_path = tmp_path / "baseline.csv"
    baseline_path.write_text("layer,v1,v2\n0,0.1,0.2\n1,0.3,0.4\n", encoding="utf-8")

    scorer_module = SimpleNamespace(
        pd=SimpleNamespace(read_csv=lambda path: FakeFrame()),
        torch=FakeTorch(),
    )
    monkeypatch.setattr(experiments.importlib, "import_module", lambda name: scorer_module)

    scorer = SimpleNamespace(
        rescale_with_baseline=True,
        _baseline_vals=None,
        baseline_path=str(baseline_path),
        model_type="microsoft/deberta-xlarge-mnli",
        lang="en",
        all_layers=False,
        num_layers=1,
    )

    experiments._preload_bertscore_baseline(scorer)

    assert copy_arguments == [True]
    assert scorer._baseline_vals.values == [0.3, 0.4]
    assert scorer._baseline_vals.dtype == "float32"


def test_get_bertscore_scorer_clips_tokenizer_model_max_length(monkeypatch):
    class FakeTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 1000000000000000019884624838656
            self.init_kwargs: dict[str, int] = {}
            self.max_len = self.model_max_length

    class FakeModel:
        config = SimpleNamespace(max_position_embeddings=512)

    class FakeBERTScorer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self._tokenizer = FakeTokenizer()
            self._model = FakeModel()
            self.rescale_with_baseline = False
            self._baseline_vals = None

    experiments._get_bertscore_scorer.cache_clear()
    monkeypatch.setitem(
        sys.modules,
        "bert_score",
        SimpleNamespace(BERTScorer=FakeBERTScorer),
    )

    scorer = experiments._get_bertscore_scorer()

    assert scorer._tokenizer.model_max_length == 512
    assert scorer._tokenizer.init_kwargs["model_max_length"] == 512
    assert scorer._tokenizer.max_len == 512
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
