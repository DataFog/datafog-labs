"""Model adapters for evaluation benchmarks."""

from __future__ import annotations

from typing import Iterable

from ..inference.pipeline import PiiPipeline


class ModelAdapter:
    name: str = "base"

    def predict(self, texts: list[str]) -> list[list[dict]]:
        raise NotImplementedError


class DataFogAdapter(ModelAdapter):
    name = "datafog"

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_seq_len: int = 256,
        max_char_len: int = 20,
    ):
        self.pipeline = PiiPipeline.from_pretrained(
            model_path,
            device=device,
            max_seq_len=max_seq_len,
            max_char_len=max_char_len,
        )

    def predict(self, texts: list[str]) -> list[list[dict]]:
        outputs = self.pipeline(texts)
        spans_batch: list[list[dict]] = []
        for entities in outputs:
            spans = [
                {
                    "start": e.start,
                    "end": e.end,
                    "label": e.label,
                }
                for e in entities
            ]
            spans_batch.append(spans)
        return spans_batch


class AllOAdapter(ModelAdapter):
    name = "all_o"

    def predict(self, texts: list[str]) -> list[list[dict]]:
        return [[] for _ in texts]


def get_adapter(name: str, **kwargs) -> ModelAdapter:
    name = name.lower()
    if name in {"datafog", "datafog-pii", "datafog_pii"}:
        return DataFogAdapter(**kwargs)
    if name in {"all_o", "all-o", "allo"}:
        return AllOAdapter()
    raise ValueError(f"Unknown model adapter: {name}")
