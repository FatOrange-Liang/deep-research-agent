from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class EvidenceReranker(Protocol):
    """
    Pluggable claim-passage reranking interface.

    A reranker scores already-retrieved passages. The score is a
    relevance/ranking score, not a factual-entailment guarantee.
    """

    def score(
        self,
        claim: str,
        passages: Sequence[str],
    ) -> list[float]:
        ...


@dataclass(frozen=True)
class RerankedPassage:
    index: int
    score: float


class MultilingualEvidenceReranker:
    """
    Lazy-loaded multilingual cross-encoder evidence reranker.

    The model is not imported/downloaded/loaded during package import or
    object construction. The first call to ``score`` loads torch,
    transformers, tokenizer, and model weights.

    This keeps ordinary unit tests and non-reranking agent runs light.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("'model_name' must be a non-empty string.")
        if max_length < 1:
            raise ValueError("'max_length' must be positive.")
        if batch_size < 1:
            raise ValueError("'batch_size' must be positive.")

        self.model_name = model_name
        self.requested_device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        if self._device is None:
            return self.requested_device
        return str(self._device)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "Semantic reranking requires 'torch' and 'transformers'. "
                "Install the optional reranker dependencies first."
            ) from exc

        device_name = self.requested_device
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"

        device = torch.device(device_name)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        )
        model.to(device)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def score(
        self,
        claim: str,
        passages: Sequence[str],
    ) -> list[float]:
        if not isinstance(claim, str):
            raise TypeError("'claim' must be a string.")

        passages = list(passages)
        if not passages:
            return []

        if not all(isinstance(passage, str) for passage in passages):
            raise TypeError("All passages must be strings.")

        self._ensure_loaded()

        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        device = self._device

        assert torch is not None
        assert tokenizer is not None
        assert model is not None
        assert device is not None

        scores: list[float] = []

        for start in range(
            0,
            len(passages),
            self.batch_size,
        ):
            batch = passages[
                start:start + self.batch_size
            ]

            inputs = tokenizer(
                [claim] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = model(**inputs)

            logits = outputs.logits

            if logits.ndim == 2:
                if logits.shape[1] == 1:
                    batch_scores = logits[:, 0]
                else:
                    batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)

            scores.extend(
                float(value)
                for value in batch_scores.detach().cpu()
            )

        if len(scores) != len(passages):
            raise ValueError(
                "Reranker returned an invalid number of scores."
            )

        return scores

    def rerank(
        self,
        claim: str,
        passages: Sequence[str],
    ) -> list[RerankedPassage]:
        scores = self.score(
            claim,
            passages,
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        return [
            RerankedPassage(
                index=index,
                score=score,
            )
            for index, score in ranked
        ]
