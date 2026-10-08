from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@dataclass(frozen=True)
class RerankedPassage:
    index: int
    score: float


class MultilingualEvidenceReranker:
    """
    Cross-encoder style multilingual evidence reranker.

    Scores a claim and passage jointly instead of comparing
    independently embedded vectors.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name,
            )
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(
        self,
        claim: str,
        passages: Sequence[str],
    ) -> list[float]:

        if not passages:
            return []

        pairs = [
            [claim, passage]
            for passage in passages
        ]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        logits = outputs.logits

        if logits.ndim == 2:
            if logits.shape[1] == 1:
                scores = logits[:, 0]
            else:
                scores = logits[:, -1]
        else:
            scores = logits.reshape(-1)

        return [
            float(x)
            for x in scores.detach().cpu()
        ]

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