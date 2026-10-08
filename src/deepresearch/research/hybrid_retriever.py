from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .claim_anchors import normalize_text
from .dense_retriever import (
    DenseEvidenceRetriever,
    EmbeddingBackend,
)
from .manifest import (
    ClaimUnit,
    EvidenceProposal,
)
from .reranker import EvidenceReranker
from .retriever import EvidenceCandidateRetriever
from .state import ResearchState


CandidateKey = tuple[str, str]


@dataclass(frozen=True)
class HybridEvidenceCandidate:
    """
    Evidence candidate produced by lexical+dense RRF fusion and
    optional semantic reranking.

    ``reranker_score`` is a relevance score. It is not an entailment
    or factual-support probability.
    """

    claim_id: str
    source_id: str
    quote: str

    rrf_score: float
    rrf_rank: int

    lexical_rank: int | None
    dense_rank: int | None

    reranker_score: float | None = None

    def to_proposal(self) -> EvidenceProposal:
        return EvidenceProposal(
            claim_id=self.claim_id,
            source_id=self.source_id,
            evidence_quote=self.quote,
        )


class HybridEvidenceRetriever:
    """
    Production evidence retrieval pipeline:

        lexical retrieval
              +
        dense multilingual retrieval
              |
              v
        Reciprocal Rank Fusion
              |
          Top candidate_top_k
              |
        optional semantic reranker
              |
          Top output_top_k

    Retrieval remains restricted to the source IDs already permitted by
    each ClaimUnit. The reranker only reorders retrieved source excerpts;
    it never generates evidence text.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingBackend,
        reranker: EvidenceReranker | None = None,
        candidate_top_k: int = 10,
        output_top_k: int = 3,
        rrf_k: int = 60,
        min_quote_chars: int = 24,
        max_quote_chars: int = 450,
        min_lexical_score: float = 0.0,
        min_similarity: float = -1.0,
    ) -> None:
        if candidate_top_k < 1:
            raise ValueError("'candidate_top_k' must be positive.")
        if output_top_k < 1:
            raise ValueError("'output_top_k' must be positive.")
        if output_top_k > candidate_top_k:
            raise ValueError(
                "'output_top_k' cannot exceed 'candidate_top_k'."
            )
        if rrf_k < 1:
            raise ValueError("'rrf_k' must be positive.")

        self.candidate_top_k = candidate_top_k
        self.output_top_k = output_top_k
        self.rrf_k = rrf_k
        self.reranker = reranker

        self.lexical_retriever = EvidenceCandidateRetriever(
            top_k_per_claim=candidate_top_k,
            min_quote_chars=min_quote_chars,
            max_quote_chars=max_quote_chars,
            min_lexical_score=min_lexical_score,
        )

        self.dense_retriever = DenseEvidenceRetriever(
            embedder=embedder,
            top_k_per_claim=candidate_top_k,
            min_similarity=min_similarity,
            min_quote_chars=min_quote_chars,
            max_quote_chars=max_quote_chars,
        )

    @staticmethod
    def _key(
        source_id: str,
        quote: str,
    ) -> CandidateKey:
        return (
            source_id,
            normalize_text(quote),
        )

    def _fuse_one_claim(
        self,
        *,
        claim: ClaimUnit,
        lexical_candidates: Sequence,
        dense_candidates: Sequence,
    ) -> tuple[HybridEvidenceCandidate, ...]:
        lexical_rank: dict[CandidateKey, int] = {}
        dense_rank: dict[CandidateKey, int] = {}
        metadata: dict[CandidateKey, tuple[str, str]] = {}

        for rank, candidate in enumerate(
            lexical_candidates,
            start=1,
        ):
            key = self._key(
                candidate.source_id,
                candidate.quote,
            )
            lexical_rank.setdefault(
                key,
                rank,
            )
            metadata.setdefault(
                key,
                (
                    candidate.source_id,
                    candidate.quote,
                ),
            )

        for rank, candidate in enumerate(
            dense_candidates,
            start=1,
        ):
            key = self._key(
                candidate.source_id,
                candidate.quote,
            )
            dense_rank.setdefault(
                key,
                rank,
            )
            metadata.setdefault(
                key,
                (
                    candidate.source_id,
                    candidate.quote,
                ),
            )

        if not metadata:
            return ()

        rrf_scores: dict[CandidateKey, float] = {}

        for key, rank in lexical_rank.items():
            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        for key, rank in dense_rank.items():
            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        fused_keys = sorted(
            metadata,
            key=lambda key: (
                -rrf_scores[key],
                key[0],
                key[1],
            ),
        )[:self.candidate_top_k]

        fused: list[HybridEvidenceCandidate] = []

        for rank, key in enumerate(
            fused_keys,
            start=1,
        ):
            source_id, quote = metadata[key]

            fused.append(
                HybridEvidenceCandidate(
                    claim_id=claim.claim_id,
                    source_id=source_id,
                    quote=quote,
                    rrf_score=rrf_scores[key],
                    rrf_rank=rank,
                    lexical_rank=lexical_rank.get(
                        key
                    ),
                    dense_rank=dense_rank.get(
                        key
                    ),
                    reranker_score=None,
                )
            )

        if self.reranker is None:
            return tuple(
                fused[:self.output_top_k]
            )

        scores = self.reranker.score(
            claim.text,
            [
                candidate.quote
                for candidate in fused
            ],
        )

        if len(scores) != len(fused):
            raise ValueError(
                "Reranker returned an invalid number of scores."
            )

        rescored = [
            HybridEvidenceCandidate(
                claim_id=candidate.claim_id,
                source_id=candidate.source_id,
                quote=candidate.quote,
                rrf_score=candidate.rrf_score,
                rrf_rank=candidate.rrf_rank,
                lexical_rank=candidate.lexical_rank,
                dense_rank=candidate.dense_rank,
                reranker_score=float(score),
            )
            for candidate, score in zip(
                fused,
                scores,
            )
        ]

        rescored.sort(
            key=lambda candidate: (
                -float(
                    candidate.reranker_score
                ),
                candidate.rrf_rank,
                candidate.source_id,
                normalize_text(
                    candidate.quote
                ),
            )
        )

        return tuple(
            rescored[:self.output_top_k]
        )

    def retrieve(
        self,
        *,
        claims: Sequence[ClaimUnit],
        state: ResearchState,
    ) -> dict[
        str,
        tuple[HybridEvidenceCandidate, ...],
    ]:
        claims = tuple(claims)

        lexical = self.lexical_retriever.retrieve(
            claims=claims,
            state=state,
        )

        dense = self.dense_retriever.retrieve(
            claims=claims,
            state=state,
        )

        output: dict[
            str,
            tuple[HybridEvidenceCandidate, ...],
        ] = {}

        for claim in claims:
            output[claim.claim_id] = (
                self._fuse_one_claim(
                    claim=claim,
                    lexical_candidates=lexical.get(
                        claim.claim_id,
                        (),
                    ),
                    dense_candidates=dense.get(
                        claim.claim_id,
                        (),
                    ),
                )
            )

        return output
