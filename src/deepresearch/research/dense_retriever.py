
from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Protocol, Sequence

from .manifest import (
    ClaimUnit,
    EvidenceProposal,
)

from .retriever import (
    EvidenceCandidateRetriever,
)

from .claim_anchors import normalize_text
from .state import ResearchState


# =========================================
# 1. Embedding Backend Interface
# =========================================

class EmbeddingBackend(Protocol):
    """
    Pluggable text embedding interface.

    Query and passage encoding are separated because
    some retrieval models require different prefixes.
    """

    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...

    def encode_passages(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...


# =========================================
# 2. Optional Real Multilingual Backend
# =========================================

class MultilingualE5Embedder:
    """
    SentenceTransformer adapter for multilingual E5.

    This class is optional and is not instantiated
    during the offline unit tests.

    Model weights are downloaded when first loaded
    unless they are already available locally.
    """

    def __init__(
        self,
        model_name: str = (
            "intfloat/multilingual-e5-small"
        ),
        device: str | None = None,
    ) -> None:

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )

        except ImportError as exc:

            raise ImportError(
                "Install the optional dependency with: "
                "pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def _encode(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        return self._encode(
            [
                f"query: {text}"
                for text in texts
            ]
        )

    def encode_passages(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:

        return self._encode(
            [
                f"passage: {text}"
                for text in texts
            ]
        )


# =========================================
# 3. Vector Similarity
# =========================================

def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """
    Deterministic cosine similarity.

    Reject mismatched dimensions rather than
    silently producing an invalid score.
    """

    if len(left) != len(right):
        raise ValueError(
            "Embedding dimension mismatch."
        )

    if not left:
        return 0.0

    a = [float(x) for x in left]
    b = [float(x) for x in right]

    if not all(
        math.isfinite(value)
        for value in a + b
    ):
        raise ValueError(
            "Embedding contains non-finite values."
        )

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(y * y for y in b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return max(
        -1.0,
        min(
            1.0,
            dot / (norm_a * norm_b),
        ),
    )


# =========================================
# 4. Dense Evidence Candidate
# =========================================

@dataclass(frozen=True)
class DenseEvidenceCandidate:

    claim_id: str

    source_id: str

    quote: str

    semantic_score: float

    def to_proposal(self) -> EvidenceProposal:

        return EvidenceProposal(
            claim_id=self.claim_id,
            source_id=self.source_id,
            evidence_quote=self.quote,
        )


# =========================================
# 5. Dense Evidence Retriever
# =========================================

class DenseEvidenceRetriever:
    """
    Retrieve source passages through embeddings.

    Rules:

    - Only search sources cited by the claim.
    - Only use sources successfully read in full.
    - Never generate new evidence text.
    - Reuse lexical retriever passage segmentation.
    - Return original source excerpts.
    - Do not interpret cosine similarity as entailment.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingBackend,
        top_k_per_claim: int = 3,
        min_similarity: float = 0.20,
        min_quote_chars: int = 24,
        max_quote_chars: int = 450,
    ) -> None:

        if top_k_per_claim < 1:
            raise ValueError(
                "'top_k_per_claim' must be positive."
            )

        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError(
                "'min_similarity' must be between "
                "-1 and 1."
            )

        if min_quote_chars < 1:
            raise ValueError(
                "'min_quote_chars' must be positive."
            )

        if max_quote_chars < min_quote_chars:
            raise ValueError(
                "Invalid quote length configuration."
            )

        self.embedder = embedder

        self.top_k_per_claim = top_k_per_claim

        self.min_similarity = min_similarity

        # Both retrievers use exactly the same
        # passage segmentation.
        self.segmenter = EvidenceCandidateRetriever(
            min_quote_chars=min_quote_chars,
            max_quote_chars=max_quote_chars,
        )

    def retrieve(
        self,
        *,
        claims: Sequence[ClaimUnit],
        state: ResearchState,
    ) -> dict[
        str,
        tuple[DenseEvidenceCandidate, ...],
    ]:

        output: dict[
            str,
            tuple[DenseEvidenceCandidate, ...],
        ] = {}

        for claim in claims:

            # ---------------------------------
            # 1. Collect permitted passages
            # ---------------------------------

            passage_records: list[
                tuple[str, str]
            ] = []

            for source_id in dict.fromkeys(
                claim.cited_source_ids
            ):

                record = state.sources.get(
                    source_id
                )

                if record is None:
                    continue

                if not record.read_full_page:
                    continue

                if not record.source.content:
                    continue

                passages = (
                    self.segmenter.extract_passages(
                        record.source.content
                    )
                )

                for passage in passages:

                    passage_records.append(
                        (
                            source_id,
                            passage,
                        )
                    )

            # A claim with no permitted evidence
            # must return no candidates.
            if not passage_records:

                output[claim.claim_id] = ()

                continue

            # ---------------------------------
            # 2. Encode query and passages
            # ---------------------------------

            query_vectors = (
                self.embedder.encode_queries(
                    [claim.text]
                )
            )

            passages = [
                passage
                for _, passage in passage_records
            ]

            passage_vectors = (
                self.embedder.encode_passages(
                    passages
                )
            )

            if len(query_vectors) != 1:
                raise ValueError(
                    "Embedding backend returned an "
                    "invalid number of query vectors."
                )

            if (
                len(passage_vectors)
                != len(passage_records)
            ):
                raise ValueError(
                    "Embedding backend returned an "
                    "invalid number of passage vectors."
                )

            query_vector = query_vectors[0]

            # ---------------------------------
            # 3. Score candidate passages
            # ---------------------------------

            candidates: list[
                DenseEvidenceCandidate
            ] = []

            for (
                source_id,
                passage,
            ), passage_vector in zip(
                passage_records,
                passage_vectors,
            ):

                similarity = cosine_similarity(
                    query_vector,
                    passage_vector,
                )

                if (
                    similarity
                    < self.min_similarity
                ):
                    continue

                candidates.append(
                    DenseEvidenceCandidate(
                        claim_id=claim.claim_id,
                        source_id=source_id,
                        quote=passage,
                        semantic_score=similarity,
                    )
                )

            # ---------------------------------
            # 4. Deterministic ranking
            # ---------------------------------

            candidates.sort(
                key=lambda item: (
                    -item.semantic_score,
                    item.source_id,
                    item.quote,
                )
            )

            # ---------------------------------
            # 5. Deduplication
            # ---------------------------------

            selected: list[
                DenseEvidenceCandidate
            ] = []

            seen: set[
                tuple[str, str]
            ] = set()

            for candidate in candidates:

                key = (
                    candidate.source_id,
                    normalize_text(
                        candidate.quote
                    ),
                )

                if key in seen:
                    continue

                seen.add(key)

                selected.append(
                    candidate
                )

                if (
                    len(selected)
                    >= self.top_k_per_claim
                ):
                    break

            output[claim.claim_id] = tuple(
                selected
            )

        return output
