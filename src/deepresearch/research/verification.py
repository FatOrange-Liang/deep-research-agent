from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .citations import (
    CitationValidation,
    validate_citations,
)
from .manifest import (
    ClaimManifest,
    ClaimManifestBuilder,
    ClaimUnit,
    EvidenceProposal,
)
from .retriever import EvidenceCandidateRetriever
from .state import ResearchState


class EvidenceCandidateLike(Protocol):
    """
    Minimal interface required by ResearchVerifier.

    Lexical, dense, hybrid, and reranked candidates can all satisfy this
    protocol without forcing them into one concrete dataclass.
    """

    claim_id: str
    source_id: str
    quote: str

    def to_proposal(self) -> EvidenceProposal:
        ...


class EvidenceRetriever(Protocol):
    """
    Pluggable retrieval interface used by ResearchVerifier.
    """

    def retrieve(
        self,
        *,
        claims: Sequence[ClaimUnit],
        state: ResearchState,
    ) -> dict[
        str,
        tuple[EvidenceCandidateLike, ...],
    ]:
        ...


@dataclass(frozen=True)
class ResearchVerificationReport:
    """
    Complete deterministic structural verification report.

    This report checks citation validity, extracted claim coverage,
    and evidence anchor coverage.

    A semantic reranker may improve candidate ordering, but this report
    still does NOT establish semantic entailment.
    """

    citation_validation: CitationValidation
    manifest: ClaimManifest

    candidates: dict[
        str,
        tuple[EvidenceCandidateLike, ...],
    ]

    @property
    def citations_valid(self) -> bool:
        return self.citation_validation.is_valid

    @property
    def claim_count(self) -> int:
        return self.manifest.claim_count

    @property
    def candidate_count(self) -> int:
        return sum(
            len(items)
            for items in self.candidates.values()
        )

    @property
    def citation_coverage(self) -> float:
        return self.manifest.citation_coverage

    @property
    def anchor_coverage(self) -> float:
        return self.manifest.anchor_coverage

    @property
    def missing_candidate_claim_ids(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            entry.claim.claim_id
            for entry in self.manifest.entries
            if not self.candidates.get(
                entry.claim.claim_id
            )
        )

    @property
    def structural_checks_passed(self) -> bool:
        """
        All extracted claim units have citations and at least one genuine
        source-text anchor.

        This does not mean the claims are factually supported by those
        excerpts.
        """
        return (
            self.citations_valid
            and self.claim_count > 0
            and self.citation_coverage == 1.0
            and self.manifest.all_claim_units_anchored
        )


class ResearchVerifier:
    """
    Compose structural verification modules with a pluggable retriever.

    Pipeline:

        Final Answer
             |
             v
       Citation Validation
             |
             v
       Claim Extraction
             |
             v
       Evidence Retrieval
       (lexical by default; hybrid/reranked optional)
             |
             v
       Evidence Proposals
             |
             v
       Claim Anchor Verification
             |
             v
       Verification Report

    The default remains EvidenceCandidateRetriever for backwards
    compatibility. No external API is required by the default path.
    """

    def __init__(
        self,
        *,
        manifest_builder: ClaimManifestBuilder | None = None,
        retriever: EvidenceRetriever | None = None,
    ) -> None:
        self.manifest_builder = (
            manifest_builder
            if manifest_builder is not None
            else ClaimManifestBuilder()
        )

        self.retriever: EvidenceRetriever = (
            retriever
            if retriever is not None
            else EvidenceCandidateRetriever()
        )

    def verify(
        self,
        *,
        answer: str,
        state: ResearchState,
    ) -> ResearchVerificationReport:
        if not isinstance(answer, str):
            raise TypeError(
                "'answer' must be a string."
            )

        citation_validation = validate_citations(
            answer,
            state.citation_sources,
        )

        claims = (
            self.manifest_builder.extract_claims(
                answer
            )
        )

        candidates = self.retriever.retrieve(
            claims=claims,
            state=state,
        )

        proposals = [
            candidate.to_proposal()
            for candidate_list in candidates.values()
            for candidate in candidate_list
        ]

        manifest = self.manifest_builder.build(
            answer=answer,
            state=state,
            proposals=proposals,
        )

        return ResearchVerificationReport(
            citation_validation=citation_validation,
            manifest=manifest,
            candidates=candidates,
        )
