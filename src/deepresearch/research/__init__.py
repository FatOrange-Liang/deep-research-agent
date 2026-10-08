from .citations import (
    CitationValidation,
    collect_sources,
    extract_citation_ids,
    extract_source_like_ids,
    validate_citations,
)

from .sources import (
    Source,
    make_source_id,
)

from .guard import (
    CitationGuard,
    CitationGuardResult,
)

from .state import (
    EvidenceRecord,
    ResearchState,
)

from .policy import (
    EvidenceAction,
    EvidenceDecision,
    EvidencePolicy,
    EvidencePolicyConfig,
)

from .controller import (
    ResearchControl,
    ResearchController,
)

from .authority import SourceAuthorityPolicy

from .claim_anchors import (
    ClaimEvidence,
    ClaimAnchorCheck,
    ClaimAnchorReport,
    ClaimAnchorVerifier,
)

from .manifest import (
    ClaimUnit,
    EvidenceProposal,
    ClaimManifestEntry,
    ClaimManifest,
    ClaimManifestBuilder,
)

from .retriever import (
    EvidenceCandidate,
    EvidenceCandidateRetriever,
)

from .verification import (
    ResearchVerifier,
    ResearchVerificationReport,
)

from .dense_retriever import (
    EmbeddingBackend,
    MultilingualE5Embedder,
    DenseEvidenceCandidate,
    DenseEvidenceRetriever,
)

__all__ = [
    "Source",
    "make_source_id",
    "CitationValidation",
    "collect_sources",
    "extract_citation_ids",
    "validate_citations",
    "CitationGuard",
    "CitationGuardResult",
    "extract_source_like_ids",
    "EvidenceRecord",
    "ResearchState",
    "EvidenceAction",
    "EvidenceDecision",
    "EvidencePolicy",
    "EvidencePolicyConfig",
    "ResearchControl",
    "ResearchController",
    "SourceAuthorityPolicy",
    "ClaimEvidence",
    "ClaimAnchorCheck",
    "ClaimAnchorReport",
    "ClaimAnchorVerifier",
    "ClaimUnit",
    "EvidenceProposal",
    "ClaimManifestEntry",
    "ClaimManifest",
    "ClaimManifestBuilder",
    "EvidenceCandidate",
    "EvidenceCandidateRetriever",
    "ResearchVerifier",
    "ResearchVerificationReport",
    "EmbeddingBackend",
    "MultilingualE5Embedder",
    "DenseEvidenceCandidate",
    "DenseEvidenceRetriever",
]