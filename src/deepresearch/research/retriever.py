
from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Sequence

from .claim_anchors import normalize_text

from .manifest import (
    ClaimUnit,
    EvidenceProposal,
)

from .state import ResearchState


# Basic stop words for lexical retrieval.
STOPWORDS = {
    "a", "an", "the", "and", "or",
    "is", "are", "was", "were",
    "be", "been", "to", "for",
    "in", "on", "of", "by",
    "with", "as", "at", "it",
    "its", "this", "that",
    "through", "after", "each",
    "so", "can", "from",
    "langgraph",
}


# Split Chinese punctuation and English
# sentence boundaries conservatively.
PASSAGE_SPLIT = re.compile(
    r'(?<=[。！？；!?])\s*'
    r'|(?<=[.!?])\s+(?=[A-Z0-9"“])'
    r'|\n{2,}'
)


def lexical_features(
    text: str,
) -> set[str]:
    """
    Extract simple multilingual lexical features.

    English: word matching with light normalization.
    Chinese: overlapping two-character features.

    This is not semantic embedding retrieval.
    """

    normalized = normalize_text(text)

    features: set[str] = set()

    # English features.
    words = re.findall(
        r"[a-z0-9]+",
        normalized,
    )

    for word in words:

        if word in STOPWORDS:
            continue

        if len(word) <= 1:
            continue

        # Light English plural normalization.
        if (
            len(word) > 5
            and word.endswith("ies")
        ):
            word = word[:-3] + "y"

        elif (
            len(word) > 4
            and word.endswith("s")
            and not word.endswith(
                ("ss", "us", "is")
            )
        ):
            word = word[:-1]

        features.add(word)

    # Chinese character features.
    chinese_groups = re.findall(
        r"[\u3400-\u9fff]+",
        normalized,
    )

    for group in chinese_groups:

        if len(group) == 1:

            features.add(group)

        else:

            for index in range(
                len(group) - 1
            ):

                features.add(
                    group[index:index + 2]
                )

    return features


@dataclass(frozen=True)
class EvidenceCandidate:
    """
    A retrieved evidence passage.

    The quote originates from stored source content,
    with whitespace normalized for readability.

    Its lexical score is NOT an entailment score.
    """

    claim_id: str

    source_id: str

    quote: str

    lexical_score: float

    matched_terms: int

    def to_proposal(self) -> EvidenceProposal:
        """
        Convert retrieved evidence to the existing
        ClaimManifestBuilder proposal format.
        """

        return EvidenceProposal(
            claim_id=self.claim_id,
            source_id=self.source_id,
            evidence_quote=self.quote,
        )


class EvidenceCandidateRetriever:
    """
    Retrieve candidate source passages for claims.

    Rules:

    1. Only search sources cited by the claim.
    2. Require a successful page-read record.
    3. Extract quotes from stored source content.
    4. Rank candidates using lexical overlap.
    5. Never generate or rewrite evidence facts.

    All operations are deterministic and offline.
    """

    def __init__(
        self,
        *,
        top_k_per_claim: int = 2,
        min_quote_chars: int = 24,
        max_quote_chars: int = 450,
        min_lexical_score: float = 0.20,
    ) -> None:

        if top_k_per_claim < 1:
            raise ValueError(
                "'top_k_per_claim' must be positive."
            )

        if min_quote_chars < 1:
            raise ValueError(
                "'min_quote_chars' must be positive."
            )

        if max_quote_chars < min_quote_chars:
            raise ValueError(
                "'max_quote_chars' cannot be smaller "
                "than 'min_quote_chars'."
            )

        if not 0 <= min_lexical_score <= 1:
            raise ValueError(
                "'min_lexical_score' must be "
                "between 0 and 1."
            )

        self.top_k_per_claim = top_k_per_claim

        self.min_quote_chars = min_quote_chars

        self.max_quote_chars = max_quote_chars

        self.min_lexical_score = min_lexical_score

    def retrieve(
        self,
        *,
        claims: Sequence[ClaimUnit],
        state: ResearchState,
    ) -> dict[str, tuple[EvidenceCandidate, ...]]:
        """
        Return candidate evidence for every claim.

        Even claims with no candidates appear in the
        returned dictionary with an empty tuple.
        """

        output: dict[
            str,
            tuple[EvidenceCandidate, ...],
        ] = {}

        for claim in claims:

            claim_features = lexical_features(
                claim.text
            )

            candidates: list[
                EvidenceCandidate
            ] = []

            if not claim_features:

                output[claim.claim_id] = ()

                continue

            # Restrict retrieval to the claim's
            # existing citation IDs.
            for source_id in claim.cited_source_ids:

                record = state.sources.get(
                    source_id
                )

                if record is None:
                    continue

                # Search snippets alone must not be
                # treated as page-level evidence.
                if not record.read_full_page:
                    continue

                content = record.source.content

                if not content:
                    continue

                for passage in self._extract_passages(
                    content
                ):

                    passage_features = lexical_features(
                        passage
                    )

                    if not passage_features:
                        continue

                    overlap = (
                        claim_features
                        & passage_features
                    )

                    if not overlap:
                        continue

                    # Prioritize claim-term recall,
                    # while also considering passage
                    # precision.
                    recall = (
                        len(overlap)
                        / len(claim_features)
                    )

                    precision = (
                        len(overlap)
                        / len(passage_features)
                    )

                    score = (
                        0.8 * recall
                        + 0.2 * precision
                    )

                    if (
                        score
                        < self.min_lexical_score
                    ):
                        continue

                    # A candidate's quote must exist
                    # in normalized source content.
                    if (
                        normalize_text(passage)
                        not in normalize_text(content)
                    ):
                        continue

                    candidates.append(
                        EvidenceCandidate(
                            claim_id=claim.claim_id,
                            source_id=source_id,
                            quote=passage,
                            lexical_score=score,
                            matched_terms=len(overlap),
                        )
                    )

            # Deterministic ranking.
            candidates.sort(
                key=lambda item: (
                    -item.lexical_score,
                    -item.matched_terms,
                    item.source_id,
                    item.quote,
                )
            )

            # Deduplicate equivalent passages.
            unique: list[
                EvidenceCandidate
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

                unique.append(
                    candidate
                )

                if (
                    len(unique)
                    >= self.top_k_per_claim
                ):
                    break

            output[claim.claim_id] = tuple(
                unique
            )

        return output

    def extract_passages(
        self,
        content: str,
    ) -> list[str]:
        """
        Public passage extraction interface.

        Shared by lexical and dense retrieval.
        """
        return self._extract_passages(content)

    def _extract_passages(
        self,
        content: str,
    ) -> list[str]:
        """
        Segment stored page text into manageable
        evidence excerpts.

        Long passages are divided into overlapping
        windows instead of keeping only their prefix.
        """

        passages: list[str] = []

        segments = PASSAGE_SPLIT.split(
            content
        )

        for segment in segments:

            # Normalize whitespace without asking
            # a language model to rewrite the source.
            cleaned = re.sub(
                r"\s+",
                " ",
                segment,
            ).strip()

            if (
                len(cleaned)
                < self.min_quote_chars
            ):
                continue

            if (
                len(cleaned)
                <= self.max_quote_chars
            ):

                passages.append(
                    cleaned
                )

                continue

            # Overlapping windows prevent useful
            # evidence in the middle of a long
            # paragraph from being discarded.
            stride = max(
                1,
                self.max_quote_chars // 2,
            )

            for start in range(
                0,
                len(cleaned),
                stride,
            ):

                excerpt = cleaned[
                    start:start + self.max_quote_chars
                ].strip()

                if (
                    len(excerpt)
                    >= self.min_quote_chars
                ):

                    passages.append(
                        excerpt
                    )

                if (
                    start + self.max_quote_chars
                    >= len(cleaned)
                ):
                    break

        return passages
