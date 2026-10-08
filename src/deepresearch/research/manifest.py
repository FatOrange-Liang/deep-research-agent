from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Sequence

from .claim_anchors import (
    ClaimAnchorCheck,
    ClaimAnchorVerifier,
    ClaimEvidence,
    iter_markdown_blocks,
)
from .state import ResearchState


# Recognize source-like markers, including malformed ones.
# CitationGuard is responsible for rejecting malformed IDs.
SOURCE_MARKER = re.compile(
    r"\[S_[^\]\s]+\]"
)

VALID_CITATION = re.compile(
    r"\[(S_[0-9a-fA-F]{8})\]"
)

BULLET_PREFIX = re.compile(
    r"^(?:[-*+]|\d+[.)])\s+"
)

TERMINATORS = set(
    "。！？!?；;."
)

MARKDOWN_LINK = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)"
)

PURE_MARKDOWN_LINKS = re.compile(
    r"^(?:\s*\[[^\]]+\]\([^)]+\)\s*)+$"
)

VISIBLE_CONTENT = re.compile(
    r"[A-Za-z0-9\u3400-\u9fff]"
)

# Presentation/meta text is not a factual claim unit.
NON_CLAIM_PREFIXES = (
    "我查阅了",
    "我参考了",
    "我搜索了",
    "以下是",
    "下面是",
    "依据官方资料",
    "根据官方资料",
    "简言之",
    "总之",
    "总结来说",
    "in short",
    "in summary",
    "to summarize",
)


@dataclass(frozen=True)
class ClaimUnit:
    """
    One deterministically extracted claim unit.

    Claim IDs are local to the current answer version.
    They must be regenerated when the answer changes.
    """

    claim_id: str
    text: str
    cited_source_ids: tuple[str, ...]

    @property
    def has_citation(self) -> bool:

        return bool(
            self.cited_source_ids
        )


@dataclass(frozen=True)
class EvidenceProposal:
    """
    Proposed evidence for an extracted claim.

    The proposal cannot create a new claim.
    It must refer to an existing ClaimUnit.
    """

    claim_id: str
    source_id: str
    evidence_quote: str


@dataclass(frozen=True)
class ClaimManifestEntry:

    claim: ClaimUnit
    anchor_checks: tuple[ClaimAnchorCheck, ...] = ()

    @property
    def anchored(self) -> bool:
        """
        True if at least one submitted source anchor has passed
        deterministic verification.
        """

        return any(
            check.anchored
            for check in self.anchor_checks
        )


@dataclass(frozen=True)
class ClaimManifest:

    entries: tuple[ClaimManifestEntry, ...]
    rejected_proposals: tuple[EvidenceProposal, ...]

    @property
    def claim_count(self) -> int:

        return len(self.entries)

    @property
    def cited_claim_count(self) -> int:

        return sum(
            entry.claim.has_citation
            for entry in self.entries
        )

    @property
    def anchored_claim_count(self) -> int:

        return sum(
            entry.anchored
            for entry in self.entries
        )

    @property
    def citation_coverage(self) -> float:

        if not self.entries:
            return 0.0

        return (
            self.cited_claim_count
            / self.claim_count
        )

    @property
    def anchor_coverage(self) -> float:

        if not self.entries:
            return 0.0

        return (
            self.anchored_claim_count
            / self.claim_count
        )

    @property
    def uncited_claim_ids(self) -> tuple[str, ...]:

        return tuple(
            entry.claim.claim_id
            for entry in self.entries
            if not entry.claim.has_citation
        )

    @property
    def unanchored_claim_ids(self) -> tuple[str, ...]:

        return tuple(
            entry.claim.claim_id
            for entry in self.entries
            if not entry.anchored
        )

    @property
    def all_claim_units_anchored(self) -> bool:
        """
        Every extracted claim unit has at least one valid evidence anchor.

        This is NOT a semantic correctness judgment.
        """

        return (
            bool(self.entries)
            and self.anchor_coverage == 1.0
            and not self.rejected_proposals
        )


class ClaimManifestBuilder:
    """
    Build a claim-level evidence manifest.

    Citation attribution is Markdown-block aware:

    - citations directly attached to a sentence stay attached;
    - when citations occur only in the final sentence of a bullet/paragraph,
      those citations are inherited by earlier factual sentences in the
      same block;
    - citations are never propagated across block boundaries.

    No LLM calls are performed.
    """

    def __init__(
        self,
        verifier: ClaimAnchorVerifier | None = None,
    ) -> None:

        self.verifier = (
            verifier
            or ClaimAnchorVerifier()
        )

    def extract_claims(
        self,
        answer: str,
    ) -> tuple[ClaimUnit, ...]:

        if not isinstance(answer, str):
            raise TypeError(
                "'answer' must be a string."
            )

        claims: list[ClaimUnit] = []

        for raw_block in iter_markdown_blocks(
            answer
        ):
            raw_block = raw_block.strip()

            # A trailing citation may cover multiple factual sentences
            # only inside one Markdown list item. Ordinary prose
            # paragraphs keep sentence-local citation semantics so that
            # "claim A. unrelated claim B [S_x]." does not attach S_x
            # to claim A.
            is_list_item = bool(
                BULLET_PREFIX.match(
                    raw_block
                )
            )

            block = BULLET_PREFIX.sub(
                "",
                raw_block,
            )

            if not block:
                continue

            # Skip presentation-only lines before sentence extraction.
            if self._is_non_claim_block(
                block
            ):
                continue

            segments = self._split_sentences(
                block
            )

            if not segments:
                continue

            direct_source_ids = [
                tuple(
                    dict.fromkeys(
                        VALID_CITATION.findall(
                            segment
                        )
                    )
                )
                for segment in segments
            ]

            effective_source_ids = (
                self._apply_block_citation_scope(
                    direct_source_ids,
                    allow_inheritance=is_list_item,
                )
            )

            for (
                segment,
                source_ids,
            ) in zip(
                segments,
                effective_source_ids,
            ):
                claim_text = (
                    self._clean_claim_text(
                        segment
                    )
                )

                if not claim_text:
                    continue

                if self._is_non_claim_text(
                    claim_text
                ):
                    continue

                claim_id = (
                    f"C{len(claims) + 1:03d}"
                )

                claims.append(
                    ClaimUnit(
                        claim_id=claim_id,
                        text=claim_text,
                        cited_source_ids=source_ids,
                    )
                )

        return tuple(claims)

    @staticmethod
    def _apply_block_citation_scope(
        direct_source_ids: Sequence[
            tuple[str, ...]
        ],
        *,
        allow_inheritance: bool,
    ) -> list[tuple[str, ...]]:
        """
        Propagate a trailing block citation only when citation ownership is
        unambiguous.

        Example:
            sentence A. sentence B. sentence C. [S_x]
        becomes:
            A -> S_x
            B -> S_x
            C -> S_x

        But if multiple segments have their own citation markers, no
        cross-sentence inference is made.
        """

        output = list(
            direct_source_ids
        )

        if not allow_inheritance:
            return output

        cited_indexes = [
            index
            for index, source_ids
            in enumerate(
                direct_source_ids
            )
            if source_ids
        ]

        if (
            len(cited_indexes) == 1
            and cited_indexes[0]
            == len(direct_source_ids) - 1
        ):
            inherited = direct_source_ids[
                cited_indexes[0]
            ]

            output = [
                source_ids
                if source_ids
                else inherited
                for source_ids
                in direct_source_ids
            ]

        return output

    @staticmethod
    def _clean_claim_text(
        segment: str,
    ) -> str:
        # If the segment is only one or more Markdown links plus optional
        # source markers, it is navigation, not a factual claim.
        without_sources = SOURCE_MARKER.sub(
            "",
            segment,
        ).strip()

        if PURE_MARKDOWN_LINKS.fullmatch(
            without_sources
        ):
            return ""

        # Preserve visible link labels but remove URLs.
        text = MARKDOWN_LINK.sub(
            r"\1",
            without_sources,
        )

        text = (
            text
            .replace("`", "")
            .replace("**", "")
            .replace("__", "")
            .replace("~~", "")
            .strip()
            .rstrip("。！？!?；;.")
            .strip()
        )

        if not VISIBLE_CONTENT.search(
            text
        ):
            return ""

        return text

    @staticmethod
    def _is_non_claim_block(
        block: str,
    ) -> bool:
        cleaned = (
            MARKDOWN_LINK.sub(
                r"\1",
                SOURCE_MARKER.sub(
                    "",
                    block,
                ),
            )
            .replace("`", "")
            .replace("**", "")
            .replace("__", "")
            .replace("~~", "")
            .strip()
            .casefold()
        )

        return any(
            cleaned.startswith(
                prefix.casefold()
            )
            for prefix
            in NON_CLAIM_PREFIXES
        )

    @staticmethod
    def _is_non_claim_text(
        text: str,
    ) -> bool:
        cleaned = text.strip().casefold()

        if not cleaned:
            return True

        if not VISIBLE_CONTENT.search(
            cleaned
        ):
            return True

        return any(
            cleaned.startswith(
                prefix.casefold()
            )
            for prefix
            in NON_CLAIM_PREFIXES
        )

    @staticmethod
    def _split_sentences(
        line: str,
    ) -> list[str]:
        """
        Conservative punctuation-based segmentation.

        Citation markers immediately following terminal punctuation remain
        attached to that sentence.
        """

        segments: list[str] = []

        start = 0
        index = 0

        while index < len(line):

            marker = SOURCE_MARKER.match(
                line,
                index,
            )

            if marker is not None:
                index = marker.end()
                continue

            char = line[index]

            if char not in TERMINATORS:
                index += 1
                continue

            # Avoid splitting a decimal or a dot directly inside an
            # ordinary word.
            if char == ".":

                next_index = index + 1

                if next_index < len(line):

                    next_char = line[
                        next_index
                    ]

                    if (
                        not next_char.isspace()
                        and not line.startswith(
                            "[S_",
                            next_index,
                        )
                    ):
                        index += 1
                        continue

            end = index + 1

            # Include citation markers appearing immediately after sentence
            # punctuation.
            probe = end

            while probe < len(line):

                while (
                    probe < len(line)
                    and line[probe].isspace()
                ):
                    probe += 1

                trailing_marker = (
                    SOURCE_MARKER.match(
                        line,
                        probe,
                    )
                )

                if trailing_marker is None:
                    break

                end = trailing_marker.end()
                probe = end

            segment = line[
                start:end
            ].strip()

            if segment:
                segments.append(
                    segment
                )

            start = end
            index = end

        remainder = line[
            start:
        ].strip()

        if remainder:
            segments.append(
                remainder
            )

        return segments

    def build(
        self,
        *,
        answer: str,
        state: ResearchState,
        proposals: Sequence[EvidenceProposal] = (),
    ) -> ClaimManifest:
        """
        Build the complete manifest.

        Evidence proposals cannot introduce new claims or silently change
        the claim's citation IDs.
        """

        claims = self.extract_claims(
            answer
        )

        claim_index = {
            claim.claim_id: claim
            for claim in claims
        }

        checks_by_claim: dict[
            str,
            list[ClaimAnchorCheck],
        ] = {
            claim.claim_id: []
            for claim in claims
        }

        rejected: list[
            EvidenceProposal
        ] = []

        for proposal in proposals:

            claim = claim_index.get(
                proposal.claim_id
            )

            if claim is None:
                rejected.append(
                    proposal
                )
                continue

            if (
                proposal.source_id
                not in claim.cited_source_ids
            ):
                rejected.append(
                    proposal
                )
                continue

            evidence = ClaimEvidence(
                claim=claim.text,
                source_id=proposal.source_id,
                evidence_quote=proposal.evidence_quote,
            )

            anchor_report = (
                self.verifier.verify(
                    answer=answer,
                    state=state,
                    items=[evidence],
                )
            )

            checks_by_claim[
                claim.claim_id
            ].extend(
                anchor_report.checks
            )

        entries = tuple(
            ClaimManifestEntry(
                claim=claim,
                anchor_checks=tuple(
                    checks_by_claim[
                        claim.claim_id
                    ]
                ),
            )
            for claim in claims
        )

        return ClaimManifest(
            entries=entries,
            rejected_proposals=tuple(
                rejected
            ),
        )
