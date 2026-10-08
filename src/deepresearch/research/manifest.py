
from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Sequence

from .claim_anchors import (
    ClaimAnchorCheck,
    ClaimAnchorVerifier,
    ClaimEvidence,
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
        True if at least one submitted source anchor
        has passed deterministic verification.
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
        Every extracted claim unit has at least one
        valid evidence anchor.

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

    Workflow:

    1. Extract claim units from the final answer.
    2. Record citation IDs attached to each unit.
    3. Accept evidence proposals only for extracted claims.
    4. Reuse ClaimAnchorVerifier to check proposed anchors.
    5. Calculate citation and anchor coverage.

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

        inside_code_block = False

        for raw_line in answer.splitlines():

            line = raw_line.strip()

            # Ignore fenced code blocks.
            if (
                line.startswith("```")
                or line.startswith("~~~")
            ):
                inside_code_block = (
                    not inside_code_block
                )
                continue

            if inside_code_block:
                continue

            if not line:
                continue

            # Ignore Markdown headings.
            if line.startswith("#"):
                continue

            # Ignore Markdown separators.
            if re.fullmatch(
                r"[-*_]{3,}",
                line,
            ):
                continue

            # Remove Markdown list prefix.
            line = BULLET_PREFIX.sub(
                "",
                line,
            )

            for segment in self._split_sentences(line):

                source_ids = tuple(
                    dict.fromkeys(
                        VALID_CITATION.findall(
                            segment
                        )
                    )
                )

                # Keep the claim text separate
                # from its citation markers.
                claim_text = (
                    SOURCE_MARKER.sub(
                        "",
                        segment,
                    )
                    .strip()
                    .rstrip("。！？!?；;.")
                    .strip()
                )

                if not claim_text:
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
    def _split_sentences(
        line: str,
    ) -> list[str]:
        """
        Conservative punctuation-based segmentation.

        Citation markers immediately following terminal
        punctuation remain attached to that sentence.

        This is a heuristic segmenter, not an atomic
        factual-claim parser.
        """

        segments: list[str] = []

        start = 0
        index = 0

        while index < len(line):

            # Skip over citation markers so punctuation
            # is not interpreted inside them.
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

            # Avoid splitting a decimal or a dot
            # directly inside an ordinary word.
            if char == ".":

                next_index = index + 1

                if next_index < len(line):

                    next_char = line[next_index]

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

            # Include citation markers appearing
            # immediately after the sentence punctuation.
            probe = end

            while probe < len(line):

                while (
                    probe < len(line)
                    and line[probe].isspace()
                ):
                    probe += 1

                trailing_marker = SOURCE_MARKER.match(
                    line,
                    probe,
                )

                if trailing_marker is None:
                    break

                end = trailing_marker.end()
                probe = end

            segment = line[start:end].strip()

            if segment:
                segments.append(segment)

            start = end
            index = end

        remainder = line[start:].strip()

        if remainder:
            segments.append(remainder)

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

        Evidence proposals cannot introduce new claims
        or silently change the claim's citation IDs.
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

        rejected: list[EvidenceProposal] = []

        for proposal in proposals:

            claim = claim_index.get(
                proposal.claim_id
            )

            # A proposal for an unknown claim must
            # never create an additional claim.
            if claim is None:

                rejected.append(
                    proposal
                )

                continue

            # Proposed source must actually be cited
            # by the extracted claim.
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

            anchor_report = self.verifier.verify(
                answer=answer,
                state=state,
                items=[evidence],
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
