
from __future__ import annotations

import re
import unicodedata

from dataclasses import dataclass
from typing import Literal, Sequence

from .state import ResearchState


ClaimAnchorStatus = Literal[
    "anchored",
    "claim_not_in_answer",
    "citation_not_attached",
    "source_not_found",
    "source_not_read",
    "quote_too_short",
    "quote_not_found",
]


def normalize_text(text: str) -> str:
    """
    Normalize text for deterministic comparison.

    This handles:
    - Unicode compatibility differences
    - repeated whitespace
    - line breaks
    - letter casing

    It does not perform semantic matching.
    """

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().casefold()


@dataclass(frozen=True)
class ClaimEvidence:
    """
    A claim together with its proposed evidence.

    The claim must appear in the answer with
    the corresponding citation immediately after it.
    """

    claim: str

    source_id: str

    evidence_quote: str


@dataclass(frozen=True)
class ClaimAnchorCheck:

    item: ClaimEvidence

    status: ClaimAnchorStatus

    @property
    def anchored(self) -> bool:
        return self.status == "anchored"


@dataclass(frozen=True)
class ClaimAnchorReport:

    checks: tuple[ClaimAnchorCheck, ...]

    @property
    def all_anchored(self) -> bool:
        """
        True only when at least one claim was
        provided and every submitted anchor passed.

        This does NOT establish claim entailment
        or complete answer coverage.
        """

        return bool(self.checks) and all(
            check.anchored
            for check in self.checks
        )

    @property
    def anchored_count(self) -> int:

        return sum(
            check.anchored
            for check in self.checks
        )


class ClaimAnchorVerifier:
    """
    Deterministic claim-to-source anchor verification.

    Checks:
    1. Claim appears in the final answer.
    2. Citation is attached to that claim.
    3. Source exists in ResearchState.
    4. Source has been read in full.
    5. Evidence quote is sufficiently specific.
    6. Evidence quote occurs in stored source content.

    Important:
    Quote presence is NOT semantic entailment.
    """

    def __init__(
        self,
        *,
        min_quote_chars: int = 24,
    ) -> None:

        if min_quote_chars < 1:
            raise ValueError(
                "'min_quote_chars' must be positive."
            )

        self.min_quote_chars = min_quote_chars

    def verify(
        self,
        *,
        answer: str,
        state: ResearchState,
        items: Sequence[ClaimEvidence],
    ) -> ClaimAnchorReport:

        normalized_answer = normalize_text(
            answer
        )

        checks: list[ClaimAnchorCheck] = []

        for item in items:

            status = self._check_one(
                item=item,
                normalized_answer=normalized_answer,
                state=state,
            )

            checks.append(
                ClaimAnchorCheck(
                    item=item,
                    status=status,
                )
            )

        return ClaimAnchorReport(
            checks=tuple(checks)
        )

    def _check_one(
        self,
        *,
        item: ClaimEvidence,
        normalized_answer: str,
        state: ResearchState,
    ) -> ClaimAnchorStatus:

        claim = normalize_text(
            item.claim
        )

        source_id = item.source_id.strip()

        # -------------------------------------
        # 1. Check whether claim exists
        # -------------------------------------

        if not claim or claim not in normalized_answer:

            return "claim_not_in_answer"

        # -------------------------------------
        # 2. Citation must be attached to claim
        # -------------------------------------

        citation_pattern = (
            re.escape(claim)
            + r"\s*[.!?。！？;；:]?\s*"
            + re.escape(
                f"[{source_id}]".casefold()
            )
        )

        if not re.search(
            citation_pattern,
            normalized_answer,
        ):

            return "citation_not_attached"

        # -------------------------------------
        # 3. Source must exist
        # -------------------------------------

        record = state.sources.get(
            source_id
        )

        if record is None:

            return "source_not_found"

        # -------------------------------------
        # 4. Require full-page evidence
        # -------------------------------------

        if not record.read_full_page:

            return "source_not_read"

        # -------------------------------------
        # 5. Require a meaningful quote length
        # -------------------------------------

        quote = normalize_text(
            item.evidence_quote
        )

        if len(quote) < self.min_quote_chars:

            return "quote_too_short"

        # -------------------------------------
        # 6. Verify that the quote is real
        # -------------------------------------

        source_content = normalize_text(
            record.source.content
        )

        if quote not in source_content:

            return "quote_not_found"

        return "anchored"
