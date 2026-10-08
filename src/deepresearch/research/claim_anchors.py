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


SOURCE_MARKER = re.compile(
    r"\[S_[^\]\s]+\]"
)

MARKDOWN_LINK = re.compile(
    r"\[([^\]]+)\]\(([^)]+)\)"
)

BULLET_PREFIX = re.compile(
    r"^(?:[-*+]|\d+[.)])\s+"
)

MARKDOWN_SEPARATOR = re.compile(
    r"[-*_]{3,}"
)


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


def plain_markdown_text(
    text: str,
    *,
    remove_source_markers: bool = True,
) -> str:
    """
    Convert lightweight Markdown into stable plain text for matching.

    This does not rewrite content semantically. It only removes
    presentation syntax that would otherwise make deterministic claim
    matching brittle.
    """

    text = BULLET_PREFIX.sub(
        "",
        text.strip(),
    )

    # Preserve the visible label of Markdown links.
    text = MARKDOWN_LINK.sub(
        r"\1",
        text,
    )

    if remove_source_markers:
        text = SOURCE_MARKER.sub(
            "",
            text,
        )

    # Preserve the contents of inline code while removing backticks.
    text = text.replace("`", "")

    # Remove common emphasis syntax. Single underscores are intentionally
    # preserved because identifiers such as thread_id are meaningful.
    text = (
        text
        .replace("**", "")
        .replace("__", "")
        .replace("~~", "")
    )

    return normalize_text(text)


def iter_markdown_blocks(
    answer: str,
) -> tuple[str, ...]:
    """
    Return citation-scope blocks from Markdown-like answer text.

    Rules:
    - fenced code is ignored;
    - headings/separators are ignored;
    - each list item is its own block;
    - consecutive ordinary lines form one paragraph block;
    - blank lines terminate a block.

    The original text is preserved inside each returned block so citation
    markers remain available for deterministic attribution.
    """

    blocks: list[str] = []
    paragraph_lines: list[str] = []
    current_bullet_lines: list[str] = []
    inside_code_block = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            block = " ".join(
                line.strip()
                for line in paragraph_lines
                if line.strip()
            ).strip()
            if block:
                blocks.append(block)
            paragraph_lines.clear()

    def flush_bullet() -> None:
        if current_bullet_lines:
            block = " ".join(
                line.strip()
                for line in current_bullet_lines
                if line.strip()
            ).strip()
            if block:
                blocks.append(block)
            current_bullet_lines.clear()

    for raw_line in answer.splitlines():
        stripped = raw_line.strip()

        if (
            stripped.startswith("```")
            or stripped.startswith("~~~")
        ):
            flush_paragraph()
            flush_bullet()
            inside_code_block = not inside_code_block
            continue

        if inside_code_block:
            continue

        if not stripped:
            flush_paragraph()
            flush_bullet()
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            flush_bullet()
            continue

        if MARKDOWN_SEPARATOR.fullmatch(
            stripped
        ):
            flush_paragraph()
            flush_bullet()
            continue

        is_bullet = bool(
            BULLET_PREFIX.match(
                stripped
            )
        )

        if is_bullet:
            flush_paragraph()
            flush_bullet()
            current_bullet_lines.append(
                stripped
            )
            continue

        # Indented/non-bullet lines directly following a bullet are treated
        # as continuation of that bullet's citation scope.
        if current_bullet_lines:
            current_bullet_lines.append(
                stripped
            )
            continue

        paragraph_lines.append(
            stripped
        )

    flush_paragraph()
    flush_bullet()

    return tuple(blocks)


@dataclass(frozen=True)
class ClaimEvidence:
    """
    A claim together with its proposed evidence.

    The claim must appear in the answer, and the corresponding source
    citation must occur in the same Markdown citation-scope block.

    This supports common answer styles where one citation at the end of a
    bullet/paragraph backs multiple factual sentences in that same block.
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
        True only when at least one claim was provided and every submitted
        anchor passed.

        This does NOT establish claim entailment or complete answer
        coverage.
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
    1. Claim appears in the final answer after Markdown normalization.
    2. Citation occurs in the same Markdown citation-scope block.
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

        answer_blocks = iter_markdown_blocks(
            answer
        )

        checks: list[ClaimAnchorCheck] = []

        for item in items:

            status = self._check_one(
                item=item,
                answer_blocks=answer_blocks,
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
        answer_blocks: Sequence[str],
        state: ResearchState,
    ) -> ClaimAnchorStatus:

        claim = plain_markdown_text(
            item.claim
        )

        source_id = item.source_id.strip()

        # -------------------------------------
        # 1 + 2. Claim must occur in a block
        # carrying the requested citation.
        # -------------------------------------

        containing_blocks = [
            block
            for block in answer_blocks
            if (
                claim
                and claim
                in plain_markdown_text(
                    block
                )
            )
        ]

        if not containing_blocks:

            return "claim_not_in_answer"

        normalized_marker = normalize_text(
            f"[{source_id}]"
        )

        citation_attached = False

        for block in containing_blocks:

            raw_block = block.strip()

            if BULLET_PREFIX.match(
                raw_block
            ):
                # Within one list item, a trailing citation can scope over
                # all factual sentences in that item.
                if (
                    normalized_marker
                    in normalize_text(raw_block)
                ):
                    citation_attached = True
                    break

                continue

            # Ordinary prose retains the original strict behavior:
            # the citation must immediately follow the claim (allowing
            # terminal punctuation and whitespace only).
            comparable_block = plain_markdown_text(
                raw_block,
                remove_source_markers=False,
            )

            citation_pattern = (
                re.escape(claim)
                + r"\s*[.!?。！？;；:]?\s*"
                + re.escape(
                    normalized_marker
                )
            )

            if re.search(
                citation_pattern,
                comparable_block,
            ):
                citation_attached = True
                break

        if not citation_attached:

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
