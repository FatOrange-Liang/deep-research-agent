from __future__ import annotations

import re
from dataclasses import dataclass


CODE_TOKENS = (
    " import ",
    " def ",
    " return ",
    " = ",
    " . add_",
    " . invoke",
    " . stream",
    "config =",
    "builder .",
    "graph .",
)


@dataclass(frozen=True)
class PassageQuality:
    keep: bool
    tier: str
    reasons: tuple[str, ...]


def code_score(text: str) -> int:
    lowered = f" {text.lower()} "

    score = sum(
        token in lowered
        for token in CODE_TOKENS
    )

    punctuation = sum(
        text.count(ch)
        for ch in "(){}[]=."
    )

    if punctuation >= 8:
        score += 1

    return score


def passage_reasons(
    text: str,
) -> tuple[str, ...]:

    stripped = text.strip()

    reasons: list[str] = []

    if len(stripped) < 30:
        reasons.append(
            "too_short"
        )

    if len(stripped.split()) < 5:
        reasons.append(
            "too_few_words"
        )

    cscore = code_score(
        stripped
    )

    if cscore >= 2:
        reasons.append(
            "code_like"
        )

    # -------------------------------------
    # Broken prefix
    # -------------------------------------

    broken_prefix = bool(
        re.match(
            r'^[\W_]*(?:["\')\]}]|ART\b|lder\b|nt\b)',
            stripped,
            flags=re.IGNORECASE,
        )
    )

    if broken_prefix:
        reasons.append(
            "broken_prefix"
        )

    # -------------------------------------
    # Strong malformed-code signature
    #
    # Example:
    #   " ) age = interrupt ( "What's your age?
    # -------------------------------------

    assignment_call = bool(
        re.search(
            r'\b[A-Za-z_]\w*'
            r'\s*=\s*'
            r'[A-Za-z_]\w*'
            r'\s*\(',
            stripped,
        )
    )

    if (
        broken_prefix
        and assignment_call
    ):
        reasons.append(
            "broken_code_fragment"
        )

    # -------------------------------------
    # Quote balance
    # -------------------------------------

    if stripped.count('"') % 2 == 1:
        reasons.append(
            "unbalanced_quotes"
        )

    # -------------------------------------
    # Broken suffix
    # -------------------------------------

    if (
        stripped.endswith(",")
        or stripped.endswith("(")
        or stripped.endswith("=")
    ):
        reasons.append(
            "broken_suffix"
        )

    return tuple(reasons)

def assess_passage(
    text: str,
) -> PassageQuality:

    reasons = passage_reasons(text)

    reason_set = set(reasons)

    # -------------------------------------
    # Hard reject:
    # only highly suspicious combinations.
    # -------------------------------------

    hard_reject = (
        "broken_code_fragment"
        in reason_set

        or (
            "broken_prefix"
            in reason_set
            and "code_like"
            in reason_set
        )

        or (
            "broken_prefix"
            in reason_set
            and "unbalanced_quotes"
            in reason_set
        )

        or (
            "broken_suffix"
            in reason_set
            and "code_like"
            in reason_set
        )

        or (
            "too_short"
            in reason_set
            and "code_like"
            in reason_set
        )
    )

    if hard_reject:
        return PassageQuality(
            keep=False,
            tier="reject",
            reasons=reasons,
        )

    # Technical-looking passages are not
    # automatically bad.
    if reasons:
        return PassageQuality(
            keep=True,
            tier="review",
            reasons=reasons,
        )

    return PassageQuality(
        keep=True,
        tier="clean",
        reasons=(),
    )