
import json

from deepresearch.llm import Message

from deepresearch.research import (
    ResearchState,
    ResearchVerifier,
)

from deepresearch.research.claim_anchors import (
    normalize_text,
)

from deepresearch.research.sources import (
    make_source_id,
)


URL = "https://docs.example.com/langgraph"

SOURCE_ID = make_source_id(URL)

CLAIM_A = (
    "LangGraph saves state through checkpointers"
)

CLAIM_B = (
    "LangGraph supports human review through interrupts"
)

QUOTE_A = (
    "A checkpointer persists graph state "
    "after each execution step."
)

QUOTE_B = (
    "Interrupts pause execution so a "
    "human can review and resume."
)

SOURCE_CONTENT = (
    "LangGraph documentation.\n"
    + QUOTE_A
    + "\n"
    + QUOTE_B
)

ANSWER = (
    f"{CLAIM_A} [{SOURCE_ID}].\n"
    f"{CLAIM_B} [{SOURCE_ID}]."
)


def create_state(
    *,
    read_page: bool = True,
    content: str = SOURCE_CONTENT,
) -> ResearchState:

    state = ResearchState(
        task="Research LangGraph"
    )

    search_payload = {
        "query": "LangGraph documentation",
        "results": [
            {
                "source_id": SOURCE_ID,
                "title": "Official Docs",
                "url": URL,
                "content": "Search snippet",
                "score": 0.9,
            }
        ],
    }

    assert state.ingest_message(
        Message(
            role="tool",
            name="web_search",
            tool_call_id="search_1",
            content=json.dumps(search_payload),
        )
    )

    if read_page:

        reader_payload = {
            "source_id": SOURCE_ID,
            "title": "Official Docs",
            "url": URL,
            "content": content,
            "content_type": "text/html",
            "truncated": False,
        }

        assert state.ingest_message(
            Message(
                role="tool",
                name="web_page_reader",
                tool_call_id="read_1",
                content=json.dumps(reader_payload),
            )
        )

    return state


# =========================================
# 1. Complete happy path
# =========================================

def test_complete_verification_pipeline() -> None:

    report = ResearchVerifier().verify(
        answer=ANSWER,
        state=create_state(),
    )

    assert report.citations_valid

    assert report.claim_count == 2

    assert report.citation_coverage == 1.0

    assert report.anchor_coverage == 1.0

    assert report.candidate_count >= 2

    assert report.structural_checks_passed


# =========================================
# 2. Missing citations reduce coverage
# =========================================

def test_uncited_claim_reduces_coverage() -> None:

    answer = (
        f"{CLAIM_A} [{SOURCE_ID}].\n"
        f"{CLAIM_B}."
    )

    report = ResearchVerifier().verify(
        answer=answer,
        state=create_state(),
    )

    assert report.claim_count == 2

    assert report.citation_coverage == 0.5

    assert report.anchor_coverage == 0.5

    assert not report.structural_checks_passed

    assert "C002" in (
        report.missing_candidate_claim_ids
    )


# =========================================
# 3. Search snippets are insufficient
# =========================================

def test_unread_source_cannot_produce_anchors() -> None:

    report = ResearchVerifier().verify(
        answer=ANSWER,
        state=create_state(
            read_page=False
        ),
    )

    assert report.citation_coverage == 1.0

    assert report.anchor_coverage == 0.0

    assert report.candidate_count == 0

    assert not report.structural_checks_passed


# =========================================
# 4. Invalid citation IDs
# =========================================

def test_invalid_citation_is_reported() -> None:

    answer = ANSWER.replace(
        SOURCE_ID,
        "S_deadbeef",
        1,
    )

    report = ResearchVerifier().verify(
        answer=answer,
        state=create_state(),
    )

    assert not report.citations_valid

    assert (
        "S_deadbeef"
        in report.citation_validation.invalid_source_ids
    )

    assert not report.structural_checks_passed


# =========================================
# 5. Unrelated page content
# =========================================

def test_unrelated_page_has_no_evidence() -> None:

    unrelated_content = (
        "This document discusses botanical gardens, "
        "fruit cultivation and tropical weather."
    )

    report = ResearchVerifier().verify(
        answer=ANSWER,
        state=create_state(
            content=unrelated_content
        ),
    )

    assert report.citations_valid

    assert report.anchor_coverage == 0.0

    assert not report.structural_checks_passed


# =========================================
# 6. Candidate quotes come from actual source
# =========================================

def test_candidate_quotes_are_traceable() -> None:

    report = ResearchVerifier().verify(
        answer=ANSWER,
        state=create_state(),
    )

    for candidate_list in report.candidates.values():

        for candidate in candidate_list:

            assert candidate.source_id == SOURCE_ID

            assert (
                normalize_text(candidate.quote)
                in normalize_text(SOURCE_CONTENT)
            )


# =========================================
# 7. Empty answer cannot pass
# =========================================

def test_empty_answer_does_not_pass() -> None:

    report = ResearchVerifier().verify(
        answer="",
        state=create_state(),
    )

    assert report.claim_count == 0

    assert report.citation_coverage == 0.0

    assert report.anchor_coverage == 0.0

    assert not report.structural_checks_passed


# =========================================
# 8. Structural checks are not entailment
# =========================================

def test_lexical_anchor_is_not_semantic_proof() -> None:

    # Deliberately contradictory claim:
    # the source says that a checkpointer DOES
    # persist graph state.
    false_claim = (
        "A checkpointer never persists graph state"
    )

    answer = (
        f"{false_claim} [{SOURCE_ID}]."
    )

    report = ResearchVerifier().verify(
        answer=answer,
        state=create_state(),
    )

    # Lexical similarity can still retrieve and
    # anchor a contradictory passage.
    #
    # This explicitly documents the limitation
    # of structural verification V1.
    assert report.candidate_count >= 1

    assert report.anchor_coverage == 1.0

    assert report.structural_checks_passed
