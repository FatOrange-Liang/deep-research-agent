
import json

from deepresearch.llm import Message

from deepresearch.research import (
    ResearchState,
    ResearchVerifier,
)

from deepresearch.research.sources import (
    make_source_id,
)


URL = (
    "https://docs.example.com/"
    "langgraph/persistence"
)

SOURCE_ID = make_source_id(URL)

ENGLISH_SOURCE_CONTENT = (
    "A checkpointer persists graph state "
    "after each execution step."
)


def create_state() -> ResearchState:

    state = ResearchState(
        task="Research LangGraph persistence"
    )

    # -------------------------------------
    # Simulate web_search
    # -------------------------------------

    search_payload = {
        "query": "LangGraph persistence",
        "results": [
            {
                "source_id": SOURCE_ID,
                "title": "Persistence Documentation",
                "url": URL,
                "content": (
                    "LangGraph persistence overview"
                ),
                "score": 0.9,
            }
        ],
    }

    assert state.ingest_message(
        Message(
            role="tool",
            name="web_search",
            tool_call_id="search_1",
            content=json.dumps(
                search_payload
            ),
        )
    )

    # -------------------------------------
    # Simulate web_page_reader
    # -------------------------------------

    reader_payload = {
        "source_id": SOURCE_ID,
        "title": "Persistence Documentation",
        "url": URL,
        "content": ENGLISH_SOURCE_CONTENT,
        "content_type": "text/html",
        "truncated": False,
    }

    assert state.ingest_message(
        Message(
            role="tool",
            name="web_page_reader",
            tool_call_id="reader_1",
            content=json.dumps(
                reader_payload
            ),
        )
    )

    return state


def test_english_claim_can_retrieve_english_evidence() -> None:

    answer = (
        "LangGraph saves graph state through "
        f"checkpointers [{SOURCE_ID}]."
    )

    report = ResearchVerifier().verify(
        answer=answer,
        state=create_state(),
    )

    assert report.citations_valid

    assert report.claim_count == 1

    assert report.candidate_count >= 1

    assert report.citation_coverage == 1.0

    assert report.anchor_coverage == 1.0

    assert report.structural_checks_passed


def test_chinese_claim_exposes_cross_language_gap() -> None:

    answer = (
        "LangGraph 通过检查点保存状态 "
        f"[{SOURCE_ID}]。"
    )

    report = ResearchVerifier().verify(
        answer=answer,
        state=create_state(),
    )

    # Citation itself is valid.
    assert report.citations_valid

    assert report.claim_count == 1

    assert report.citation_coverage == 1.0

    # Current lexical retriever has no shared
    # Chinese/English terms after normalization.
    #
    # Failure to retrieve is NOT proof that
    # the underlying claim is factually false.

    assert report.candidate_count == 0

    assert report.anchor_coverage == 0.0

    assert report.missing_candidate_claim_ids == (
        "C001",
    )

    assert not report.structural_checks_passed
