
import json

from deepresearch.llm import Message

from deepresearch.research import (
    ClaimAnchorVerifier,
    ClaimEvidence,
    ResearchState,
)

from deepresearch.research.sources import (
    make_source_id,
)


URL = "https://docs.example.com/langgraph"

SOURCE_ID = make_source_id(URL)

CLAIM = (
    "LangGraph uses a checkpointer "
    "for state persistence"
)

QUOTE = (
    "A checkpointer saves graph state "
    "after each execution step."
)

ANSWER = (
    f"{CLAIM} [{SOURCE_ID}]."
)


def create_state(
    *,
    read_page: bool = True,
) -> ResearchState:

    state = ResearchState(
        task="Research LangGraph"
    )

    search_data = {
        "query": "LangGraph persistence",
        "results": [
            {
                "source_id": SOURCE_ID,
                "title": "LangGraph Docs",
                "url": URL,
                "content": "Search snippet",
                "score": 0.9,
            }
        ],
    }

    state.ingest_message(
        Message(
            role="tool",
            name="web_search",
            tool_call_id="search_1",
            content=json.dumps(search_data),
        )
    )

    if read_page:

        page_data = {
            "source_id": SOURCE_ID,
            "title": "LangGraph Docs",
            "url": URL,
            "content": (
                "LangGraph persistence documentation. "
                + QUOTE
                + " Interrupts enable human input."
            ),
            "content_type": "text/html",
            "truncated": False,
        }

        state.ingest_message(
            Message(
                role="tool",
                name="web_page_reader",
                tool_call_id="read_1",
                content=json.dumps(page_data),
            )
        )

    return state


def make_item(
    *,
    claim: str = CLAIM,
    source_id: str = SOURCE_ID,
    quote: str = QUOTE,
) -> ClaimEvidence:

    return ClaimEvidence(
        claim=claim,
        source_id=source_id,
        evidence_quote=quote,
    )


def test_valid_claim_anchor() -> None:

    verifier = ClaimAnchorVerifier()

    report = verifier.verify(
        answer=ANSWER,
        state=create_state(),
        items=[make_item()],
    )

    assert report.all_anchored

    assert report.anchored_count == 1

    assert report.checks[0].status == "anchored"


def test_hallucinated_quote_is_rejected() -> None:

    verifier = ClaimAnchorVerifier()

    report = verifier.verify(
        answer=ANSWER,
        state=create_state(),
        items=[
            make_item(
                quote=(
                    "LangGraph exclusively uses "
                    "MySQL for all persistence."
                )
            )
        ],
    )

    assert not report.all_anchored

    assert (
        report.checks[0].status
        == "quote_not_found"
    )


def test_search_snippet_is_not_full_evidence() -> None:

    verifier = ClaimAnchorVerifier()

    report = verifier.verify(
        answer=ANSWER,
        state=create_state(read_page=False),
        items=[make_item()],
    )

    assert not report.all_anchored

    assert (
        report.checks[0].status
        == "source_not_read"
    )


def test_unknown_source_is_rejected() -> None:

    unknown_id = "S_deadbeef"

    verifier = ClaimAnchorVerifier()

    report = verifier.verify(
        answer=f"{CLAIM} [{unknown_id}].",
        state=create_state(),
        items=[
            make_item(
                source_id=unknown_id
            )
        ],
    )

    assert not report.all_anchored

    assert (
        report.checks[0].status
        == "source_not_found"
    )


def test_detached_citation_is_rejected() -> None:

    verifier = ClaimAnchorVerifier()

    report = verifier.verify(
        answer=(
            f"{CLAIM}. "
            f"A different statement [{SOURCE_ID}]."
        ),
        state=create_state(),
        items=[make_item()],
    )

    assert not report.all_anchored

    assert (
        report.checks[0].status
        == "citation_not_attached"
    )


def test_generic_short_quote_is_rejected() -> None:

    verifier = ClaimAnchorVerifier(
        min_quote_chars=24
    )

    report = verifier.verify(
        answer=ANSWER,
        state=create_state(),
        items=[
            make_item(
                quote="LangGraph"
            )
        ],
    )

    assert not report.all_anchored

    assert (
        report.checks[0].status
        == "quote_too_short"
    )


def test_empty_manifest_does_not_pass() -> None:

    verifier = ClaimAnchorVerifier()

    report = verifier.verify(
        answer=ANSWER,
        state=create_state(),
        items=[],
    )

    assert not report.all_anchored

    assert report.anchored_count == 0
