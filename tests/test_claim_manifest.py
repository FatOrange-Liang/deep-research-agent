
import json

from deepresearch.llm import Message

from deepresearch.research import (
    ClaimManifestBuilder,
    EvidenceProposal,
    ResearchState,
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

ANSWER = (
    f"{CLAIM_A} [{SOURCE_ID}].\n"
    f"{CLAIM_B} [{SOURCE_ID}]."
)


def create_state() -> ResearchState:

    state = ResearchState(
        task="Research LangGraph"
    )

    search_data = {
        "query": "LangGraph",
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

    state.ingest_message(
        Message(
            role="tool",
            name="web_search",
            tool_call_id="search_1",
            content=json.dumps(search_data),
        )
    )

    page_data = {
        "source_id": SOURCE_ID,
        "title": "Official Docs",
        "url": URL,
        "content": (
            "LangGraph documentation. "
            + QUOTE_A
            + " "
            + QUOTE_B
        ),
        "content_type": "text/html",
        "truncated": False,
    }

    state.ingest_message(
        Message(
            role="tool",
            name="web_page_reader",
            tool_call_id="reader_1",
            content=json.dumps(page_data),
        )
    )

    return state


def proposal_a() -> EvidenceProposal:

    return EvidenceProposal(
        claim_id="C001",
        source_id=SOURCE_ID,
        evidence_quote=QUOTE_A,
    )


def proposal_b() -> EvidenceProposal:

    return EvidenceProposal(
        claim_id="C002",
        source_id=SOURCE_ID,
        evidence_quote=QUOTE_B,
    )


def test_extracts_two_cited_claims() -> None:

    builder = ClaimManifestBuilder()

    claims = builder.extract_claims(
        ANSWER
    )

    assert len(claims) == 2

    assert claims[0].claim_id == "C001"

    assert claims[1].claim_id == "C002"

    assert claims[0].text == CLAIM_A

    assert claims[1].text == CLAIM_B

    assert claims[0].cited_source_ids == (
        SOURCE_ID,
    )


def test_uncited_claim_reduces_coverage() -> None:

    answer = (
        f"{CLAIM_A} [{SOURCE_ID}].\n"
        f"{CLAIM_B}."
    )

    manifest = ClaimManifestBuilder().build(
        answer=answer,
        state=create_state(),
    )

    assert manifest.claim_count == 2

    assert manifest.cited_claim_count == 1

    assert manifest.citation_coverage == 0.5

    assert manifest.uncited_claim_ids == (
        "C002",
    )


def test_no_evidence_proposals_means_zero_anchor_coverage() -> None:

    manifest = ClaimManifestBuilder().build(
        answer=ANSWER,
        state=create_state(),
    )

    assert manifest.claim_count == 2

    assert manifest.citation_coverage == 1.0

    assert manifest.anchor_coverage == 0.0

    assert not manifest.all_claim_units_anchored


def test_valid_proposals_cover_both_claims() -> None:

    manifest = ClaimManifestBuilder().build(
        answer=ANSWER,
        state=create_state(),
        proposals=[
            proposal_a(),
            proposal_b(),
        ],
    )

    assert manifest.claim_count == 2

    assert manifest.anchored_claim_count == 2

    assert manifest.anchor_coverage == 1.0

    assert manifest.all_claim_units_anchored

    assert not manifest.rejected_proposals


def test_one_anchor_cannot_hide_another_claim() -> None:

    manifest = ClaimManifestBuilder().build(
        answer=ANSWER,
        state=create_state(),
        proposals=[
            proposal_a(),
        ],
    )

    assert manifest.claim_count == 2

    assert manifest.anchored_claim_count == 1

    assert manifest.anchor_coverage == 0.5

    assert manifest.unanchored_claim_ids == (
        "C002",
    )

    assert not manifest.all_claim_units_anchored


def test_hallucinated_quote_does_not_count() -> None:

    false_proposal = EvidenceProposal(
        claim_id="C001",
        source_id=SOURCE_ID,
        evidence_quote=(
            "LangGraph uses MySQL as its "
            "only persistence backend."
        ),
    )

    manifest = ClaimManifestBuilder().build(
        answer=ANSWER,
        state=create_state(),
        proposals=[
            false_proposal,
        ],
    )

    assert manifest.anchored_claim_count == 0

    assert (
        manifest.entries[0]
        .anchor_checks[0]
        .status
        == "quote_not_found"
    )


def test_wrong_source_proposal_is_rejected() -> None:

    wrong_proposal = EvidenceProposal(
        claim_id="C001",
        source_id="S_deadbeef",
        evidence_quote=QUOTE_A,
    )

    manifest = ClaimManifestBuilder().build(
        answer=ANSWER,
        state=create_state(),
        proposals=[
            wrong_proposal,
        ],
    )

    assert manifest.anchor_coverage == 0.0

    assert len(
        manifest.rejected_proposals
    ) == 1


def test_unknown_claim_cannot_be_injected() -> None:

    proposal = EvidenceProposal(
        claim_id="C999",
        source_id=SOURCE_ID,
        evidence_quote=QUOTE_A,
    )

    manifest = ClaimManifestBuilder().build(
        answer=ANSWER,
        state=create_state(),
        proposals=[proposal],
    )

    assert manifest.claim_count == 2

    assert len(
        manifest.rejected_proposals
    ) == 1


def test_chinese_citation_after_punctuation() -> None:

    answer = (
        f"它支持状态持久化。[{SOURCE_ID}]"
        "它也支持人工审核。"
    )

    claims = ClaimManifestBuilder().extract_claims(
        answer
    )

    assert len(claims) == 2

    assert claims[0].text == (
        "它支持状态持久化"
    )

    assert claims[0].cited_source_ids == (
        SOURCE_ID,
    )

    assert not claims[1].has_citation


def test_empty_answer_cannot_pass() -> None:

    manifest = ClaimManifestBuilder().build(
        answer="",
        state=create_state(),
    )

    assert manifest.claim_count == 0

    assert manifest.citation_coverage == 0.0

    assert manifest.anchor_coverage == 0.0

    assert not manifest.all_claim_units_anchored
