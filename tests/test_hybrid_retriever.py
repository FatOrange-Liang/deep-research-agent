from deepresearch.research import (
    ClaimUnit,
    DenseEvidenceCandidate,
    EvidenceCandidate,
    HybridEvidenceRetriever,
    ResearchState,
)


class FakeEmbedder:
    def encode_queries(self, texts):
        return [[1.0] for _ in texts]

    def encode_passages(self, texts):
        return [[1.0] for _ in texts]


class FakeReranker:
    def __init__(self, scores):
        self.scores = list(scores)

    def score(self, claim, passages):
        assert len(passages) == len(self.scores)
        return list(self.scores)


class FakeLexicalRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve(self, *, claims, state):
        return {
            claim.claim_id: tuple(self.candidates)
            for claim in claims
        }


class FakeDenseRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve(self, *, claims, state):
        return {
            claim.claim_id: tuple(self.candidates)
            for claim in claims
        }


def make_claim():
    return ClaimUnit(
        claim_id="C1",
        text="claim text",
        cited_source_ids=("S1",),
    )


def test_rrf_fusion_deduplicates_and_preserves_best_shared_candidate():
    retriever = HybridEvidenceRetriever(
        embedder=FakeEmbedder(),
        candidate_top_k=3,
        output_top_k=3,
        rrf_k=60,
    )

    shared = EvidenceCandidate(
        claim_id="C1",
        source_id="S1",
        quote="shared passage",
        lexical_score=1.0,
        matched_terms=2,
    )
    lexical_only = EvidenceCandidate(
        claim_id="C1",
        source_id="S1",
        quote="lexical only",
        lexical_score=0.8,
        matched_terms=1,
    )

    dense_shared = DenseEvidenceCandidate(
        claim_id="C1",
        source_id="S1",
        quote="shared passage",
        semantic_score=0.95,
    )
    dense_only = DenseEvidenceCandidate(
        claim_id="C1",
        source_id="S1",
        quote="dense only",
        semantic_score=0.8,
    )

    retriever.lexical_retriever = FakeLexicalRetriever(
        [shared, lexical_only]
    )
    retriever.dense_retriever = FakeDenseRetriever(
        [dense_shared, dense_only]
    )

    result = retriever.retrieve(
        claims=[make_claim()],
        state=ResearchState(task="test"),
    )["C1"]

    assert len(result) == 3
    assert result[0].quote == "shared passage"
    assert result[0].lexical_rank == 1
    assert result[0].dense_rank == 1


def test_semantic_reranker_reorders_fused_top_k():
    retriever = HybridEvidenceRetriever(
        embedder=FakeEmbedder(),
        reranker=FakeReranker(
            [0.1, 0.9, 0.2]
        ),
        candidate_top_k=3,
        output_top_k=2,
        rrf_k=60,
    )

    lexical = [
        EvidenceCandidate(
            claim_id="C1",
            source_id="S1",
            quote="A",
            lexical_score=1.0,
            matched_terms=1,
        ),
        EvidenceCandidate(
            claim_id="C1",
            source_id="S1",
            quote="B",
            lexical_score=0.8,
            matched_terms=1,
        ),
        EvidenceCandidate(
            claim_id="C1",
            source_id="S1",
            quote="C",
            lexical_score=0.7,
            matched_terms=1,
        ),
    ]

    dense = [
        DenseEvidenceCandidate(
            claim_id="C1",
            source_id="S1",
            quote="A",
            semantic_score=0.9,
        ),
        DenseEvidenceCandidate(
            claim_id="C1",
            source_id="S1",
            quote="B",
            semantic_score=0.8,
        ),
        DenseEvidenceCandidate(
            claim_id="C1",
            source_id="S1",
            quote="C",
            semantic_score=0.7,
        ),
    ]

    retriever.lexical_retriever = FakeLexicalRetriever(
        lexical
    )
    retriever.dense_retriever = FakeDenseRetriever(
        dense
    )

    result = retriever.retrieve(
        claims=[make_claim()],
        state=ResearchState(task="test"),
    )["C1"]

    assert [item.quote for item in result] == [
        "B",
        "C",
    ]
    assert result[0].reranker_score == 0.9
    assert result[0].rrf_rank == 2


def test_empty_candidate_sets_return_empty_tuple():
    retriever = HybridEvidenceRetriever(
        embedder=FakeEmbedder(),
        reranker=FakeReranker([]),
        candidate_top_k=10,
        output_top_k=3,
    )

    retriever.lexical_retriever = FakeLexicalRetriever(
        []
    )
    retriever.dense_retriever = FakeDenseRetriever(
        []
    )

    result = retriever.retrieve(
        claims=[make_claim()],
        state=ResearchState(task="test"),
    )

    assert result["C1"] == ()
