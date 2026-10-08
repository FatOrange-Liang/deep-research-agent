from deepresearch.research.reranker import (
    MultilingualEvidenceReranker,
    RerankedPassage,
)


def test_reranked_passage():
    row = RerankedPassage(
        index=2,
        score=0.9,
    )

    assert row.index == 2
    assert row.score == 0.9


def test_reranker_is_lazy_before_first_score():
    reranker = MultilingualEvidenceReranker(
        device="cpu",
    )

    assert reranker.is_loaded is False
    assert reranker.device == "cpu"


def test_rerank_uses_scores_without_real_model():
    reranker = object.__new__(
        MultilingualEvidenceReranker
    )

    reranker.score = lambda claim, passages: [
        0.1,
        0.9,
        0.4,
    ]

    result = reranker.rerank(
        "claim",
        [
            "A",
            "B",
            "C",
        ],
    )

    assert [
        row.index
        for row in result
    ] == [1, 2, 0]
