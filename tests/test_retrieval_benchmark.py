
from types import SimpleNamespace

from scripts.benchmark_retrieval import (
    candidate_key,
    rank_position,
    rrf_fuse,
    summarize_ranks,
)


A = ("S_11111111", "passage a")
B = ("S_11111111", "passage b")
C = ("S_11111111", "passage c")


def test_candidate_key_normalization():

    candidate = SimpleNamespace(
        source_id="S_11111111",
        quote="  Passage   A  ",
    )

    assert candidate_key(candidate) == A


def test_rrf_combines_two_rankings():

    lexical = [A, B, C]

    dense = [B, C, A]

    fused = rrf_fuse(
        lexical,
        dense,
    )

    assert len(fused) == 3

    assert fused[0] == B

    assert set(fused) == {
        A,
        B,
        C,
    }


def test_rrf_deduplicates_candidates():

    ranking = [A, A, B]

    fused = rrf_fuse(
        ranking,
        [A, C],
    )

    assert len(fused) == 3

    assert len(set(fused)) == 3


def test_retrieval_metrics():

    ranks = [
        1,
        2,
        None,
        3,
    ]

    result = summarize_ranks(
        ranks
    )

    assert result["n"] == 4

    assert result["recall_at_1"] == 0.25

    assert result["recall_at_3"] == 0.75

    expected_mrr = (
        1.0
        + 0.5
        + 0.0
        + 1.0 / 3.0
    ) / 4.0

    assert abs(
        result["mrr"] - expected_mrr
    ) < 1e-10

    assert rank_position(
        [A, B, C],
        B,
    ) == 2

    assert rank_position(
        [A, B, C],
        ("S_deadbeef", "missing"),
    ) is None
