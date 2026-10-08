"""Unit tests for scripts/benchmark_retrieval_v2.py. No HF download/API required."""
from __future__ import annotations

import hashlib

import pytest

from scripts.benchmark_retrieval_v2 import (
    candidate_urls,
    cosine_similarity,
    dense_ranking_cached,
    first_gold_rank,
    group_metrics,
    rrf_fuse,
    summarize_ranks,
    validate_dataset,
)
from deepresearch.research import EvidenceCandidateRetriever
from deepresearch.research.claim_anchors import normalize_text
from deepresearch.research.sources import make_source_id


def fixture_data():
    u1 = "https://example.org/a"
    u2 = "https://example.org/b"
    a = "Checkpointing persists graph state for a thread."
    b = "Stores preserve application data across different threads."
    s1 = {"title": "A", "url": u1, "content": a,
          "content_sha256": hashlib.sha256(a.encode()).hexdigest()}
    s2 = {"title": "B", "url": u2, "content": b,
          "content_sha256": hashlib.sha256(b.encode()).hexdigest()}
    index = {"documents": [
        {"url": u1, "truncated": False,
         "passages": [{"passage_id": "D01_P0001", "text": a}]},
        {"url": u2, "truncated": False,
         "passages": [{"passage_id": "D02_P0001", "text": b}]},
    ]}
    dataset = {"name": "small_fixture", "sources": [s1, s2], "examples": [
        {"id": "Q001", "claim": "Checkpoint saves state", "language": "en",
         "topic": "A", "cited_urls": [u1], "annotation_status": "approved",
         "gold_evidence": [{"url": u1, "passage_id": "D01_P0001",
                            "quote": a, "review_status": "approved"}]}
    ]}
    return dataset, index


def test_first_gold_rank_multiple_golds():
    a, b, c = ("S1", "a"), ("S1", "b"), ("S1", "c")
    assert first_gold_rank([a, b, c], {b, c}) == 2
    assert first_gold_rank([a, b, c], {c}) == 3
    assert first_gold_rank([a, b], {c}) is None


def test_multi_gold_metrics():
    assert summarize_ranks([2, 1, None, 3]) == {
        "n": 4, "recall_at_1": 0.25, "recall_at_3": 0.75,
        "mrr": pytest.approx((0.5 + 1 + 0 + 1/3) / 4),
    }


def test_rrf_deduplicates_and_rejects_invalid_k():
    a, b = ("S1", "a"), ("S1", "b")
    assert rrf_fuse([a, a, b], [b, a])[0] == a  # a: 1 + 2; b: 3 + 1
    assert set(rrf_fuse([a], [b])) == {a, b}
    with pytest.raises(ValueError, match="positive"):
        rrf_fuse([a], k=0)


def test_group_metrics_separates_languages_and_topics():
    rows = [
        {"topic": "Overview", "language": "zh", "lexical_rank": None,
         "dense_rank": 2, "hybrid_rrf_rank": 2},
        {"topic": "Overview", "language": "en", "lexical_rank": 1,
         "dense_rank": 1, "hybrid_rrf_rank": 1},
        {"topic": "Interrupts", "language": "zh", "lexical_rank": None,
         "dense_rank": None, "hybrid_rrf_rank": 3},
    ]
    stats = group_metrics(rows)
    assert stats["overall"]["dense"]["recall_at_3"] == pytest.approx(2/3)
    assert stats["by_language"]["zh"]["dense"]["recall_at_3"] == pytest.approx(0.5)
    assert stats["by_language"]["en"]["dense"]["recall_at_1"] == 1.0
    assert stats["by_topic_language"]["Overview/zh"]["dense"]["mrr"] == 0.5
    assert stats["by_topic"]["Interrupts"]["hybrid_rrf"]["mrr"] == pytest.approx(1/3)


def test_validate_dataset_accepts_approved_exact_passage():
    data, index = fixture_data()
    sources, keys = validate_dataset(data, index, EvidenceCandidateRetriever())
    assert len(sources) == 2
    assert (make_source_id(data["sources"][0]["url"]),
            normalize_text(data["examples"][0]["gold_evidence"][0]["quote"])) in keys


def test_validate_dataset_rejects_pending_gold_and_pending_claim():
    data, index = fixture_data()
    data["examples"][0]["gold_evidence"][0]["review_status"] = "pending"
    with pytest.raises(ValueError, match="gold not human approved"):
        validate_dataset(data, index, EvidenceCandidateRetriever())
    data["examples"][0]["gold_evidence"][0]["review_status"] = "approved"
    data["examples"][0]["annotation_status"] = "candidate_pending_human_review"
    with pytest.raises(ValueError, match="claim is not human approved"):
        validate_dataset(data, index, EvidenceCandidateRetriever())


def test_validate_dataset_rejects_invented_quote_or_bad_passage_id():
    data, index = fixture_data()
    data["examples"][0]["gold_evidence"][0]["quote"] = "An invented passage."
    with pytest.raises(ValueError, match="gold text/ID"):
        validate_dataset(data, index, EvidenceCandidateRetriever())
    data, index = fixture_data()
    data["examples"][0]["gold_evidence"][0]["passage_id"] = "MISSING"
    with pytest.raises(ValueError, match="gold text/ID"):
        validate_dataset(data, index, EvidenceCandidateRetriever())


def test_validate_dataset_rejects_mutated_snapshot_and_index():
    data, index = fixture_data()
    data["sources"][0]["content"] += " tampered"
    with pytest.raises(ValueError, match="SHA256"):
        validate_dataset(data, index, EvidenceCandidateRetriever())
    data, index = fixture_data()
    index["documents"][0]["passages"][0]["text"] = "Altered passage."
    with pytest.raises(ValueError, match="segmentation"):
        validate_dataset(data, index, EvidenceCandidateRetriever())


def test_scope_all_does_not_leak_gold_source():
    data, _ = fixture_data()
    sources = {source["url"]: source for source in data["sources"]}
    example = data["examples"][0]
    assert len(candidate_urls(example, sources, "all")) == 2
    assert len(candidate_urls(example, sources, "cited")) == 1
    with pytest.raises(ValueError, match="Invalid candidate scope"):
        candidate_urls(example, sources, "unknown")


def test_cached_dense_scoring_respects_scope_and_deduplicates():
    pool = [
        {"source_id": "S_a", "url": "a", "quote": "gold one", "key": ("S_a", "gold one")},
        {"source_id": "S_a", "url": "a", "quote": "gold one", "key": ("S_a", "gold one")},
        {"source_id": "S_b", "url": "b", "quote": "negative", "key": ("S_b", "negative")},
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert dense_ranking_cached([1.0, 0.0], pool, vectors, {"a", "b"}) == [
        ("S_a", "gold one"), ("S_b", "negative")]
    assert dense_ranking_cached([1.0, 0.0], pool, vectors, {"b"}) == [
        ("S_b", "negative")]
    with pytest.raises(ValueError, match="[Nn]umber of embedding vectors"):
        dense_ranking_cached([1.0, 0.0], pool, vectors[:1], {"a", "b"})


def test_cosine_robustness():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="dimension"):
        cosine_similarity([1], [1, 2])
    with pytest.raises(ValueError, match="non-finite"):
        cosine_similarity([float("nan")], [1])
