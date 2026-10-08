#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-gold bilingual retrieval benchmark over frozen official-document snapshots.

V2 is a NEW script. It does not edit scripts/benchmark_retrieval.py (V1).

Default candidate scope is ALL frozen sources, rather than each question's
cited_urls, to avoid giving the retriever a gold-source hint at evaluation.

Recall@K here uses the V1 query-level HIT@K convention: success when at least
one approved gold passage is retrieved within K, not fraction of all golds.

The source snapshot and passage index are intentionally supplied together;
review_status="approved" is required for every evaluated gold passage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from deepresearch.llm import Message
from deepresearch.research import (
    ClaimUnit,
    EvidenceCandidateRetriever,
    MultilingualE5Embedder,
    ResearchState,
)
from deepresearch.research.claim_anchors import normalize_text
from deepresearch.research.sources import make_source_id

CandidateKey = tuple[str, str]  # (stable source ID, normalized exact passage)
METHODS = ("lexical", "dense", "hybrid_rrf")


def candidate_key(candidate: Any) -> CandidateKey:
    return candidate.source_id, normalize_text(candidate.quote)


def rrf_fuse(*rankings: Sequence[CandidateKey], k: int = 60) -> list[CandidateKey]:
    """Deterministic reciprocal-rank fusion, deduplicating within each ranking."""
    if k < 1:
        raise ValueError("'k' must be positive.")
    scores: dict[CandidateKey, float] = {}
    for ranking in rankings:
        seen: set[CandidateKey] = set()
        for position, key in enumerate(ranking, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position)
    return sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))


def first_gold_rank(
    ranking: Sequence[CandidateKey], gold_keys: set[CandidateKey]
) -> int | None:
    """Position of the first ANY-gold hit (1-indexed), or None."""
    for rank, key in enumerate(ranking, start=1):
        if key in gold_keys:
            return rank
    return None


def summarize_ranks(ranks: Sequence[int | None]) -> dict[str, float | int]:
    if not ranks:
        raise ValueError("Cannot evaluate an empty dataset.")
    count = len(ranks)
    return {
        "n": count,
        "recall_at_1": sum(r == 1 for r in ranks) / count,
        "recall_at_3": sum(r is not None and r <= 3 for r in ranks) / count,
        "mrr": sum(1.0 / r if r is not None else 0.0 for r in ranks) / count,
    }


def group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall, language, topic, and topic/language metrics."""
    if not rows:
        raise ValueError("No per-example results.")

    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            method: summarize_ranks([row[f"{method}_rank"] for row in group])
            for method in METHODS
        }

    categories: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_language": defaultdict(list),
        "by_topic": defaultdict(list),
        "by_topic_language": defaultdict(list),
    }
    for row in rows:
        categories["by_language"][row["language"]].append(row)
        categories["by_topic"][row["topic"]].append(row)
        categories["by_topic_language"][
            f"{row['topic']}/{row['language']}"
        ].append(row)
    result: dict[str, Any] = {"overall": summarize(rows)}
    for name, buckets in categories.items():
        result[name] = {key: summarize(buckets[key]) for key in sorted(buckets)}
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def validate_dataset(
    dataset: dict[str, Any],
    passage_index: dict[str, Any],
    segmenter: EvidenceCandidateRetriever,
) -> tuple[dict[str, dict[str, Any]], dict[CandidateKey, dict[str, Any]]]:
    """Reject bad snapshots, passage mismatches, and unapproved/invalid golds.

    `passage_index` must be the separately frozen artifacts/eval/langgraph_passages.json.
    Only structural validity and declared approval are checked, NOT entailment.
    """
    sources_list = dataset.get("sources")
    docs_list = passage_index.get("documents")
    examples = dataset.get("examples")
    if not isinstance(sources_list, list) or not sources_list:
        raise ValueError("Dataset has no sources.")
    if not isinstance(docs_list, list) or not docs_list:
        raise ValueError("Passage index has no documents.")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Dataset has no examples.")
    if not all(isinstance(s, dict) and s.get("url") for s in sources_list):
        raise ValueError("Invalid dataset source.")
    if not all(isinstance(d, dict) and d.get("url") for d in docs_list):
        raise ValueError("Invalid passage-index document.")

    sources = {s["url"]: s for s in sources_list}
    docs = {d["url"]: d for d in docs_list}
    if len(sources) != len(sources_list) or len(docs) != len(docs_list):
        raise ValueError("Duplicate source URL or index document URL.")
    if set(sources) != set(docs):
        raise ValueError("Dataset sources and passage-index URLs differ.")

    by_passage_id: dict[tuple[str, str], str] = {}
    key_metadata: dict[CandidateKey, dict[str, Any]] = {}
    for url, source in sources.items():
        document = docs[url]
        if document.get("truncated"):
            raise ValueError(f"Truncated document is in passage index: {url}")
        content = source.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty or invalid source content: {url}")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if source.get("content_sha256") != digest:
            raise ValueError(f"Frozen snapshot SHA256 mismatch: {url}")
        indexed = document.get("passages")
        if not isinstance(indexed, list) or not indexed:
            raise ValueError(f"Missing passages in index: {url}")
        current = segmenter.extract_passages(content)
        if current != [p["text"] for p in indexed]:
            raise ValueError(f"Passage segmentation differs from frozen index: {url}")
        for passage in indexed:
            passage_id = passage["passage_id"]
            quote = passage["text"]
            pkey = (url, passage_id)
            if pkey in by_passage_id:
                raise ValueError(f"Duplicate passage ID: {pkey}")
            by_passage_id[pkey] = quote
            key = make_source_id(url), normalize_text(quote)
            if key not in key_metadata:
                key_metadata[key] = {
                    "source_id": key[0],
                    "url": url,
                    "quote": quote,
                    "passage_ids": [],
                }
            key_metadata[key]["passage_ids"].append(passage_id)

    seen_ids: set[str] = set()
    for e in examples:
        eid = e.get("id")
        if not isinstance(eid, str) or not eid.strip() or eid in seen_ids:
            raise ValueError(f"Missing or duplicate example ID: {eid!r}")
        seen_ids.add(eid)
        if e.get("language") not in {"zh", "en"}:
            raise ValueError(f"{eid}: language must be zh or en.")
        if not isinstance(e.get("topic"), str) or not e["topic"].strip():
            raise ValueError(f"{eid}: missing topic.")
        if not isinstance(e.get("claim"), str) or not e["claim"].strip():
            raise ValueError(f"{eid}: missing claim.")
        if e.get("annotation_status") != "approved":
            raise ValueError(f"{eid}: claim is not human approved.")
        cited_urls = e.get("cited_urls")
        if not isinstance(cited_urls, list) or not cited_urls:
            raise ValueError(f"{eid}: cited_urls must be a nonempty list.")
        if len(set(cited_urls)) != len(cited_urls) or not set(cited_urls) <= sources.keys():
            raise ValueError(f"{eid}: cited_urls have duplicates or unknown URLs.")
        golds = e.get("gold_evidence")
        if not isinstance(golds, list) or not golds:
            raise ValueError(f"{eid}: no gold evidence.")
        seen_gold: set[tuple[str, str]] = set()
        for gold in golds:
            pkey = gold.get("url"), gold.get("passage_id")
            if pkey[0] not in cited_urls:
                raise ValueError(f"{eid}: gold URL not included in cited_urls: {pkey}")
            if pkey in seen_gold:
                raise ValueError(f"{eid}: duplicate gold passage: {pkey}")
            seen_gold.add(pkey)
            if pkey not in by_passage_id or gold.get("quote") != by_passage_id[pkey]:
                raise ValueError(f"{eid}: gold text/ID not an exact indexed passage: {pkey}")
            if gold.get("review_status") != "approved":
                raise ValueError(f"{eid}: gold not human approved: {pkey}")
            key = make_source_id(pkey[0]), normalize_text(gold["quote"])
            if key not in key_metadata:
                raise ValueError(f"{eid}: gold cannot occur in shared candidate pool: {pkey}")
    return sources, key_metadata


def build_research_state(dataset: dict[str, Any]) -> ResearchState:
    """Use the same stored-source ingestion pattern as the existing V1 benchmark."""
    state = ResearchState(task="Offline multi-gold bilingual retrieval benchmark")
    search_results = []
    for source in dataset["sources"]:
        url = source["url"]
        search_results.append({
            "source_id": make_source_id(url), "title": source["title"],
            "url": url, "content": source["content"][:300], "score": 1.0,
        })
    search_message = Message(
        role="tool", name="web_search", tool_call_id="benchmark_v2_search",
        content=json.dumps({"query": "retrieval benchmark v2", "results": search_results},
                           ensure_ascii=False),
    )
    if not state.ingest_message(search_message):
        raise RuntimeError("Failed to ingest offline search results.")
    for idx, source in enumerate(dataset["sources"], start=1):
        url = source["url"]
        page_message = Message(
            role="tool", name="web_page_reader", tool_call_id=f"benchmark_v2_page_{idx}",
            content=json.dumps({
                "source_id": make_source_id(url), "title": source["title"],
                "url": url, "content": source["content"],
                "content_type": "text/html", "truncated": False,
            }, ensure_ascii=False),
        )
        if not state.ingest_message(page_message):
            raise RuntimeError(f"Failed to ingest offline page: {url}")
    return state


def candidate_urls(example: dict[str, Any], sources: dict[str, Any], scope: str) -> list[str]:
    """Default `all` avoids leaking the known relevant source through cited_urls."""
    if scope == "all":
        return list(sources)
    if scope == "cited":
        return list(example["cited_urls"])
    raise ValueError(f"Invalid candidate scope: {scope}")


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Same cosine scoring used by DenseEvidenceRetriever; vectors can be cached."""
    if len(left) != len(right):
        raise ValueError("Embedding dimension mismatch.")
    if not left:
        return 0.0
    a, b = [float(x) for x in left], [float(x) for x in right]
    if not all(math.isfinite(v) for v in a + b):
        raise ValueError("Embedding contains non-finite values.")
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    value = sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
    return max(-1.0, min(1.0, value))


def dense_ranking_cached(
    query_vector: Sequence[float],
    pool: Sequence[dict[str, Any]],
    vectors: Sequence[Sequence[float]],
    allowed_urls: set[str],
) -> list[CandidateKey]:
    """Rank all allowed passages; sort/dedup like V1 DenseEvidenceRetriever."""
    if len(pool) != len(vectors):
        raise ValueError("Number of embedding vectors differs from passage pool.")
    scored = []
    for passage, vector in zip(pool, vectors):
        if passage["url"] not in allowed_urls:
            continue
        scored.append((cosine_similarity(query_vector, vector), passage))
    scored.sort(key=lambda item: (
        -item[0], item[1]["source_id"], item[1]["quote"]
    ))
    output: list[CandidateKey] = []
    seen: set[CandidateKey] = set()
    for _, passage in scored:
        key = passage["key"]
        if key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def run_benchmark(
    *,
    dataset: dict[str, Any],
    passage_index: dict[str, Any],
    state: ResearchState,
    source_index: dict[str, dict[str, Any]],
    key_metadata: dict[CandidateKey, dict[str, Any]],
    model_name: str,
    device: str | None,
    scope: str = "all",
    rrf_k: int = 60,
    top_n: int = 3,
    embedder: Any = None,
) -> dict[str, Any]:
    if top_n < 1 or rrf_k < 1:
        raise ValueError("top_n and rrf_k must be positive.")
    segmenter = EvidenceCandidateRetriever()
    pool: list[dict[str, Any]] = []
    for source in dataset["sources"]:
        url = source["url"]
        source_id = make_source_id(url)
        for quote in segmenter.extract_passages(source["content"]):
            pool.append({
                "source_id": source_id, "url": url, "quote": quote,
                "key": (source_id, normalize_text(quote)),
            })
    if not pool:
        raise ValueError("Empty retrieval passage pool.")
    if embedder is None:
        print("\nLoading multilingual embedding model...")
        embedder = MultilingualE5Embedder(model_name=model_name, device=device)
    print(f"Encoding {len(pool)} passages once (cached across claims)...")
    passage_vectors = embedder.encode_passages([p["quote"] for p in pool])
    query_vectors = embedder.encode_queries([e["claim"] for e in dataset["examples"]])
    if len(passage_vectors) != len(pool):
        raise ValueError("Embedding backend returned wrong number of passage vectors.")
    if len(query_vectors) != len(dataset["examples"]):
        raise ValueError("Embedding backend returned wrong number of query vectors.")
    rows: list[dict[str, Any]] = []
    lexical = EvidenceCandidateRetriever(
        top_k_per_claim=len(pool), min_lexical_score=0.0
    )
    for example, query_vector in zip(dataset["examples"], query_vectors):
        eid = example["id"]
        allowed = candidate_urls(example, source_index, scope)
        claim = ClaimUnit(
            claim_id=eid, text=example["claim"],
            cited_source_ids=tuple(make_source_id(url) for url in allowed),
        )
        lexical_candidates = lexical.retrieve(claims=[claim], state=state)[eid]
        lexical_keys = [candidate_key(candidate) for candidate in lexical_candidates]
        dense_keys = dense_ranking_cached(
            query_vector, pool, passage_vectors, set(allowed)
        )
        hybrid_keys = rrf_fuse(lexical_keys, dense_keys, k=rrf_k)
        rankings = {
            "lexical": lexical_keys,
            "dense": dense_keys,
            "hybrid_rrf": hybrid_keys,
        }
        gold_keys: set[CandidateKey] = {
            (make_source_id(g["url"]), normalize_text(g["quote"]))
            for g in example["gold_evidence"]
        }
        row: dict[str, Any] = {
            "id": eid,
            "language": example["language"],
            "topic": example["topic"],
            "claim": example["claim"],
            "candidate_scope": scope,
            "candidate_source_count": len(allowed),
            "gold_passages": [
                {"url": g["url"], "passage_id": g["passage_id"]}
                for g in example["gold_evidence"]
            ],
            "top_candidates": {},
        }
        for method, ranking in rankings.items():
            row[f"{method}_rank"] = first_gold_rank(ranking, gold_keys)
            row["top_candidates"][method] = [
                {**key_metadata[key], "rank": rank, "is_gold": key in gold_keys}
                for rank, key in enumerate(ranking[:top_n], start=1)
            ]
        rows.append(row)
        print(
            f"[{eid} | {row['topic']}/{row['language']}] "
            f"Lexical={row['lexical_rank']} | "
            f"Dense={row['dense_rank']} | "
            f"Hybrid={row['hybrid_rrf_rank']}"
        )
    return {
        "dataset": dataset.get("name", "unnamed"),
        "model": model_name,
        "candidate_scope": scope,
        "rrf_k": rrf_k,
        "num_sources": len(source_index),
        "num_passages": len(pool),
        "num_examples": len(rows),
        "num_approved_gold_passages": sum(len(e["gold_evidence"]) for e in dataset["examples"]),
        "metric_definition": "query-level Hit@1 / Hit@3 (named recall_at_1/3 for V1 compatibility); MRR of first any-gold hit",
        "metrics": group_metrics(rows),
        "per_example": rows,
    }


def print_metric_table(label: str, metric_group: dict[str, Any]) -> None:
    print(f"\n{label}")
    print(f"{'Method':<18}{'N':>6}{'Recall@1':>12}{'Recall@3':>12}{'MRR':>12}")
    for method in METHODS:
        m = metric_group[method]
        print(
            f"{method:<18}{m['n']:>6}"
            f"{m['recall_at_1']:>12.4f}{m['recall_at_3']:>12.4f}"
            f"{m['mrr']:>12.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--passages", required=True, type=Path,
                        help="Frozen artifacts/eval/langgraph_passages.json")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scope", choices=("all", "cited"), default="all",
                        help="all is default to avoid gold-source leakage")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/eval/retrieval_bilingual_v1_12_result.json"))
    args = parser.parse_args()
    dataset = load_json(args.dataset)
    index = load_json(args.passages)
    segmenter = EvidenceCandidateRetriever()
    source_index, key_metadata = validate_dataset(dataset, index, segmenter)
    n_gold = sum(len(e["gold_evidence"]) for e in dataset["examples"])
    print("MULTI_GOLD_DATASET_VALIDATED")
    print(f"Sources: {len(source_index)} | Passages: {len(key_metadata)} unique keys")
    print(f"Examples: {len(dataset['examples'])} | Approved gold passages: {n_gold}")
    print(f"Candidate scope: {args.scope}")
    if args.validate_only:
        print("VALIDATION_ONLY_COMPLETE")
        return
    state = build_research_state(dataset)
    result = run_benchmark(
        dataset=dataset, passage_index=index, state=state,
        source_index=source_index, key_metadata=key_metadata,
        model_name=args.model, device=args.device, scope=args.scope,
        rrf_k=args.rrf_k, top_n=args.top_n,
    )
    print("\n" + "=" * 72)
    print("MULTI-GOLD BILINGUAL RETRIEVAL BENCHMARK V2")
    print("=" * 72)
    print_metric_table("OVERALL", result["metrics"]["overall"])
    for language, group in result["metrics"]["by_language"].items():
        print_metric_table(f"LANGUAGE: {language}", group)
    for topic, group in result["metrics"]["by_topic"].items():
        print_metric_table(f"TOPIC: {topic}", group)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"\nResult saved to: {args.output}")
    print("BENCHMARK_V2_COMPLETE")


if __name__ == "__main__":
    main()
