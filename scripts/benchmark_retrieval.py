
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline retrieval benchmark.

Methods:
    1. Lexical
    2. Multilingual E5 Dense
    3. Hybrid RRF

All methods use the same stored source content
and the same passage segmentation.

The gold evidence must be manually verified.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Sequence

from deepresearch.llm import Message

from deepresearch.research import (
    ClaimUnit,
    DenseEvidenceRetriever,
    EvidenceCandidateRetriever,
    MultilingualE5Embedder,
    ResearchState,
)

from deepresearch.research.claim_anchors import (
    normalize_text,
)

from deepresearch.research.sources import (
    make_source_id,
)


# =========================================
# 1. Candidate identity
# =========================================

CandidateKey = tuple[str, str]


def candidate_key(candidate) -> CandidateKey:

    return (
        candidate.source_id,
        normalize_text(candidate.quote),
    )


# =========================================
# 2. Reciprocal Rank Fusion
# =========================================

def rrf_fuse(
    *rankings: Sequence[CandidateKey],
    k: int = 60,
) -> list[CandidateKey]:
    """
    Fuse multiple ranked candidate lists.

    A candidate is identified by its source ID
    and normalized original evidence quote.
    """

    if k < 1:
        raise ValueError(
            "'k' must be positive."
        )

    scores: dict[
        CandidateKey,
        float,
    ] = {}

    for ranking in rankings:

        seen: set[CandidateKey] = set()

        for position, key in enumerate(
            ranking,
            start=1,
        ):

            if key in seen:
                continue

            seen.add(key)

            scores[key] = (
                scores.get(key, 0.0)
                + 1.0 / (k + position)
            )

    return sorted(
        scores,
        key=lambda key: (
            -scores[key],
            key[0],
            key[1],
        ),
    )


# =========================================
# 3. Evaluation metrics
# =========================================

def rank_position(
    ranking: Sequence[CandidateKey],
    gold: CandidateKey,
) -> int | None:

    try:
        return ranking.index(gold) + 1

    except ValueError:
        return None


def summarize_ranks(
    ranks: Sequence[int | None],
) -> dict[str, float | int]:

    if not ranks:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    n = len(ranks)

    recall_at_1 = sum(
        rank == 1
        for rank in ranks
    ) / n

    recall_at_3 = sum(
        rank is not None and rank <= 3
        for rank in ranks
    ) / n

    mrr = sum(
        1.0 / rank
        if rank is not None
        else 0.0
        for rank in ranks
    ) / n

    return {
        "n": n,
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "mrr": mrr,
    }


# =========================================
# 4. Dataset and ResearchState
# =========================================

def load_dataset(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)

    if not dataset.get("sources"):
        raise ValueError(
            "Dataset has no sources."
        )

    if not dataset.get("examples"):
        raise ValueError(
            "Dataset has no examples."
        )

    return dataset


def build_research_state(
    dataset: dict,
) -> tuple[
    ResearchState,
    dict[str, dict],
]:

    state = ResearchState(
        task="Offline bilingual retrieval benchmark"
    )

    source_index: dict[str, dict] = {}

    search_results = []

    for source in dataset["sources"]:

        url = source["url"]

        if url in source_index:
            raise ValueError(
                f"Duplicate source URL: {url}"
            )

        if not source["content"].strip():
            raise ValueError(
                f"Empty source content: {url}"
            )

        source_index[url] = source

        search_results.append(
            {
                "source_id": make_source_id(url),
                "title": source["title"],
                "url": url,
                "content": source["content"][:300],
                "score": 1.0,
            }
        )

    # Simulate a successful web search.
    search_message = Message(
        role="tool",
        name="web_search",
        tool_call_id="benchmark_search",
        content=json.dumps(
            {
                "query": "retrieval benchmark",
                "results": search_results,
            },
            ensure_ascii=False,
        ),
    )

    if not state.ingest_message(search_message):
        raise RuntimeError(
            "Failed to ingest benchmark search."
        )

    # Store every page as full-page evidence.
    for index, source in enumerate(
        dataset["sources"],
        start=1,
    ):

        url = source["url"]

        page_message = Message(
            role="tool",
            name="web_page_reader",
            tool_call_id=f"benchmark_page_{index}",
            content=json.dumps(
                {
                    "source_id": make_source_id(url),
                    "title": source["title"],
                    "url": url,
                    "content": source["content"],
                    "content_type": "text/html",
                    "truncated": False,
                },
                ensure_ascii=False,
            ),
        )

        if not state.ingest_message(page_message):
            raise RuntimeError(
                f"Failed to ingest page: {url}"
            )

    return state, source_index


# =========================================
# 5. Gold annotation validation
# =========================================

def validate_examples(
    dataset: dict,
    source_index: dict[str, dict],
    segmenter: EvidenceCandidateRetriever,
) -> None:
    """
    Every gold quote must correspond to an actual
    passage in the shared candidate pool.

    Reject paraphrased or invented gold evidence.
    """

    seen_ids: set[str] = set()

    for example in dataset["examples"]:

        example_id = example["id"]

        if example_id in seen_ids:
            raise ValueError(
                f"Duplicate example ID: {example_id}"
            )

        seen_ids.add(example_id)

        cited_urls = example["cited_urls"]

        gold_url = example["gold_url"]

        if not cited_urls:
            raise ValueError(
                f"{example_id}: no cited URLs."
            )

        for url in cited_urls:

            if url not in source_index:
                raise ValueError(
                    f"{example_id}: unknown cited URL: {url}"
                )

        if gold_url not in cited_urls:
            raise ValueError(
                f"{example_id}: gold URL is not cited."
            )

        gold_quote = normalize_text(
            example["gold_quote"]
        )

        passages = segmenter.extract_passages(
            source_index[gold_url]["content"]
        )

        passage_set = {
            normalize_text(passage)
            for passage in passages
        }

        if gold_quote not in passage_set:

            raise ValueError(
                f"{example_id}: gold_quote is not "
                "an exact passage in the shared "
                "candidate pool. Check segmentation "
                "and source text."
            )


# =========================================
# 6. Main benchmark
# =========================================

def run_benchmark(
    *,
    dataset: dict,
    state: ResearchState,
    source_index: dict[str, dict],
    model_name: str,
    device: str | None,
) -> dict:

    print()
    print("Loading multilingual embedding model...")

    embedder = MultilingualE5Embedder(
        model_name=model_name,
        device=device,
    )

    segmenter = EvidenceCandidateRetriever()

    rows = []

    methods = (
        "lexical",
        "dense",
        "hybrid_rrf",
    )

    for example in dataset["examples"]:

        example_id = example["id"]

        cited_urls = example["cited_urls"]

        cited_source_ids = tuple(
            make_source_id(url)
            for url in cited_urls
        )

        claim = ClaimUnit(
            claim_id=example_id,
            text=example["claim"],
            cited_source_ids=cited_source_ids,
        )

        # Count the complete shared passage pool.
        # This prevents top_k truncation from
        # distorting the benchmark rankings.
        pool_size = sum(
            len(
                segmenter.extract_passages(
                    source_index[url]["content"]
                )
            )
            for url in cited_urls
        )

        pool_size = max(
            1,
            pool_size,
        )

        # ---------------------------------
        # Lexical
        # ---------------------------------

        lexical = EvidenceCandidateRetriever(
            top_k_per_claim=pool_size,
            min_lexical_score=0.0,
        )

        lexical_candidates = lexical.retrieve(
            claims=[claim],
            state=state,
        )[example_id]

        # ---------------------------------
        # Dense E5
        # ---------------------------------

        dense = DenseEvidenceRetriever(
            embedder=embedder,
            top_k_per_claim=pool_size,
            min_similarity=-1.0,
        )

        dense_candidates = dense.retrieve(
            claims=[claim],
            state=state,
        )[example_id]

        # ---------------------------------
        # Convert to common ranked keys
        # ---------------------------------

        lexical_ranking = [
            candidate_key(candidate)
            for candidate in lexical_candidates
        ]

        dense_ranking = [
            candidate_key(candidate)
            for candidate in dense_candidates
        ]

        hybrid_ranking = rrf_fuse(
            lexical_ranking,
            dense_ranking,
        )

        gold = (
            make_source_id(
                example["gold_url"]
            ),
            normalize_text(
                example["gold_quote"]
            ),
        )

        # ---------------------------------
        # Per-example metrics
        # ---------------------------------

        rankings = {
            "lexical": lexical_ranking,
            "dense": dense_ranking,
            "hybrid_rrf": hybrid_ranking,
        }

        
        # =====================================
        # Q001 Hard Negative Diagnostics
        # =====================================

        if example_id == "Q001":

            print()
            print("=" * 70)
            print("Q001 RETRIEVAL DIAGNOSTICS")
            print("=" * 70)

            print(
                "Claim:",
                example["claim"],
            )

            print(
                "Gold evidence:",
                example["gold_quote"],
            )

            print()

            for method, ranking in rankings.items():

                print(
                    f"--- {method.upper()} TOP-3 ---"
                )

                if not ranking:

                    print(
                        "No candidates retrieved."
                    )

                for position, key in enumerate(
                    ranking[:3],
                    start=1,
                ):

                    source_id, quote = key

                    is_gold = (
                        key == gold
                    )

                    print(
                        f"Rank {position}"
                        f" | Gold={is_gold}"
                    )

                    print(
                        "Source:",
                        source_id,
                    )

                    print(
                        "Evidence:",
                        quote,
                    )

                    print()

            print("=" * 70)


        row = {
            "id": example_id,
            "claim": example["claim"],
            "gold_source_id": gold[0],
        }

        for method, ranking in rankings.items():

            row[f"{method}_rank"] = (
                rank_position(
                    ranking,
                    gold,
                )
            )

        rows.append(row)

        print(
            f"[{example_id}] "
            f"Lexical={row['lexical_rank']} | "
            f"Dense={row['dense_rank']} | "
            f"Hybrid={row['hybrid_rrf_rank']}"
        )

    # =====================================
    # Aggregate metrics
    # =====================================

    summary = {}

    for method in methods:

        ranks = [
            row[f"{method}_rank"]
            for row in rows
        ]

        summary[method] = summarize_ranks(
            ranks
        )

    return {
        "dataset": dataset.get(
            "name",
            "unnamed",
        ),
        "model": model_name,
        "num_examples": len(rows),
        "metrics": summary,
        "per_example": rows,
    }


# =========================================
# 7. CLI
# =========================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate annotations without "
            "loading an embedding model."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/eval/retrieval_result.json"
        ),
    )

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset
    )

    state, source_index = build_research_state(
        dataset
    )

    segmenter = EvidenceCandidateRetriever()

    validate_examples(
        dataset,
        source_index,
        segmenter,
    )

    print(
        "Dataset validated:",
        len(dataset["examples"]),
        "examples",
    )

    if args.validate_only:

        print(
            "VALIDATION_ONLY_COMPLETE"
        )

        return

    result = run_benchmark(
        dataset=dataset,
        state=state,
        source_index=source_index,
        model_name=args.model,
        device=args.device,
    )

    # =====================================
    # Print final metrics
    # =====================================

    print()
    print("=" * 65)
    print("RETRIEVAL BENCHMARK")
    print("=" * 65)

    print(
        f"{'Method':<18}"
        f"{'Recall@1':>12}"
        f"{'Recall@3':>12}"
        f"{'MRR':>12}"
    )

    for method, metrics in result["metrics"].items():

        print(
            f"{method:<18}"
            f"{metrics['recall_at_1']:>12.4f}"
            f"{metrics['recall_at_3']:>12.4f}"
            f"{metrics['mrr']:>12.4f}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Result saved to:",
        args.output,
    )

    print(
        "BENCHMARK_COMPLETE"
    )


if __name__ == "__main__":
    main()
