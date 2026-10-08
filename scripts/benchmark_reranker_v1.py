#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline benchmark for semantic evidence reranking.

Pipeline
--------
Hybrid RRF Top-K
    -> multilingual cross-encoder reranker
    -> reranked evidence ranking

The candidate generator is frozen. This benchmark measures only
whether semantic reranking improves the ordering of already-retrieved
candidate passages.
"""

from __future__ import annotations

import argparse
import json

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from deepresearch.research.reranker import (
    MultilingualEvidenceReranker,
)


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object: {path}"
        )

    return data


def summarize_ranks(
    ranks: Sequence[int | None],
) -> dict[str, float | int]:

    if not ranks:
        raise ValueError(
            "Cannot summarize empty ranks."
        )

    n = len(ranks)

    return {
        "n":
            n,

        "hit_at_1":
            sum(
                rank == 1
                for rank in ranks
            ) / n,

        "hit_at_3":
            sum(
                rank is not None
                and rank <= 3
                for rank in ranks
            ) / n,

        "hit_at_10":
            sum(
                rank is not None
                and rank <= 10
                for rank in ranks
            ) / n,

        "mrr":
            sum(
                1.0 / rank
                if rank is not None
                else 0.0
                for rank in ranks
            ) / n,
    }


def metric_pair(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:

    return {
        "baseline_hybrid":
            summarize_ranks(
                [
                    row[
                        "baseline_topk_rank"
                    ]
                    for row in rows
                ]
            ),

        "reranked":
            summarize_ranks(
                [
                    row[
                        "reranked_rank"
                    ]
                    for row in rows
                ]
            ),
    }


def group_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:

    if not rows:
        raise ValueError(
            "No benchmark rows."
        )

    by_language = defaultdict(list)
    by_topic = defaultdict(list)

    for row in rows:

        by_language[
            row["language"]
        ].append(row)

        by_topic[
            row["topic"]
        ].append(row)

    return {
        "overall":
            metric_pair(rows),

        "by_language": {
            key:
                metric_pair(
                    by_language[key]
                )
            for key in sorted(
                by_language
            )
        },

        "by_topic": {
            key:
                metric_pair(
                    by_topic[key]
                )
            for key in sorted(
                by_topic
            )
        },
    }


def validate_inputs(
    dataset: dict[str, Any],
    retrieval_result: dict[str, Any],
    *,
    candidate_method: str,
    rerank_top_k: int,
) -> None:

    examples = dataset.get(
        "examples"
    )

    result_rows = retrieval_result.get(
        "per_example"
    )

    if (
        not isinstance(examples, list)
        or not examples
    ):
        raise ValueError(
            "Dataset has no examples."
        )

    if (
        not isinstance(result_rows, list)
        or not result_rows
    ):
        raise ValueError(
            "Retrieval result has no "
            "per_example rows."
        )

    dataset_by_id = {
        example["id"]: example
        for example in examples
    }

    result_by_id = {
        row["id"]: row
        for row in result_rows
    }

    if set(dataset_by_id) != set(
        result_by_id
    ):
        raise ValueError(
            "Dataset IDs differ from "
            "retrieval-result IDs."
        )

    for example_id, example in (
        dataset_by_id.items()
    ):

        row = result_by_id[
            example_id
        ]

        if (
            row.get("claim")
            != example.get("claim")
        ):
            raise ValueError(
                f"{example_id}: claim mismatch."
            )

        top_candidates = row.get(
            "top_candidates"
        )

        if not isinstance(
            top_candidates,
            dict,
        ):
            raise ValueError(
                f"{example_id}: missing "
                "top_candidates."
            )

        candidates = top_candidates.get(
            candidate_method
        )

        if not isinstance(
            candidates,
            list,
        ):
            raise ValueError(
                f"{example_id}: missing "
                f"{candidate_method} candidates."
            )

        if len(candidates) < rerank_top_k:

            raise ValueError(
                f"{example_id}: only "
                f"{len(candidates)} stored "
                f"{candidate_method} candidates; "
                f"need {rerank_top_k}. "
                "Re-run benchmark_retrieval_v2.py "
                f"with --top-n {rerank_top_k}."
            )


def run_benchmark(
    *,
    dataset: dict[str, Any],
    retrieval_result: dict[str, Any],
    reranker: MultilingualEvidenceReranker,
    candidate_method: str,
    rerank_top_k: int,
) -> dict[str, Any]:

    dataset_by_id = {
        example["id"]: example
        for example in dataset["examples"]
    }

    rows: list[
        dict[str, Any]
    ] = []

    for retrieval_row in (
        retrieval_result[
            "per_example"
        ]
    ):

        example_id = retrieval_row[
            "id"
        ]

        example = dataset_by_id[
            example_id
        ]

        claim = example[
            "claim"
        ]

        candidates = (
            retrieval_row[
                "top_candidates"
            ][candidate_method][
                :rerank_top_k
            ]
        )

        passages = [
            candidate["quote"]
            for candidate in candidates
        ]

        scores = reranker.score(
            claim,
            passages,
        )

        if len(scores) != len(
            candidates
        ):
            raise ValueError(
                f"{example_id}: reranker "
                "returned wrong number "
                "of scores."
            )

        # Stable tie-breaking:
        # if scores are equal, preserve
        # original retrieval order.
        order = sorted(
            range(
                len(candidates)
            ),
            key=lambda index: (
                -scores[index],
                candidates[index][
                    "rank"
                ],
            ),
        )

        reranked_candidates = []

        reranked_rank = None

        for new_rank, index in enumerate(
            order,
            start=1,
        ):

            candidate = candidates[
                index
            ]

            is_gold = bool(
                candidate[
                    "is_gold"
                ]
            )

            if (
                reranked_rank is None
                and is_gold
            ):
                reranked_rank = (
                    new_rank
                )

            reranked_candidates.append(
                {
                    "reranked_rank":
                        new_rank,

                    "original_rank":
                        candidate[
                            "rank"
                        ],

                    "score":
                        float(
                            scores[index]
                        ),

                    "is_gold":
                        is_gold,

                    "source_id":
                        candidate[
                            "source_id"
                        ],

                    "url":
                        candidate[
                            "url"
                        ],

                    "passage_ids":
                        candidate[
                            "passage_ids"
                        ],

                    "quote":
                        candidate[
                            "quote"
                        ],
                }
            )

        full_baseline_rank = (
            retrieval_row.get(
                f"{candidate_method}_rank"
            )
        )

        baseline_topk_rank = (
            full_baseline_rank

            if (
                full_baseline_rank
                is not None
                and full_baseline_rank
                <= rerank_top_k
            )

            else None
        )

        row = {
            "id":
                example_id,

            "language":
                example["language"],

            "topic":
                example["topic"],

            "claim":
                claim,

            "candidate_method":
                candidate_method,

            "candidate_top_k":
                rerank_top_k,

            "baseline_full_rank":
                full_baseline_rank,

            "baseline_topk_rank":
                baseline_topk_rank,

            "reranked_rank":
                reranked_rank,

            "reranked_candidates":
                reranked_candidates,
        }

        rows.append(row)

        print(
            f"[{example_id} | "
            f"{row['topic']}/"
            f"{row['language']}] "
            f"Hybrid="
            f"{full_baseline_rank} "
            f"-> Reranked="
            f"{reranked_rank}"
        )

    return {
        "dataset":
            dataset.get(
                "name",
                "unnamed",
            ),

        "retrieval_dataset":
            retrieval_result.get(
                "dataset"
            ),

        "candidate_method":
            candidate_method,

        "candidate_top_k":
            rerank_top_k,

        "num_examples":
            len(rows),

        "metric_definition":
            (
                "Query-level Hit@K and "
                "MRR of first any-gold "
                "passage. Reranker only "
                "reorders the frozen "
                "candidate Top-K."
            ),

        "metrics":
            group_metrics(
                rows
            ),

        "per_example":
            rows,
    }


def print_metrics(
    metrics: dict[str, Any],
) -> None:

    print()
    print("=" * 76)
    print(
        "SEMANTIC EVIDENCE RERANKER "
        "BENCHMARK"
    )
    print("=" * 76)

    overall = metrics[
        "overall"
    ]

    print()
    print("OVERALL")

    print(
        f"{'Method':<22}"
        f"{'N':>5}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
    )

    for name in (
        "baseline_hybrid",
        "reranked",
    ):

        row = overall[name]

        print(
            f"{name:<22}"
            f"{row['n']:>5}"
            f"{row['hit_at_1']:>10.4f}"
            f"{row['hit_at_3']:>10.4f}"
            f"{row['hit_at_10']:>10.4f}"
            f"{row['mrr']:>10.4f}"
        )

    for language, group in (
        metrics[
            "by_language"
        ].items()
    ):

        print()
        print(
            f"LANGUAGE: {language}"
        )

        for name in (
            "baseline_hybrid",
            "reranked",
        ):

            row = group[name]

            print(
                f"{name:<22}"
                f"Hit@1="
                f"{row['hit_at_1']:.4f} "
                f"Hit@3="
                f"{row['hit_at_3']:.4f} "
                f"MRR="
                f"{row['mrr']:.4f}"
            )

    for topic, group in (
        metrics[
            "by_topic"
        ].items()
    ):

        print()
        print(
            f"TOPIC: {topic}"
        )

        for name in (
            "baseline_hybrid",
            "reranked",
        ):

            row = group[name]

            print(
                f"{name:<22}"
                f"Hit@1="
                f"{row['hit_at_1']:.4f} "
                f"Hit@3="
                f"{row['hit_at_3']:.4f} "
                f"MRR="
                f"{row['mrr']:.4f}"
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--retrieval-result",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        default=(
            "BAAI/"
            "bge-reranker-v2-m3"
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--candidate-method",
        default="hybrid_rrf",
        choices=(
            "lexical",
            "dense",
            "hybrid_rrf",
        ),
    )

    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/eval/"
            "reranker_v1_result.json"
        ),
    )

    args = parser.parse_args()

    if args.rerank_top_k < 1:
        raise ValueError(
            "--rerank-top-k must "
            "be positive."
        )

    dataset = load_json(
        args.dataset
    )

    retrieval_result = load_json(
        args.retrieval_result
    )

    validate_inputs(
        dataset,
        retrieval_result,
        candidate_method=(
            args.candidate_method
        ),
        rerank_top_k=(
            args.rerank_top_k
        ),
    )

    print(
        "RERANKER_INPUT_VALIDATED"
    )

    print(
        "Examples:",
        len(
            dataset["examples"]
        ),
    )

    print(
        "Candidate method:",
        args.candidate_method,
    )

    print(
        "Rerank Top-K:",
        args.rerank_top_k,
    )

    print()
    print(
        "Loading semantic reranker..."
    )

    reranker = (
        MultilingualEvidenceReranker(
            model_name=args.model,
            device=args.device,
        )
    )

    result = run_benchmark(
        dataset=dataset,
        retrieval_result=(
            retrieval_result
        ),
        reranker=reranker,
        candidate_method=(
            args.candidate_method
        ),
        rerank_top_k=(
            args.rerank_top_k
        ),
    )

    result["model"] = args.model
    result["device"] = args.device

    print_metrics(
        result["metrics"]
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Result saved to:",
        args.output,
    )

    print(
        "RERANKER_BENCHMARK_COMPLETE"
    )


if __name__ == "__main__":
    main()