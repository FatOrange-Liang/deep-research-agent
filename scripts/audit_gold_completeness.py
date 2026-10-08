#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_IDS = {
    "Q001",
    "Q005",
    "Q009",
    "Q011",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/eval/gold_completeness_audit.json"
        ),
    )

    args = parser.parse_args()

    result = load_json(args.result)
    dataset = load_json(args.dataset)

    dataset_index = {
        example["id"]: example
        for example in dataset["examples"]
    }

    audit_rows = []

    for row in result["per_example"]:

        example_id = row["id"]

        if example_id not in TARGET_IDS:
            continue

        original = dataset_index[example_id]

        gold_ids = {
            evidence["passage_id"]
            for evidence in original["gold_evidence"]
        }

        candidates = {}

        for method in (
            "lexical",
            "dense",
            "hybrid_rrf",
        ):
            method_rows = []

            for candidate in row[
                "top_candidates"
            ].get(method, [])[:5]:

                passage_ids = candidate.get(
                    "passage_ids",
                    [],
                )

                method_rows.append(
                    {
                        "rank":
                            candidate["rank"],

                        "url":
                            candidate["url"],

                        "passage_ids":
                            passage_ids,

                        "quote":
                            candidate["quote"],

                        "currently_gold":
                            any(
                                pid in gold_ids
                                for pid
                                in passage_ids
                            ),

                        # Fill manually:
                        #
                        # supported:
                        #   independently supports
                        #   the complete claim
                        #
                        # partial:
                        #   relevant, but does not
                        #   support the full claim
                        #
                        # not_supported:
                        #   does not support claim
                        #
                        "support_label":
                            "pending",

                        # Set true only after
                        # human review.
                        "add_to_gold":
                            None,

                        "review_note":
                            "",
                    }
                )

            candidates[method] = method_rows

        audit_rows.append(
            {
                "id": example_id,
                "language":
                    row["language"],
                "topic":
                    row["topic"],
                "claim":
                    row["claim"],

                "current_gold_evidence":
                    original["gold_evidence"],

                "retrieval_ranks": {
                    "lexical":
                        row["lexical_rank"],
                    "dense":
                        row["dense_rank"],
                    "hybrid_rrf":
                        row["hybrid_rrf_rank"],
                },

                "candidates":
                    candidates,

                "audit_status":
                    "pending_human_review",
            }
        )

    output = {
        "name":
            "gold_completeness_audit_v1",

        "instructions": {
            "supported":
                "Candidate independently supports "
                "the complete claim.",

            "partial":
                "Candidate is relevant but does not "
                "support the complete claim.",

            "not_supported":
                "Candidate does not support the claim.",

            "add_to_gold":
                "Set true only for independently "
                "supporting passages after human review.",
        },

        "examples":
            audit_rows,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "GOLD_COMPLETENESS_AUDIT_CREATED"
    )

    print(
        "Examples:",
        len(audit_rows),
    )

    print(
        "Output:",
        args.output,
    )


if __name__ == "__main__":
    main()