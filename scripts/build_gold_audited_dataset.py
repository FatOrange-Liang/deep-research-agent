#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import json
from pathlib import Path


BASE_DATASET = Path(
    "data/eval/retrieval_bilingual_v1_12.json"
)

PASSAGES_PATH = Path(
    "artifacts/eval/langgraph_passages.json"
)

OUTPUT_DATASET = Path(
    "data/eval/retrieval_bilingual_v1_12_gold_audited.json"
)


# These passages were manually reviewed and judged
# to independently support the complete claim.
APPROVED_ADDITIONS = {
    "Q001": [
        "D01_P0002",
    ],
    "Q005": [
        "D03_P0008",
    ],
    "Q009": [
        "D03_P0003",
        "D03_P0012",
    ],
}


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_passage_index(
    passages_data: dict,
) -> dict[str, dict]:

    index = {}

    for document in passages_data["documents"]:

        url = document["url"]

        for passage in document["passages"]:

            passage_id = passage["passage_id"]

            if passage_id in index:
                raise ValueError(
                    f"Duplicate passage ID: "
                    f"{passage_id}"
                )

            index[passage_id] = {
                "url": url,
                "quote": passage["text"],
            }

    return index


def main() -> None:

    dataset = load_json(
        BASE_DATASET
    )

    passages_data = load_json(
        PASSAGES_PATH
    )

    passage_index = build_passage_index(
        passages_data
    )

    output = copy.deepcopy(
        dataset
    )

    example_index = {
        example["id"]: example
        for example in output["examples"]
    }

    added_count = 0

    for example_id, passage_ids in (
        APPROVED_ADDITIONS.items()
    ):

        if example_id not in example_index:
            raise ValueError(
                f"Unknown example ID: "
                f"{example_id}"
            )

        example = example_index[
            example_id
        ]

        existing_ids = {
            evidence["passage_id"]
            for evidence
            in example["gold_evidence"]
        }

        for passage_id in passage_ids:

            if passage_id in existing_ids:

                print(
                    "Already gold:",
                    example_id,
                    passage_id,
                )

                continue

            if passage_id not in passage_index:

                raise ValueError(
                    f"Unknown passage ID: "
                    f"{passage_id}"
                )

            passage = passage_index[
                passage_id
            ]

                        # Keep source provenance consistent:
            # every gold evidence source must also
            # appear in cited_urls.
            gold_url = passage["url"]

            cited_urls = example.setdefault(
                "cited_urls",
                [],
            )

            if gold_url not in cited_urls:
                cited_urls.append(
                    gold_url
                )

                print(
                    "Added cited URL:",
                    example_id,
                    gold_url,
                )

            evidence = {
                "url":
                    passage["url"],

                "passage_id":
                    passage_id,

                "quote":
                    passage["quote"],

                "review_status":
                    "approved",

                "review_note":
                    (
                        "Added during "
                        "gold-completeness audit; "
                        "independently supports "
                        "the complete claim."
                    ),
            }

            example[
                "gold_evidence"
            ].append(
                evidence
            )

            existing_ids.add(
                passage_id
            )

            added_count += 1

            print(
                "Added:",
                example_id,
                passage_id,
            )

        # The complete example remains
        # human-reviewed.
        example[
            "annotation_status"
        ] = "approved"

    # Record benchmark revision metadata.
    output["name"] = (
        "retrieval_bilingual_v1_12_gold_audited"
    )

    output[
        "description"
    ] = (
        "Human-reviewed bilingual retrieval "
        "benchmark after gold-completeness audit."
    )

    output[
        "annotation_revision"
    ] = "gold_completeness_v1"

    output[
        "annotation_notes"
    ] = (
        "Initial approved gold annotations were "
        "preserved. Four independently supporting "
        "passages were added after reviewing "
        "retrieval errors and additional candidate "
        "passages."
    )

    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_DATASET.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    total_gold = sum(
        len(example["gold_evidence"])
        for example in output["examples"]
    )

    print()
    print(
        "GOLD_AUDITED_DATASET_CREATED"
    )

    print(
        "Added gold passages:",
        added_count,
    )

    print(
        "Total gold passages:",
        total_gold,
    )

    print(
        "Output:",
        OUTPUT_DATASET,
    )


if __name__ == "__main__":
    main()