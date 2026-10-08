#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Passage V2 using conservative passage-quality gating.

Important design rules
----------------------
1. Preserve original passage IDs.
2. Do not re-segment text yet.
3. Remove only passages classified as hard reject.
4. Never silently remove gold evidence.
5. Keep quality metadata for later analysis.

This is an experimental evaluation artifact.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from deepresearch.research.passage_quality import (
    assess_passage,
)


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def collect_gold_ids(
    dataset: dict,
) -> set[str]:

    gold_ids: set[str] = set()

    for example in dataset["examples"]:

        for evidence in example[
            "gold_evidence"
        ]:

            gold_ids.add(
                evidence["passage_id"]
            )

    return gold_ids


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Original passage-index JSON."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Audited multi-gold benchmark "
            "dataset."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output Passage V2 JSON."
        ),
    )

    args = parser.parse_args()

    source = load_json(
        args.input
    )

    dataset = load_json(
        args.dataset
    )

    gold_ids = collect_gold_ids(
        dataset
    )

    output = {
        "name":
            "langgraph_official_passages_v2",

        "segmentation_version":
            "quality_gate_v1",

        "source_passage_file":
            str(args.input),

        "documents": [],
    }

    total = 0
    kept = 0
    rejected = 0

    clean = 0
    review = 0

    rejected_rows = []

    seen_passage_ids: set[str] = set()

    source_passage_ids: set[str] = set()

    kept_passage_ids: set[str] = set()

    # =====================================
    # Process all passages
    # =====================================

    for document in source[
        "documents"
    ]:

        new_passages = []

        for passage in document[
            "passages"
        ]:

            total += 1

            passage_id = passage[
                "passage_id"
            ]

            if passage_id in seen_passage_ids:

                raise ValueError(
                    "Duplicate passage ID: "
                    f"{passage_id}"
                )

            seen_passage_ids.add(
                passage_id
            )

            source_passage_ids.add(
                passage_id
            )

            text = passage["text"]

            assessment = assess_passage(
                text
            )

            # =================================
            # Gold safety gate
            # =================================

            if (
                not assessment.keep
                and passage_id in gold_ids
            ):

                raise ValueError(
                    "GOLD_PASSAGE_WOULD_BE_REJECTED: "
                    f"{passage_id} "
                    f"reasons="
                    f"{assessment.reasons}"
                )

            # =================================
            # Reject malformed passage
            # =================================

            if not assessment.keep:

                rejected += 1

                rejected_rows.append(
                    {
                        "passage_id":
                            passage_id,

                        "title":
                            document.get(
                                "title"
                            ),

                        "url":
                            document["url"],

                        "text":
                            text,

                        "quality_tier":
                            assessment.tier,

                        "reasons":
                            list(
                                assessment.reasons
                            ),
                    }
                )

                continue

            # =================================
            # Keep passage
            # =================================

            kept += 1

            kept_passage_ids.add(
                passage_id
            )

            if (
                assessment.tier
                == "clean"
            ):
                clean += 1

            elif (
                assessment.tier
                == "review"
            ):
                review += 1

            else:

                raise ValueError(
                    "Unexpected quality tier: "
                    f"{assessment.tier}"
                )

            new_passage = dict(
                passage
            )

            new_passage[
                "quality_tier"
            ] = assessment.tier

            new_passage[
                "quality_reasons"
            ] = list(
                assessment.reasons
            )

            new_passages.append(
                new_passage
            )

        new_document = dict(
            document
        )

        new_document[
            "passages"
        ] = new_passages

        output[
            "documents"
        ].append(
            new_document
        )

    # =====================================
    # Gold integrity validation
    # =====================================

    missing_gold_from_source = (
        gold_ids
        - source_passage_ids
    )

    if missing_gold_from_source:

        raise ValueError(
            "Gold passages missing from "
            "original passage index: "
            f"{sorted(missing_gold_from_source)}"
        )

    missing_gold_after_filter = (
        gold_ids
        - kept_passage_ids
    )

    if missing_gold_after_filter:

        raise ValueError(
            "Gold passages missing after "
            "quality filtering: "
            f"{sorted(missing_gold_after_filter)}"
        )

    # =====================================
    # Summary
    # =====================================

    output["quality_summary"] = {
        "total":
            total,

        "kept":
            kept,

        "rejected":
            rejected,

        "rejected_ratio":
            (
                rejected / total
                if total
                else 0.0
            ),

        "clean":
            clean,

        "review":
            review,

        "gold_passages":
            len(gold_ids),

        "gold_passages_preserved":
            len(
                gold_ids
                & kept_passage_ids
            ),
    }

    output[
        "rejected_passages"
    ] = rejected_rows

    # =====================================
    # Write
    # =====================================

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

    # =====================================
    # Console report
    # =====================================

    print("=" * 60)

    print(
        "PASSAGE V2 BUILD"
    )

    print("=" * 60)

    print(
        "Total:",
        total,
    )

    print(
        "Kept:",
        kept,
    )

    print(
        "Rejected:",
        rejected,
    )

    print(
        "Rejected ratio:",
        (
            f"{rejected / total:.2%}"
            if total
            else "0.00%"
        ),
    )

    print(
        "Clean:",
        clean,
    )

    print(
        "Review:",
        review,
    )

    print(
        "Gold passages:",
        len(gold_ids),
    )

    print(
        "Gold preserved:",
        len(
            gold_ids
            & kept_passage_ids
        ),
    )

    print(
        "Output:",
        args.output,
    )

    print(
        "PASSAGE_V2_BUILD_COMPLETE"
    )


if __name__ == "__main__":
    main()