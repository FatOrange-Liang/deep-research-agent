#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re

from collections import Counter
from pathlib import Path

from deepresearch.research.passage_quality import (
    assess_passage,
    code_score,
)


def normalize(text: str) -> str:
    return " ".join(
        text.lower().split()
    )

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--passages",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/eval/"
            "passage_quality_audit.json"
        ),
    )

    args = parser.parse_args()

    with args.passages.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rows = []

    normalized_map = {}

    reason_counter = Counter()

    total = 0

    for document in data["documents"]:

        for passage in document[
            "passages"
        ]:

            total += 1

            text = passage["text"]

            assessment = assess_passage(
                text
            )

            reasons = list(
                assessment.reasons
            )

            for reason in reasons:
                reason_counter[
                    reason
                ] += 1

            key = normalize(text)

            normalized_map.setdefault(
                key,
                [],
            ).append(
                passage["passage_id"]
            )

            rows.append(
                {
                    "passage_id":
                        passage[
                            "passage_id"
                        ],

                    "title":
                        document["title"],

                    "url":
                        document["url"],

                    "text":
                        text,

                    "length":
                        len(text),

                    "word_count":
                        len(
                            text.split()
                        ),

                    "code_score":
                        code_score(text),

                    "reasons":
                        reasons,

                    "suspicious":
                        bool(reasons),

                    "quality_tier":
                        assessment.tier,

                    "keep":
                        assessment.keep,
                }
            )

    duplicate_groups = [
        passage_ids
        for passage_ids
        in normalized_map.values()
        if len(passage_ids) > 1
    ]

    suspicious_rows = [
        row
        for row in rows
        if row["suspicious"]
    ]

    output = {
        "summary": {
            "total_passages":
                total,

            "suspicious_passages":
                len(
                    suspicious_rows
                ),

            "suspicious_ratio":
                (
                    len(
                        suspicious_rows
                    )
                    / total
                    if total
                    else 0.0
                ),

            "reason_counts":
                dict(
                    reason_counter
                ),

            "duplicate_groups":
                len(
                    duplicate_groups
                ),
        },

        "duplicate_passage_ids":
            duplicate_groups,

        "suspicious_passages":
            suspicious_rows,
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
        "=" * 60
    )

    print(
        "PASSAGE QUALITY AUDIT"
    )

    print(
        "=" * 60
    )

    print(
        "Total passages:",
        total,
    )

    print(
        "Suspicious passages:",
        len(suspicious_rows),
    )

    print(
        "Suspicious ratio:",
        (
            f"{len(suspicious_rows) / total:.2%}"
            if total
            else "0.00%"
        ),
    )

    print(
        "Reason counts:"
    )

    for reason, count in sorted(
        reason_counter.items()
    ):
        print(
            f"  {reason:<20} "
            f"{count}"
        )

    print(
        "Duplicate groups:",
        len(
            duplicate_groups
        ),
    )

    print()

    print(
        "Output:",
        args.output,
    )

    print(
        "PASSAGE_QUALITY_AUDIT_COMPLETE"
    )


if __name__ == "__main__":
    main()