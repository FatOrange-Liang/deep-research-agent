#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path


INPUT = Path(
    "artifacts/eval/gold_completeness_audit.json"
)

OUTPUT = Path(
    "artifacts/eval/gold_completeness_audit_reviewed.json"
)


# (example_id, passage_id):
#     (support_label, add_to_gold, review_note)
LABELS = {
    # Q001
    ("Q001", "D01_P0036"): (
        "not_supported",
        False,
        "Mentions agent abstractions built on LangGraph but does not establish that LangGraph focuses on low-level agent orchestration.",
    ),
    ("Q001", "D01_P0004"): (
        "supported",
        False,
        "Existing approved gold; directly states that LangGraph is low-level and focused on agent orchestration.",
    ),
    ("Q001", "D01_P0012"): (
        "not_supported",
        False,
        "Describes LangChain rather than LangGraph's role.",
    ),
    ("Q001", "D01_P0008"): (
        "partial",
        False,
        "Supports the agent-orchestration focus but does not independently establish the complete low-level characterization.",
    ),
    ("Q001", "D01_P0032"): (
        "not_supported",
        False,
        "Discusses prototyping and production observability rather than LangGraph's low-level orchestration role.",
    ),

    # Q005
    ("Q005", "D02_P0003"): (
        "supported",
        False,
        "Existing approved gold; explicitly states that checkpointers persist a thread's graph state as checkpoints.",
    ),
    ("Q005", "D02_P0017"): (
        "partial",
        False,
        "Discusses persistent checkpointers and checkpoint accumulation but does not fully state the claim.",
    ),
    ("Q005", "D02_P0021"): (
        "partial",
        False,
        "States that checkpointers persist thread state but does not independently state the full checkpoint formulation.",
    ),
    ("Q005", "D03_P0008"): (
        "supported",
        True,
        "Independently states that the checkpointer writes exact graph state for later resumption and uses thread_id to identify the state.",
    ),
    ("Q005", "D03_P0072"): (
        "not_supported",
        False,
        "Fragmented code passage; does not independently support the complete claim.",
    ),
    ("Q005", "D02_P0019"): (
        "not_supported",
        False,
        "Primarily concerns pruning and retention of checkpoints.",
    ),

    # Q009
    ("Q009", "D03_P0001"): (
        "supported",
        False,
        "Existing approved gold; directly states pause, waiting for external input, and continuation.",
    ),
    ("Q009", "D03_P0123"): (
        "partial",
        False,
        "Supports saving state and waiting for external input but does not independently establish the complete pause-and-continue claim.",
    ),
    ("Q009", "D03_P0023"): (
        "not_supported",
        False,
        "Only describes how graph.invoke surfaces interrupts.",
    ),
    ("Q009", "D03_P0124"): (
        "partial",
        False,
        "Describes behavior after execution resumes but not the complete interruption process.",
    ),

    # Q011
    ("Q011", "D03_P0004"): (
        "supported",
        False,
        "Existing approved gold; directly states interrupt() can be called at any point in graph nodes.",
    ),
    ("Q011", "D03_P0135"): (
        "partial",
        False,
        "Shows an interrupt() call but does not establish that it can be placed at any point.",
    ),
    ("Q011", "D03_P0136"): (
        "partial",
        False,
        "Shows an interrupt() call but does not establish that it can be placed at any point.",
    ),
    ("Q011", "D03_P0140"): (
        "partial",
        False,
        "Shows an interrupt() call but does not establish that it can be placed at any point.",
    ),
    ("Q011", "D03_P0166"): (
        "not_supported",
        False,
        "The sentence is too underspecified to support the claim.",
    ),
    ("Q011", "D03_P0056"): (
        "not_supported",
        False,
        "Concerns resuming multiple interrupts rather than where interrupt() can be called.",
    ),
    ("Q011", "D03_P0060"): (
        "not_supported",
        False,
        "Fragmented code passage; does not establish arbitrary interrupt placement.",
    ),
    ("Q011", "D03_P0109"): (
        "partial",
        False,
        "Shows that interrupt() is called inside a node but does not establish arbitrary placement within graph nodes.",
    ),
}


def main():
    with INPUT.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    missing = []

    for example in data["examples"]:
        eid = example["id"]

        for method_candidates in example[
            "candidates"
        ].values():

            for candidate in method_candidates:

                passage_ids = candidate[
                    "passage_ids"
                ]

                labels = []

                for passage_id in passage_ids:
                    key = (eid, passage_id)

                    if key in LABELS:
                        labels.append(
                            LABELS[key]
                        )

                if not labels:
                    missing.append(
                        (
                            eid,
                            tuple(passage_ids),
                        )
                    )
                    continue

                # A candidate may correspond to duplicate
                # passage IDs with identical text.
                #
                # Prefer the strongest reviewed label.
                priority = {
                    "supported": 3,
                    "partial": 2,
                    "not_supported": 1,
                }

                best = max(
                    labels,
                    key=lambda x:
                        priority[x[0]],
                )

                candidate[
                    "support_label"
                ] = best[0]

                candidate[
                    "add_to_gold"
                ] = best[1]

                candidate[
                    "review_note"
                ] = best[2]

        example["audit_status"] = (
            "human_reviewed"
        )

    if missing:
        print(
            "WARNING: UNLABELED CANDIDATES"
        )

        for item in sorted(
            set(missing)
        ):
            print(item)

        raise SystemExit(1)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "GOLD_AUDIT_LABELS_APPLIED"
    )

    print(
        "Output:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()