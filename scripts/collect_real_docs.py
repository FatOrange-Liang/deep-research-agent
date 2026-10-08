
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect real official documentation snapshots.

Outputs:
1. Raw documentation snapshots.
2. Passage index for manual annotation.
3. Draft retrieval benchmark dataset.

The script does not call an LLM.
"""

import json

from datetime import datetime, timezone
from pathlib import Path

from deepresearch.tools import WebPageReaderTool

from deepresearch.research.retriever import (
    EvidenceCandidateRetriever,
)


URLS = [
    (
        "https://docs.langchain.com/"
        "oss/python/langgraph/overview"
    ),
    (
        "https://docs.langchain.com/"
        "oss/python/langgraph/persistence"
    ),
    (
        "https://docs.langchain.com/"
        "oss/python/langgraph/interrupts"
    ),
]


OUTPUT_DIR = Path(
    "artifacts/eval"
)

DATASET_DIR = Path(
    "data/eval"
)


def save_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:

    # Use a larger extraction limit for
    # offline documentation collection.
    #
    # The normal Agent WebPageReaderTool still
    # keeps its default max_chars=20000.

    reader = WebPageReaderTool(
        timeout=30.0,
        max_chars=80_000,
        trust_env=False,
    )

    segmenter = EvidenceCandidateRetriever()

    snapshots = []
    passage_index = []
    benchmark_sources = []

    for index, url in enumerate(
        URLS,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(URLS)}] Reading:"
        )

        print(url)

        try:

            raw = reader.execute(
                url=url
            )

            page = json.loads(
                raw
            )

        except Exception as exc:

            print(
                "READ_FAILED:",
                type(exc).__name__,
                str(exc),
            )

            continue

        content = page.get(
            "content",
            "",
        )

        if not isinstance(content, str):
            print(
                "INVALID_CONTENT"
            )
            continue

        if not content.strip():

            print(
                "EMPTY_CONTENT"
            )

            continue

        title = page.get(
            "title"
        ) or url

        truncated = bool(
            page.get(
                "truncated",
                False,
            )
        )

        retrieved_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        snapshot = {
            "title": title,
            "url": url,
            "retrieved_at": retrieved_at,
            "content": content,
            "truncated": truncated,
        }

        snapshots.append(
            snapshot
        )

        # Extract passages using exactly the same
        # segmenter used by the retrieval benchmark.

        passages = segmenter.extract_passages(
            content
        )

        indexed_passages = [
            {
                "passage_id": (
                    f"D{index:02d}_P{j:04d}"
                ),
                "text": passage,
            }
            for j, passage in enumerate(
                passages,
                start=1,
            )
        ]

        passage_index.append(
            {
                "title": title,
                "url": url,
                "truncated": truncated,
                "passages": indexed_passages,
            }
        )

        # Only non-truncated pages are eligible
        # for the initial full-page benchmark.

        if not truncated:

            benchmark_sources.append(
                {
                    "title": title,
                    "url": url,
                    "content": content,
                }
            )

        print(
            "Characters:",
            len(content),
        )

        print(
            "Passages:",
            len(passages),
        )

        print(
            "Truncated:",
            truncated,
        )

                # =====================================
        # Collection quality diagnostics
        # =====================================

        print(
            "Content beginning:",
            repr(content[:180]),
        )

        print(
            "Content ending:",
            repr(content[-250:]),
        )

        if truncated:

            print(
                "QC WARNING: This page was truncated. "
                "Do not use it as a complete "
                "benchmark source."
            )

    # =====================================
    # Save source snapshots
    # =====================================

    save_json(
        OUTPUT_DIR
        / "langgraph_docs_snapshot.json",
        {
            "name": "langgraph_official_snapshot",
            "sources": snapshots,
        },
    )

    # =====================================
    # Save passage index
    # =====================================

    save_json(
        OUTPUT_DIR
        / "langgraph_passages.json",
        {
            "name": "langgraph_official_passages",
            "documents": passage_index,
        },
    )

    # =====================================
    # Prepare annotation dataset
    # =====================================

    save_json(
        DATASET_DIR
        / "retrieval_bilingual_v1_draft.json",
        {
            "name": "retrieval_bilingual_v1",
            "description": (
                "Real documentation retrieval "
                "benchmark requiring manual "
                "gold evidence annotation."
            ),
            "sources": benchmark_sources,
            "examples": [],
        },
    )

    print()
    print("=" * 60)

    print(
        "COLLECTION_COMPLETE"
    )

    print(
        "Snapshots:",
        len(snapshots),
    )

    print(
        "Eligible full-page sources:",
        len(benchmark_sources),
    )

    print(
        "Indexed passages:",
        sum(
            len(doc["passages"])
            for doc in passage_index
        ),
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
