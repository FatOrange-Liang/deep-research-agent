
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a harder synthetic retrieval benchmark.

Preserves:
- Original 8 queries
- Original gold quotes
- Original source URLs

Adds:
- Lexically similar distractor passages
- Alternative mechanisms
- Negation and condition differences
"""

import json

from pathlib import Path


INPUT_PATH = Path(
    "data/eval/retrieval_smoke_multi_source.json"
)

OUTPUT_PATH = Path(
    "data/eval/retrieval_smoke_hard.json"
)


GRAPH_DISTRACTORS = [
    (
        "A checkpointer may archive configuration values "
        "without persisting the current graph state."
    ),
    (
        "Some graph runtimes save state only when execution "
        "ends, rather than after each execution step."
    ),
    (
        "Interrupt handling may resume automatically "
        "without requesting human review."
    ),
    (
        "A human can inspect finished runs without "
        "pausing graph execution."
    ),
    (
        "Thread identifiers can be used as analytics "
        "tags without preserving a conversation."
    ),
    (
        "Streaming sometimes sends only a final result "
        "rather than intermediate updates."
    ),
    (
        "A stored checkpoint can be reset before "
        "a thread is resumed."
    ),
    (
        "Monitoring tools may emit event streams "
        "unrelated to application execution."
    ),
]


AGENT_DISTRACTORS = [
    (
        "A tool catalog may contain descriptions "
        "without executable tools or input schemas."
    ),
    (
        "Some agent runtimes discard tool output "
        "instead of adding it to message history."
    ),
    (
        "A citation parser can accept syntactically "
        "valid identifiers that are absent from "
        "the evidence store."
    ),
    (
        "Evidence search may retrieve unverified "
        "snippets rather than stored full-page text."
    ),
    (
        "Input schemas can be generated without "
        "registering any callable tools."
    ),
    (
        "A model may include tool results in a summary "
        "without recording individual messages."
    ),
    (
        "A citation repair routine can reformat markers "
        "without checking source existence."
    ),
    (
        "Candidate evidence might be newly generated "
        "text rather than a source excerpt."
    ),
]


def main():

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)

    for source in dataset["sources"]:

        url = source["url"]

        if url == (
            "https://example.org/synthetic-graph"
        ):

            distractors = GRAPH_DISTRACTORS

        elif url == (
            "https://example.org/synthetic-agent"
        ):

            distractors = AGENT_DISTRACTORS

        else:

            raise ValueError(
                f"Unexpected source URL: {url}"
            )

        # Keep the original gold evidence unchanged.
        # Append distractors as independent sentences.

        source["content"] = (
            source["content"].rstrip()
            + "\n"
            + "\n".join(distractors)
        )

    dataset["name"] = (
        "bilingual_retrieval_hard_smoke_v1"
    )

    dataset["description"] = (
        "Synthetic retrieval benchmark with "
        "lexically similar hard negatives."
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Created:",
        OUTPUT_PATH,
    )

    print(
        "Examples:",
        len(dataset["examples"]),
    )

    print(
        "Additional distractors:",
        len(GRAPH_DISTRACTORS)
        + len(AGENT_DISTRACTORS),
    )


if __name__ == "__main__":
    main()
