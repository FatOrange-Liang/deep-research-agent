#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate candidate labels against frozen LangGraph passage index and source snapshots.

This is structural validation only. Human review must decide whether evidence
semantically supports claims; text existence is not entailment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SPLIT = re.compile(
    r'(?<=[。！？；!?])\s*'
    r'|(?<=[.!?])\s+(?=[A-Z0-9"“])'
    r'|\n{2,}'
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def extract_passages(
    content: str, min_quote_chars: int = 24, max_quote_chars: int = 450
) -> list[str]:
    """Must match EvidenceCandidateRetriever._extract_passages (V1)."""
    results: list[str] = []
    for segment in SPLIT.split(content):
        cleaned = re.sub(r"\s+", " ", segment).strip()
        if len(cleaned) < min_quote_chars:
            continue
        if len(cleaned) <= max_quote_chars:
            results.append(cleaned)
            continue
        stride = max(1, max_quote_chars // 2)
        for start in range(0, len(cleaned), stride):
            excerpt = cleaned[start:start + max_quote_chars].strip()
            if len(excerpt) >= min_quote_chars:
                results.append(excerpt)
            if start + max_quote_chars >= len(cleaned):
                break
    return results


def validate(dataset: dict, index: dict, *, require_approved: bool = False) -> None:
    sources = {s["url"]: s for s in dataset["sources"]}
    docs = {d["url"]: d for d in index["documents"]}
    if len(sources) != len(dataset["sources"]):
        raise ValueError("Duplicate dataset source URL.")
    if set(sources) != set(docs):
        raise ValueError("Dataset source URLs and passage-index URLs differ.")

    passage_id_index: dict[tuple[str, str], str] = {}
    for url, src in sources.items():
        doc = docs[url]
        if doc.get("truncated"):
            raise ValueError(f"Truncated source must not be labeled: {url}")
        digest = hashlib.sha256(src["content"].encode("utf-8")).hexdigest()
        if src.get("content_sha256") != digest:
            raise ValueError(f"Source snapshot checksum mismatch: {url}")

        resegmented = extract_passages(src["content"])
        from_index = [p["text"] for p in doc["passages"]]
        if resegmented != from_index:
            raise ValueError(f"Passage segmentation differs from saved index: {url}")
        for p in doc["passages"]:
            key = (url, p["passage_id"])
            if key in passage_id_index:
                raise ValueError(f"Duplicate passage ID: {key}")
            passage_id_index[key] = p["text"]

    if not dataset["examples"]:
        raise ValueError("No examples.")

    seen_ids: set[str] = set()
    n_gold = 0
    pending_gold = 0
    groups = Counter()
    for e in dataset["examples"]:
        eid = e["id"]
        if eid in seen_ids:
            raise ValueError(f"Duplicate example ID: {eid}")
        seen_ids.add(eid)
        if e.get("language") not in {"zh", "en"} or not e["claim"].strip():
            raise ValueError(f"{eid}: missing claim or invalid language")
        groups[(e["topic"], e["language"])] += 1
        cited = set(e["cited_urls"])
        if not cited or not cited.issubset(sources):
            raise ValueError(f"{eid}: invalid cited_urls")
        if not e["gold_evidence"]:
            raise ValueError(f"{eid}: no gold_evidence")
        if require_approved and e.get("annotation_status") != "approved":
            raise ValueError(f"{eid}: claim has not been approved by a human reviewer")

        seen_gold: set[tuple[str, str]] = set()
        for g in e["gold_evidence"]:
            key = (g["url"], g["passage_id"])
            if g["url"] not in cited:
                raise ValueError(f"{eid}: gold URL is outside cited_urls: {key}")
            expected = passage_id_index.get(key)
            if expected is None or expected != g["quote"]:
                raise ValueError(f"{eid}: gold quote differs from saved passage: {key}")
            if key in seen_gold:
                raise ValueError(f"{eid}: duplicate gold passage: {key}")
            seen_gold.add(key)
            if normalize(g["quote"]) not in normalize(sources[g["url"]]["content"]):
                raise ValueError(f"{eid}: gold quote not found in source snapshot: {key}")
            if g.get("review_status") != "approved":
                pending_gold += 1
                if require_approved:
                    raise ValueError(f"{eid}: gold {key} has not been approved")
            n_gold += 1

    print("STRUCTURAL_ANNOTATION_VALIDATION_PASSED")
    print(f"Sources: {len(sources)} | Passage index: {len(passage_id_index)}")
    print(f"Examples: {len(seen_ids)} | Proposed gold passages: {n_gold}")
    print("Groups:", dict(sorted((f"{topic}/{lang}", n) for (topic, lang), n in groups.items())))
    print(f"Gold passages awaiting human approval: {pending_gold}")
    if pending_gold:
        print("NOTE: Do not report retrieval scores as human-validated gold yet.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--require-approved", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    passages = json.loads(args.passages.read_text(encoding="utf-8-sig"))
    validate(data, passages, require_approved=args.require_approved)


if __name__ == "__main__":
    main()
