"""
Analysis: per-document topic distributions and corpus-level topic trends.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gensim.models import LdaModel


def per_document_topic_distribution(
    doc_topics: list[list[tuple[int, float]]],
    num_topics: int,
) -> list[dict[int, float]]:
    """
    Convert raw document-topic lists to normalized distributions (topic_id -> proportion).

    Args:
        doc_topics: Output of model.get_document_topics per document.
        num_topics: Total number of topics in the model.

    Returns:
        List of dicts: for each doc, topic_id -> probability.
    """
    result = []
    for topics in doc_topics:
        dist = {i: 0.0 for i in range(num_topics)}
        total = sum(p for _, p in topics)
        if total > 0:
            for tid, p in topics:
                dist[tid] = p / total
        result.append(dist)
    return result


def dominant_topic_per_document(
    doc_topic_distributions: list[dict[int, float]],
) -> list[tuple[int, float]]:
    """
    For each document, return (dominant_topic_id, proportion).

    Returns:
        List of (topic_id, proportion) for the dominant topic per document.
    """
    return [
        max(dist.items(), key=lambda x: x[1])
        for dist in doc_topic_distributions
    ]


def global_topic_prevalence(
    doc_topic_distributions: list[dict[int, float]],
    num_topics: int,
) -> dict[int, float]:
    """
    Aggregate across the corpus: average proportion per topic (global trend).

    Returns:
        topic_id -> mean proportion across documents.
    """
    sums = defaultdict(float)
    n = len(doc_topic_distributions)
    if n == 0:
        return {i: 0.0 for i in range(num_topics)}
    for dist in doc_topic_distributions:
        for tid, p in dist.items():
            sums[tid] += p
    return {tid: sums[tid] / n for tid in range(num_topics)}


def topic_prevalence_by_group(
    doc_topic_distributions: list[dict[int, float]],
    groups: list[str | int],
    num_topics: int,
) -> dict[str | int, dict[int, float]]:
    """
    Aggregate topic prevalence per group (e.g. per label or time bucket).

    Args:
        doc_topic_distributions: Per-document topic distributions.
        groups: One group label per document (same length as doc_topic_distributions).
        num_topics: Number of topics.

    Returns:
        group_id -> { topic_id -> mean proportion }.
    """
    by_group: dict[str | int, list[dict[int, float]]] = defaultdict(list)
    for dist, g in zip(doc_topic_distributions, groups):
        by_group[g].append(dist)
    result = {}
    for g, dists in by_group.items():
        result[g] = global_topic_prevalence(dists, num_topics)
    return result
