"""
Topic modeling via LDA (Latent Dirichlet Allocation) using Gensim.

Each topic is a distribution over words; each document is a distribution over topics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gensim.models import LdaModel, LdaMulticore
from gensim.corpora import Dictionary

if TYPE_CHECKING:
    from gensim.models import TfidfModel


def train_lda(
    corpus: list[list[tuple[int, int]]] | list[list[tuple[int, float]]],
    id2word: Dictionary,
    num_topics: int = 10,
    alpha: str | float = "auto",
    eta: str | float = "auto",
    passes: int = 10,
    use_multicore: bool = True,
    workers: int | None = None,
    random_state: int | None = 42,
) -> LdaModel:
    """
    Train LDA on a BoW or TF-IDF corpus.

    Args:
        corpus: List of document vectors (BoW or TF-IDF).
        id2word: Gensim Dictionary mapping id -> token.
        num_topics: Number of latent topics.
        alpha: Document-topic prior ('auto' or float).
        eta: Topic-word prior ('auto' or float).
        passes: Number of passes over the corpus.
        use_multicore: Use LdaMulticore for speed.
        workers: Number of workers (default: num_cpus - 1).
        random_state: Seed for reproducibility.

    Returns:
        Fitted LdaModel.
    """
    kwargs = dict(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        alpha=alpha,
        eta=eta,
        passes=passes,
        random_state=random_state,
    )
    if use_multicore:
        model = LdaMulticore(workers=workers, **kwargs)
    else:
        model = LdaModel(**kwargs)
    return model


def get_document_topics(
    model: LdaModel,
    corpus: list[list[tuple[int, int]]] | list[list[tuple[int, float]]],
    minimum_probability: float | None = 0.0,
) -> list[list[tuple[int, float]]]:
    """
    Get per-document topic distributions.

    Returns:
        For each document, list of (topic_id, probability) sorted by probability.
    """
    return [
        model.get_document_topics(doc, minimum_probability=minimum_probability or 0.0)
        for doc in corpus
    ]
