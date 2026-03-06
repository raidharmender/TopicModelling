"""
Document ingestion: preprocessing and corpus building.

Pipeline: raw text → tokenize → filter (stopwords, min/max length) → lemmatize
→ Gensim Dictionary → Corpus (BoW or TF-IDF).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# Ensure NLTK data is available (run once)
try:
    stopwords.words("english")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)


def preprocess_text(
    text: str,
    min_token_len: int = 2,
    max_token_len: int = 50,
    stop_lang: str = "english",
) -> list[str]:
    """
    Tokenize, remove stopwords, and lemmatize a single document.

    Returns:
        List of processed tokens.
    """
    text = text.lower().strip()
    tokens = word_tokenize(text)
    stop = set(stopwords.words(stop_lang))
    lemmatizer = WordNetLemmatizer()

    result = []
    for t in tokens:
        t_clean = re.sub(r"[^a-z0-9]", "", t.lower())
        if not t_clean or len(t_clean) < min_token_len or len(t_clean) > max_token_len:
            continue
        if t_clean in stop:
            continue
        lemma = lemmatizer.lemmatize(t_clean)
        result.append(lemma)
    return result


def build_dictionary(
    tokenized_docs: Iterable[list[str]],
    no_below: int = 2,
    no_above: float = 0.5,
    keep_n: int = 100_000,
) -> Dictionary:
    """
    Build a Gensim Dictionary from tokenized documents.

    Args:
        tokenized_docs: Iterable of token lists (one per document).
        no_below: Drop tokens in fewer than this many docs.
        no_above: Drop tokens in more than this fraction of docs (0–1).
        keep_n: Keep only this many most frequent tokens.

    Returns:
        Gensim Dictionary.
    """
    id2word = Dictionary(tokenized_docs)
    id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    return id2word


def corpus_bow(
    tokenized_docs: Iterable[list[str]],
    id2word: Dictionary,
) -> list[list[tuple[int, int]]]:
    """
    Convert tokenized documents to bag-of-words corpus.

    Returns:
        List of (term_id, count) vectors (one per document).
    """
    return [id2word.doc2bow(doc) for doc in tokenized_docs]


def corpus_tfidf(
    bow_corpus: list[list[tuple[int, int]]],
) -> tuple[list[list[tuple[int, float]]], TfidfModel]:
    """
    Convert BoW corpus to TF-IDF representation.

    Returns:
        (tfidf_corpus, fitted TfidfModel).
    """
    model = TfidfModel(bow_corpus)
    tfidf_corpus = [model[doc] for doc in bow_corpus]
    return tfidf_corpus, model


def load_documents_from_files(paths: Iterable[Path]) -> list[str]:
    """
    Load raw text from files (one document per file).

    Returns:
        List of document strings.
    """
    docs = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            docs.append(p.read_text(encoding="utf-8", errors="replace"))
    return docs


def load_documents_from_lines(path: Path) -> list[str]:
    """
    Load documents from a file (one document per line or paragraph).

    Returns:
        List of document strings.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]
