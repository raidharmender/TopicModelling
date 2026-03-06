"""
Visualization: pyLDAvis, matplotlib/seaborn, networkx, Plotly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from gensim.models import LdaModel
    from gensim.corpora import Dictionary


def save_pyldavis(
    model: "LdaModel",
    corpus: list[list[tuple[int, int]]] | list[list[tuple[int, float]]],
    id2word: "Dictionary",
    output_path: Path | str,
) -> None:
    """
    Create interactive pyLDAvis visualization and save to HTML.

    Args:
        model: Fitted LDA model.
        corpus: BoW or TF-IDF corpus used for training.
        id2word: Gensim Dictionary.
        output_path: Path for output HTML file.
    """
    import pyLDAvis
    import pyLDAvis.gensim_models as gensim_vis

    vis = gensim_vis.prepare(model, corpus, id2word)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pyLDAvis.save_html(vis, str(path))


def plot_topic_prevalence_bar(
    topic_prevalence: dict[int, float],
    title: str = "Global topic prevalence",
    output_path: Path | str | None = None,
) -> None:
    """
    Bar chart of topic prevalence (matplotlib/seaborn).
    """
    topics = sorted(topic_prevalence.keys())
    values = [topic_prevalence[t] for t in topics]
    labels = [f"Topic {t}" for t in topics]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=labels, y=values, ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel("Mean proportion")
    ax.set_xlabel("Topic")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def plot_topic_heatmap(
    prevalence_by_group: dict[str | int, dict[int, float]],
    title: str = "Topic prevalence by group",
    output_path: Path | str | None = None,
) -> None:
    """
    Heatmap: groups (rows) x topics (columns), values = proportion (seaborn).
    """
    groups = sorted(prevalence_by_group.keys(), key=str)
    topic_ids = sorted(
        set(t for g in prevalence_by_group for t in prevalence_by_group[g])
    )
    data = np.array([
        [prevalence_by_group[g].get(t, 0.0) for t in topic_ids]
        for g in groups
    ])
    row_labels = [str(g) for g in groups]
    col_labels = [f"T{t}" for t in topic_ids]

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.6), max(4, len(row_labels) * 0.4)))
    sns.heatmap(
        data,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
    )
    ax.set_title(title)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def build_topic_word_network(
    model: "LdaModel",
    id2word: "Dictionary",
    top_n_per_topic: int = 10,
):
    """
    Build a networkx graph: topic and word nodes; edges from topic to top words.

    Returns:
        networkx Graph (topic nodes: 'topic_0', ...; word nodes: token strings).
    """
    import networkx as nx

    G = nx.DiGraph()
    for tid in range(model.num_topics):
        G.add_node(f"topic_{tid}", node_type="topic")
        for wid, prob in model.get_topic_terms(tid, topn=top_n_per_topic):
            word = id2word[wid]
            G.add_node(word, node_type="word")
            G.add_edge(f"topic_{tid}", word, weight=float(prob))
    return G


def plot_topic_word_network(
    model: "LdaModel",
    id2word: "Dictionary",
    top_n_per_topic: int = 5,
    output_path: Path | str | None = None,
) -> None:
    """
    Draw topic–word network with networkx (matplotlib backend).
    """
    import networkx as nx

    G = build_topic_word_network(model, id2word, top_n_per_topic)
    pos = nx.spring_layout(G, k=1.5, seed=42)
    node_colors = [
        "lightcoral" if G.nodes[n].get("node_type") == "topic" else "lightblue"
        for n in G.nodes
    ]
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        ax=ax,
        font_size=8,
        arrows=True,
    )
    ax.set_title("Topic–word network")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def plot_interactive_topic_dashboard(
    topic_prevalence: dict[int, float],
    doc_dominant: list[tuple[int, float]],
    output_path: Path | str | None = None,
) -> None:
    """
    Simple Plotly dashboard: topic prevalence bar + document dominant-topic distribution.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    topics = sorted(topic_prevalence.keys())
    prev_vals = [topic_prevalence[t] for t in topics]
    dominant_counts = {t: 0 for t in topics}
    for tid, _ in doc_dominant:
        dominant_counts[tid] = dominant_counts.get(tid, 0) + 1
    dom_vals = [dominant_counts[t] for t in topics]
    labels = [f"Topic {t}" for t in topics]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Global topic prevalence", "Documents per dominant topic"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(x=labels, y=prev_vals, name="Mean proportion"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=labels, y=dom_vals, name="Doc count"),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=600,
        title_text="Topic modelling dashboard",
        showlegend=True,
    )
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
    fig.show()
