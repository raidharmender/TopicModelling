"""
Topic Modelling pipeline: ingestion → LDA → analysis → visualization.

"""

from pathlib import Path

from src.ingestion import (
    preprocess_text,
    build_dictionary,
    corpus_bow,
    corpus_tfidf,
)
from src.modeling import train_lda, get_document_topics
from src.analysis import (
    per_document_topic_distribution,
    dominant_topic_per_document,
    global_topic_prevalence,
    topic_prevalence_by_group,
)
from src.visualization import (
    save_pyldavis,
    plot_topic_prevalence_bar,
    plot_topic_heatmap,
    plot_topic_word_network,
    plot_interactive_topic_dashboard,
)


# --- Sample corpus (replace with your documents or load from files) ---
SAMPLE_DOCS = [
    "Machine learning and deep learning are transforming artificial intelligence.",
    "Natural language processing uses algorithms to understand human language.",
    "Data science involves statistics programming and domain expertise.",
    "Python is widely used for data analysis and machine learning projects.",
    "Neural networks learn from data through backpropagation and gradient descent.",
    "Topic modeling discovers latent themes in text collections.",
    "Clustering algorithms group similar documents together.",
    "Information retrieval systems help users find relevant documents.",
    "Text mining extracts useful patterns from unstructured text.",
    "Statistical models underpin many machine learning techniques.",
] * 5  # Repeat for a slightly larger corpus


def main() -> None:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # 1. Document ingestion
    tokenized = [preprocess_text(d) for d in SAMPLE_DOCS]
    id2word = build_dictionary(tokenized, no_below=1, no_above=0.6)
    bow_corpus = corpus_bow(tokenized, id2word)
    tfidf_corpus, tfidf_model = corpus_tfidf(bow_corpus)

    # 2. Topic modeling (LDA) — use BoW or TF-IDF corpus
    num_topics = 4
    lda = train_lda(
        bow_corpus,
        id2word,
        num_topics=num_topics,
        passes=15,
        use_multicore=True,
        random_state=42,
    )

    # 3. Analysis
    doc_topics = get_document_topics(lda, bow_corpus, minimum_probability=0.0)
    doc_distributions = per_document_topic_distribution(doc_topics, num_topics)
    dominant = dominant_topic_per_document(doc_distributions)
    global_prevalence = global_topic_prevalence(doc_distributions, num_topics)

    # 4. Visualization
    save_pyldavis(lda, bow_corpus, id2word, out_dir / "lda_vis.html")
    plot_topic_prevalence_bar(
        global_prevalence,
        title="Global topic prevalence",
        output_path=out_dir / "topic_prevalence_bar.png",
    )
    plot_topic_word_network(
        lda,
        id2word,
        top_n_per_topic=5,
        output_path=out_dir / "topic_word_network.png",
    )
    plot_interactive_topic_dashboard(
        global_prevalence,
        dominant,
        output_path=out_dir / "dashboard.html",
    )

    # Optional: heatmap by "group" (here we fake groups by doc index % 2)
    groups = [f"group_{i % 2}" for i in range(len(doc_distributions))]
    by_group = topic_prevalence_by_group(doc_distributions, groups, num_topics)
    plot_topic_heatmap(
        by_group,
        title="Topic prevalence by group",
        output_path=out_dir / "topic_heatmap.png",
    )

    print("Done. Check the 'output' folder for visualizations.")


if __name__ == "__main__":
    main()
