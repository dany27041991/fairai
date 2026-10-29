"""Thesis ANALISYS cell 49 — lexical / sentiment / TF-IDF / optional t-SNE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fairai.domain.analysis_stats import lexical_diversity, pos_distribution, top_pos_tags
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class TextSemanticsReport:
    n_samples: int
    lexical_diversity_by_bias_label: dict[str, float]
    pos_top_by_label: dict[str, list[dict[str, float]]]
    top_tfidf_by_label: dict[str, list[dict[str, float]]]
    length_ttest: dict[str, float | None]
    sentiment_means: dict[str, float]
    semantic_similarity_by_bias: dict[str, float | None]
    plot_paths: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "lexical_diversity_by_bias_label": self.lexical_diversity_by_bias_label,
            "pos_top_by_label": self.pos_top_by_label,
            "top_tfidf_by_label": self.top_tfidf_by_label,
            "length_ttest": self.length_ttest,
            "sentiment_means": self.sentiment_means,
            "semantic_similarity_by_bias": self.semantic_similarity_by_bias,
            "plot_paths": self.plot_paths,
            "notes": self.notes,
        }


def analyze_text_semantics(
    dataset_path: Path,
    *,
    out_dir: Path | None = None,
    max_samples: int | None = None,
    write_plots: bool = True,
    skip_embeddings: bool = False,
    skip_tsne: bool = False,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_embed_per_bias: int = 500,
    device: str = "cpu",
) -> TextSemanticsReport:
    examples = load_examples(dataset_path, max_samples=max_samples)
    # Thesis drops neutrals for this cell
    examples = [ex for ex in examples if int(ex.label) != 2]
    notes: list[str] = []
    stop_words = _stop_words(notes)

    lex: dict[str, float] = {}
    for bias in sorted({ex.bias_type for ex in examples if ex.bias_type}):
        for label in (0, 1):
            texts = [
                ex.sentence
                for ex in examples
                if ex.bias_type == bias and int(ex.label) == label
            ]
            key = f"{bias}|{label}"
            lex[key] = lexical_diversity(texts, stop_words=stop_words)

    lengths0 = [len(ex.sentence.split()) for ex in examples if int(ex.label) == 0]
    lengths1 = [len(ex.sentence.split()) for ex in examples if int(ex.label) == 1]
    length_ttest = _welch_ttest(lengths0, lengths1)

    sentiment_means, sent_scores = _sentiment_scores(examples, notes)
    top_tfidf = _top_tfidf(examples, notes)
    pos_top = _pos_top_by_label(examples, notes)
    sim_by_bias: dict[str, float | None] = {}
    plot_paths: list[str] = []

    if write_plots and out_dir is not None and examples:
        plot_paths.extend(
            _boxplot_length_sentiment(out_dir, examples, sent_scores, notes)
        )
        plot_paths.extend(_plot_pos_bars(out_dir, pos_top, notes))

    if not skip_embeddings:
        sim_by_bias, emb_plots = _embedding_analysis(
            examples,
            out_dir=out_dir if write_plots else None,
            model_name=embedding_model,
            max_per_bias=max_embed_per_bias,
            skip_tsne=skip_tsne,
            device=device,
            notes=notes,
        )
        plot_paths.extend(emb_plots)
    else:
        notes.append("embeddings skipped")

    report = TextSemanticsReport(
        n_samples=len(examples),
        lexical_diversity_by_bias_label=lex,
        pos_top_by_label=pos_top,
        top_tfidf_by_label=top_tfidf,
        length_ttest=length_ttest,
        sentiment_means=sentiment_means,
        semantic_similarity_by_bias=sim_by_bias,
        plot_paths=plot_paths,
        notes=notes,
    )
    if out_dir is not None:
        write_json(out_dir / "text_semantics_report.json", report.to_dict())
    return report


def _pos_top_by_label(examples: list, notes: list[str]) -> dict[str, list[dict[str, float]]]:
    out: dict[str, list[dict[str, float]]] = {}
    for label, name in ((0, "stereotype"), (1, "anti-stereotype")):
        texts = [ex.sentence for ex in examples if int(ex.label) == label]
        dist = pos_distribution(texts)
        if not dist and texts:
            notes.append("nltk POS unavailable — pos_top_by_label empty")
        out[name] = [{"tag": t, "freq": float(f)} for t, f in top_pos_tags(dist, k=10)]
    return out


def _plot_pos_bars(
    out_dir: Path,
    pos_top: dict[str, list[dict[str, float]]],
    notes: list[str],
) -> list[str]:
    if not any(pos_top.values()):
        return []
    try:
        from fairai.infrastructure.viz.bars import plot_horizontal_counts
    except ImportError:
        notes.append("viz missing — POS plots skipped")
        return []
    paths: list[str] = []
    for name, rows in pos_top.items():
        if not rows:
            continue
        path = plot_horizontal_counts(
            [r["tag"] for r in rows],
            [r["freq"] for r in rows],
            xlabel="Relative frequency",
            ylabel="POS tag",
            title=f"Top POS tags — {name}",
            out_path=out_dir / f"text_pos_{name.replace('-', '_')}.png",
        )
        paths.append(str(path))
    return paths


def _stop_words(notes: list[str]) -> set[str]:
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))
    except ImportError:
        notes.append("nltk missing — empty stopword set")
        return set()


def _welch_ttest(a: list[float], b: list[float]) -> dict[str, float | None]:
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": None, "p_value": None}
    try:
        from scipy.stats import ttest_ind

        t_stat, p_val = ttest_ind(a, b, equal_var=False)
        return {"t_stat": float(t_stat), "p_value": float(p_val)}
    except ImportError:
        return {"t_stat": None, "p_value": None}


def _sentiment_scores(
    examples: list,
    notes: list[str],
) -> tuple[dict[str, float], dict[int, float]]:
    scores: dict[int, float] = {}
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
    except ImportError:
        notes.append("nltk/vader missing — sentiment skipped")
        return {}, scores

    for i, ex in enumerate(examples):
        scores[i] = float(sia.polarity_scores(ex.sentence)["compound"])
    means: dict[str, float] = {}
    for label, name in ((0, "stereotype"), (1, "anti-stereotype")):
        vals = [scores[i] for i, ex in enumerate(examples) if int(ex.label) == label]
        means[name] = sum(vals) / len(vals) if vals else 0.0
    return means, scores


def _top_tfidf(examples: list, notes: list[str], k: int = 10) -> dict[str, list[dict[str, float]]]:
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        notes.append("sklearn/numpy missing — TF-IDF skipped")
        return {}

    sentences = [ex.sentence for ex in examples]
    labels = [int(ex.label) for ex in examples]
    if not sentences:
        return {}
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    matrix = vectorizer.fit_transform(sentences)
    names = np.array(vectorizer.get_feature_names_out())
    out: dict[str, list[dict[str, float]]] = {}
    for label, name in ((0, "stereotype"), (1, "anti-stereotype")):
        mask = [i for i, lab in enumerate(labels) if lab == label]
        if not mask:
            out[name] = []
            continue
        mean = matrix[mask].mean(axis=0).A1
        order = [i for i in mean.argsort()[::-1] if names[i].lower() != "sep"]
        out[name] = [
            {"term": str(names[i]), "avg_tfidf": float(mean[i])} for i in order[:k]
        ]
    return out


def _boxplot_length_sentiment(
    out_dir: Path,
    examples: list,
    sent_scores: dict[int, float],
    notes: list[str],
) -> list[str]:
    try:
        import pandas as pd
        import seaborn as sns

        from fairai.infrastructure.viz.savefig import require_matplotlib, save_figure
    except ImportError:
        notes.append("pandas/seaborn missing — boxplots skipped")
        return []

    plt = require_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(examples):
        if not ex.bias_type:
            continue
        rows.append(
            {
                "bias_type": ex.bias_type,
                "label_name": "stereotype" if int(ex.label) == 0 else "anti-stereotype",
                "sent_len": len(ex.sentence.split()),
                "sentiment": sent_scores.get(i),
            }
        )
    if not rows:
        return []
    df = pd.DataFrame(rows)
    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=df, y="bias_type", x="sent_len", hue="label_name", ax=ax)
    ax.set_title("Sentence Length Distribution by Bias Type and Label")
    paths.append(str(save_figure(fig, out_dir / "text_len_boxplot.png")))
    if df["sentiment"].notna().any():
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=df, y="bias_type", x="sentiment", hue="label_name", ax=ax2)
        ax2.set_title("Sentiment Score Distribution by Bias Type and Label")
        paths.append(str(save_figure(fig2, out_dir / "text_sentiment_boxplot.png")))
    return paths


def _embedding_analysis(
    examples: list,
    *,
    out_dir: Path | None,
    model_name: str,
    max_per_bias: int,
    skip_tsne: bool,
    device: str,
    notes: list[str],
) -> tuple[dict[str, float | None], list[str]]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from torch.nn.functional import cosine_similarity
    except ImportError:
        notes.append("sentence-transformers/torch missing — embeddings skipped")
        return {}, []

    model = SentenceTransformer(model_name, device=device)
    sim_by_bias: dict[str, float | None] = {}
    for bias in sorted({ex.bias_type for ex in examples if ex.bias_type}):
        stereo = [
            ex.sentence
            for ex in examples
            if ex.bias_type == bias and int(ex.label) == 0
        ][:max_per_bias]
        anti = [
            ex.sentence
            for ex in examples
            if ex.bias_type == bias and int(ex.label) == 1
        ][:max_per_bias]
        if not stereo or not anti:
            sim_by_bias[bias] = None
            continue
        emb_s = model.encode(stereo, convert_to_tensor=True, device=device)
        emb_a = model.encode(anti, convert_to_tensor=True, device=device)
        # Blocked cosine like Thesis (batched lightly)
        sims = []
        bs = 256
        for i in range(0, emb_s.size(0), bs):
            block = cosine_similarity(
                emb_s[i : i + bs].unsqueeze(1),
                emb_a.unsqueeze(0),
                dim=2,
            )
            sims.append(block)
        mat = torch.cat(sims, dim=0)
        sim_by_bias[bias] = float(
            (mat.max(dim=1).values.mean() + mat.max(dim=0).values.mean()).item() / 2
        )

    paths: list[str] = []
    if out_dir is not None and not skip_tsne and examples:
        try:
            import pandas as pd
            import seaborn as sns
            from sklearn.manifold import TSNE

            from fairai.infrastructure.viz.savefig import require_matplotlib, save_figure

            plt = require_matplotlib()
            emb = model.encode([ex.sentence for ex in examples], show_progress_bar=False)
            xy = TSNE(n_components=2, random_state=42, init="pca").fit_transform(emb)
            df = pd.DataFrame(
                {
                    "tsne_1": xy[:, 0],
                    "tsne_2": xy[:, 1],
                    "label_name": [
                        "stereotype" if int(ex.label) == 0 else "anti-stereotype"
                        for ex in examples
                    ],
                    "bias_type": [ex.bias_type or "unknown" for ex in examples],
                }
            )
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(
                data=df, x="tsne_1", y="tsne_2", hue="label_name", style="bias_type", ax=ax
            )
            ax.set_title("t-SNE Visualization of Sentence Embeddings")
            paths.append(str(save_figure(fig, out_dir / "text_tsne.png")))
        except Exception as exc:  # noqa: BLE001 — optional viz path
            notes.append(f"t-SNE skipped: {exc}")
    return sim_by_bias, paths
