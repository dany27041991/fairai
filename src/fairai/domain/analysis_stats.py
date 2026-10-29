"""Pure descriptive / comparison stats used by Thesis ANALISYS cells."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


LABEL_NAMES = {0: "stereotype", 1: "anti-stereotype", 2: "neutral"}


@dataclass(slots=True)
class MeanStd:
    mean: float
    std: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class PairedCompareResult:
    metric: str
    baseline: MeanStd
    treatment: MeanStd
    t_stat: float | None
    p_value: float | None
    significant: bool | None
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline.to_dict(),
            "treatment": self.treatment.to_dict(),
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "significant": self.significant,
            "alpha": self.alpha,
        }


def mean_std(values: Sequence[float]) -> MeanStd:
    n = len(values)
    if n == 0:
        return MeanStd(mean=0.0, std=0.0, n=0)
    mean = sum(values) / n
    if n == 1:
        return MeanStd(mean=mean, std=0.0, n=1)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return MeanStd(mean=mean, std=var**0.5, n=n)


def paired_ttest(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, bool | None]:
    """Paired t-test (scipy if available). Returns (t, p, significant)."""
    if len(baseline) != len(treatment) or len(baseline) < 2:
        return None, None, None
    try:
        from scipy.stats import ttest_rel
    except ImportError:
        return None, None, None
    t_stat, p_val = ttest_rel(list(treatment), list(baseline))
    return float(t_stat), float(p_val), bool(p_val < alpha)


def compare_paired(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    metric: str,
    alpha: float = 0.05,
) -> PairedCompareResult:
    t_stat, p_val, sig = paired_ttest(baseline, treatment, alpha=alpha)
    return PairedCompareResult(
        metric=metric,
        baseline=mean_std(baseline),
        treatment=mean_std(treatment),
        t_stat=t_stat,
        p_value=p_val,
        significant=sig,
        alpha=alpha,
    )


def accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true:
        return 0.0
    ok = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b)
    return ok / len(y_true)


def accuracy_by_group(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Sequence[Any],
) -> dict[str, float]:
    buckets: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    for t, p, g in zip(y_true, y_pred, groups, strict=True):
        buckets[str(g)].append((t, p))
    return {
        g: accuracy([t for t, _ in pairs], [p for _, p in pairs])
        for g, pairs in buckets.items()
    }


def error_rate_by_group(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    groups: Sequence[Any],
) -> dict[str, float]:
    return {g: 1.0 - acc for g, acc in accuracy_by_group(y_true, y_pred, groups).items()}


def count_by(values: Iterable[Any]) -> dict[str, int]:
    return {str(k if k is not None else "No bias"): int(v) for k, v in Counter(values).items()}


def label_counts_per_bias(
    labels: Sequence[int],
    biases: Sequence[Any],
) -> dict[str, dict[str, int]]:
    nested: dict[str, Counter[int]] = defaultdict(Counter)
    for label, bias in zip(labels, biases, strict=True):
        key = str(bias if bias is not None else "No bias")
        nested[key][int(label)] += 1
    return {
        bias: {LABEL_NAMES.get(lab, str(lab)): int(c) for lab, c in counter.items()}
        for bias, counter in nested.items()
    }


def avg_length_by(
    sentences: Sequence[str],
    keys: Sequence[Any],
) -> dict[str, float]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for sent, key in zip(sentences, keys, strict=True):
        buckets[str(key if key is not None else "No bias")].append(len(sent.split()))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in buckets.items()}


def lexical_diversity(texts: Sequence[str], *, stop_words: set[str] | None = None) -> float:
    stop = stop_words or set()
    tokens: list[str] = []
    for text in texts:
        tokens.extend(w.lower() for w in text.split() if w.lower() not in stop)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def pos_distribution(texts: Sequence[str]) -> dict[str, float]:
    """Thesis cell 49 — NLTK POS tag relative frequencies. Empty if nltk missing."""
    try:
        import nltk
    except ImportError:
        return {}
    try:
        nltk.pos_tag(["test"])
    except LookupError:
        resources = (
            "punkt",
            "punkt_tab",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        )
        for res in resources:
            try:
                nltk.download(res, quiet=True)
            except Exception:  # noqa: BLE001
                pass
    tags: list[str] = []
    for text in texts:
        tokens = nltk.word_tokenize(text)
        tags.extend(tag for _, tag in nltk.pos_tag(tokens))
    total = len(tags)
    if total == 0:
        return {}
    counts = Counter(tags)
    return {tag: count / total for tag, count in counts.items()}


def top_pos_tags(dist: Mapping[str, float], *, k: int = 10) -> list[tuple[str, float]]:
    return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]


def composite_metric(
    true_labels: Sequence[Any],
    pred_labels: Sequence[Any],
    *,
    null_value: Any = -1,
) -> float:
    """Thesis threshold Optuna objective: accuracy × coverage (non-abstentions)."""
    if not true_labels:
        return 0.0
    kept = [(t, p) for t, p in zip(true_labels, pred_labels, strict=True) if p != null_value]
    coverage = len(kept) / len(true_labels)
    if not kept:
        return 0.0
    acc = sum(1 for t, p in kept if t == p) / len(kept)
    return acc * coverage


def _tokenize_words(text: str) -> list[str]:
    """Thesis uses ``nltk.word_tokenize`` when available; else whitespace split."""
    try:
        from nltk.tokenize import word_tokenize

        return [t.lower() for t in word_tokenize(text.lower())]
    except Exception:  # noqa: BLE001
        return text.lower().split()


def distinct_n(text: str, n: int) -> float:
    """Thesis `distinct_n` — unique n-grams / total n-grams."""
    tokens = _tokenize_words(text)
    if len(tokens) < n or n < 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Thesis RL-eval BLEU (``nltk.translate.bleu_score.sentence_bleu``)."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError:
        # Simple unigram overlap fallback (not Thesis-exact)
        ref = set(reference.lower().split())
        hyp = hypothesis.lower().split()
        if not hyp:
            return 0.0
        return sum(1 for w in hyp if w in ref) / len(hyp)
    return float(sentence_bleu([reference.split()], hypothesis.split()))


def summarize_metric_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str = "model",
    metrics: Sequence[str] = ("perplexity", "distinct_1", "distinct_2", "bleu"),
) -> list[dict[str, Any]]:
    """Thesis RL eval summary: mean±std per model for generation metrics."""
    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        g = str(row[group_key])
        for m in metrics:
            if m in row and row[m] is not None:
                by_group[g][m].append(float(row[m]))
    out: list[dict[str, Any]] = []
    for g, metric_map in sorted(by_group.items()):
        entry: dict[str, Any] = {group_key: g}
        for m, vals in metric_map.items():
            stats = mean_std(vals)
            entry[f"{m}_mean"] = stats.mean
            entry[f"{m}_std"] = stats.std
            entry[f"{m}_n"] = stats.n
        out.append(entry)
    return out
