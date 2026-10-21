"""Optional WordNet synonym expansion (no-op if NLTK unavailable)."""

from __future__ import annotations

import logging
import random
from itertools import product

from fairai.infrastructure.generation.normalize import normalize_sentence

logger = logging.getLogger(__name__)


def _wordnet_ready() -> bool:
    try:
        import nltk
        from nltk.corpus import wordnet  # noqa: F401

        try:
            wordnet.synsets("test")  # type: ignore[name-defined]
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info("WordNet unavailable (%s); skipping synonym expansion", exc)
        return False


def generate_synonym_sentences(
    sentence: str,
    *,
    max_samples: int = 5,
    max_syn_per_word: int = 1,
    seed: int = 42,
) -> list[str]:
    """Return original + limited WordNet paraphrases (best-effort)."""
    out = [normalize_sentence(sentence)]
    if max_samples <= 1 or not _wordnet_ready():
        return out

    from nltk import pos_tag, word_tokenize
    from nltk.corpus import wordnet

    rng = random.Random(seed)
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    alternatives: list[list[str]] = []
    for word, tag in tagged:
        pos = {
            "NN": wordnet.NOUN,
            "NNS": wordnet.NOUN,
            "JJ": wordnet.ADJ,
            "RB": wordnet.ADV,
            "NNP": wordnet.NOUN,
        }.get(tag)
        opts = [word]
        if pos:
            syns: set[str] = set()
            for synset in wordnet.synsets(word, pos=pos):
                for lemma in synset.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != word.lower():
                        syns.add(name)
            picked = rng.sample(sorted(syns), min(len(syns), max_syn_per_word)) if syns else []
            opts.extend(picked)
        alternatives.append(opts)

    for combo in product(*alternatives):
        if len(out) >= max_samples:
            break
        candidate = normalize_sentence(" ".join(combo))
        if candidate not in out:
            out.append(candidate)
    return out
