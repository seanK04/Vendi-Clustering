"""Evaluation metrics over TopicModelOutput.

Every metric takes a TopicModelOutput, so BERTopic and native topic models are
scored by the same code.
"""

from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .types import TopicModelOutput

COHERENCE_TOP_N = 10
VENDI_Q_VALUES = {"0.5": 0.5, "1": 1.0, "2": 2.0, "10": 10.0, "inf": "inf"}

WordEmbedder = Callable[[List[str]], np.ndarray]


def tokenize(docs: Sequence[str], analyzer: Callable[[str], List[str]]) -> List[List[str]]:
    """Tokenize the reference corpus.

    `analyzer` is required: coherence is only comparable across methods when every
    method is scored against an identically tokenized corpus.
    """
    if analyzer is None:
        raise ValueError("analyzer is required; coherence depends on the tokenization")
    return [analyzer(doc) for doc in docs]


def _coherence(
    output: TopicModelOutput,
    tokenized_docs: List[List[str]],
    coherence: str,
    top_n: int = COHERENCE_TOP_N,
) -> float:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    dictionary = Dictionary(tokenized_docs)

    topics = []
    for words in output.top_words(top_n):
        in_vocab = [w for w in words if w in dictionary.token2id]
        if len(in_vocab) >= 2:
            topics.append(in_vocab)

    if not topics:
        return float("nan")

    model = CoherenceModel(
        topics=topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=coherence,
    )
    return model.get_coherence()


def coherence_cv(output, tokenized_docs, top_n=COHERENCE_TOP_N) -> float:
    return _coherence(output, tokenized_docs, "c_v", top_n)


def coherence_npmi(output, tokenized_docs, top_n=COHERENCE_TOP_N) -> float:
    return _coherence(output, tokenized_docs, "c_npmi", top_n)


def word_uniqueness(output: TopicModelOutput, top_n: int = 10) -> float:
    """Fraction of distinct words across all topics' top-n words."""
    all_words = [w for words in output.top_words(top_n) for w in words]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def _topic_vectors(output: TopicModelOutput, use_word_embeddings: bool) -> Optional[np.ndarray]:
    return output.word_embeddings if use_word_embeddings else output.topic_embeddings


def _mean_upper_triangle(matrix: np.ndarray) -> Optional[float]:
    similarity = cosine_similarity(matrix)
    upper = similarity[np.triu_indices(similarity.shape[0], k=1)]
    return float(upper.mean()) if upper.size else None


def mean_intertopic_cosine(output: TopicModelOutput, use_word_embeddings: bool = False) -> float:
    """One minus the mean pairwise cosine similarity between topic vectors."""
    vectors = _topic_vectors(output, use_word_embeddings)
    if vectors is None or len(vectors) < 2:
        return 0.0

    mean = _mean_upper_triangle(vectors)
    return 0.0 if mean is None else 1.0 - mean


def vendi_diversity(
    output: TopicModelOutput,
    q: Union[float, str] = 1.0,
    use_word_embeddings: bool = False,
) -> float:
    """Vendi Score of the topic set: the effective number of distinct topics."""
    from vendi_score import vendi

    vectors = _topic_vectors(output, use_word_embeddings)
    if vectors is None or len(vectors) < 2:
        return 0.0

    return vendi.score_K(cosine_similarity(vectors), q=q)


def topic_word_coherence(
    output: TopicModelOutput, embed_words: WordEmbedder, top_n: int = COHERENCE_TOP_N
) -> float:
    """COH: mean pairwise cosine of a topic's top-word embeddings, averaged over topics.

    Arguments:
        embed_words: Maps a word list to a (len(words), dim) embedding matrix.
    """
    scores = []
    for words in output.top_words(top_n):
        if len(words) < 2:
            continue
        mean = _mean_upper_triangle(np.asarray(embed_words(words)))
        if mean is not None:
            scores.append(mean)

    return float(np.mean(scores)) if scores else float("nan")


def evaluate(
    output: TopicModelOutput,
    docs: Sequence[str],
    analyzer: Callable[[str], List[str]],
    use_word_embeddings: bool = False,
    embed_words: Optional[WordEmbedder] = None,
) -> Dict[str, float]:
    """Compute every metric for one topic model output.

    `coh` is included only when `embed_words` is given.
    """
    tokenized_docs = tokenize(docs, analyzer)

    n_outliers = output.n_outliers
    metrics: Dict[str, float] = {
        "n_topics": output.n_topics,
        "n_outliers": n_outliers,
        "outlier_ratio": n_outliers / len(output.doc_topics) if output.doc_topics else 0.0,
        "coherence_cv": coherence_cv(output, tokenized_docs),
        "coherence_npmi": coherence_npmi(output, tokenized_docs),
        "word_uniqueness_10": word_uniqueness(output, top_n=10),
        "word_uniqueness_25": word_uniqueness(output, top_n=25),
        "mean_intertopic_cosine": mean_intertopic_cosine(output, use_word_embeddings),
    }

    for label, q in VENDI_Q_VALUES.items():
        metrics[f"vendi_diversity_{label}"] = vendi_diversity(output, q, use_word_embeddings)

    if embed_words is not None:
        metrics["coh"] = topic_word_coherence(output, embed_words)

    return metrics
