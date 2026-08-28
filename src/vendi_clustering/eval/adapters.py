"""Build a TopicModelOutput from a fitted model."""

from typing import Any, Dict, List, Optional

import numpy as np

from .types import OUTLIER_TOPIC, TopicModelOutput

TOP_WORDS = 25


def bertopic_analyzer(topic_model: Any):
    """The tokenizer BERTopic used, for building the coherence reference corpus."""
    vectorizer = getattr(topic_model, "vectorizer_model", None)
    if vectorizer is None:
        raise ValueError("model has no vectorizer_model; cannot recover its tokenization")
    return vectorizer.build_analyzer()


def from_bertopic(
    topic_model: Any,
    method: str = "bertopic",
    top_n: int = TOP_WORDS,
    word_embeddings: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TopicModelOutput:
    """Extract a TopicModelOutput from a fitted BERTopic model."""
    topic_ids = [t for t in sorted(topic_model.get_topics()) if t != OUTLIER_TOPIC]

    topic_words: List[List[str]] = [
        [word for word, _ in topic_model.get_topic(topic)[:top_n]] for topic in topic_ids
    ]

    embeddings = topic_model.topic_embeddings_
    if embeddings is not None:
        all_ids = sorted(topic_model.topic_sizes_.keys())
        keep = [i for i, topic in enumerate(all_ids) if topic != OUTLIER_TOPIC]
        embeddings = np.asarray(embeddings)[keep]

    return TopicModelOutput(
        method=method,
        topic_ids=topic_ids,
        topic_words=topic_words,
        doc_topics=list(topic_model.topics_),
        topic_embeddings=embeddings,
        word_embeddings=word_embeddings,
        metadata=metadata or {},
    )
