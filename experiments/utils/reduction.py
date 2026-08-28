"""Dispatch a topic reduction onto the method named by a ModelConfig."""

from typing import Any, Sequence

from vendi_clustering.bertopic_adapter import cluster_topics


def apply_reduction(
    topic_model: Any,
    docs: Sequence[str],
    nr_topics: int,
    method: str,
    use_ctfidf: bool = False,
    q: float = 2.0,
) -> Any:
    """Reduce `topic_model` to `nr_topics` using `method`.

    "vendi" goes through the package adapter; "agglomerative" through BERTopic's
    own `reduce_topics`.
    """
    if method == "vendi":
        return cluster_topics(
            topic_model, docs, nr_topics=nr_topics, use_ctfidf=use_ctfidf, q=q
        )
    if method == "agglomerative":
        topic_model.reduce_topics(docs, nr_topics=nr_topics, use_ctfidf=use_ctfidf)
        return topic_model
    raise ValueError(f"unknown reduction method: {method!r}")
