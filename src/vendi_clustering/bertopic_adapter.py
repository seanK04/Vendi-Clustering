"""Apply Vendi Clustering to a fitted BERTopic model.

Reads `topic_embeddings_`, `c_tf_idf_` and `topic_sizes_`, and applies the result
through `merge_topics`.

Requires the optional dependency: `pip install vendi-clustering[bertopic]`.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clustering import VendiClustering
from .clustering_general import GeneralVendiClustering

OUTLIER_TOPIC = -1


def select_topic_representation(
    topic_model: Any, use_ctfidf: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """Return the topic representation to cluster on, plus its topic IDs.

    Arguments:
        topic_model: A fitted BERTopic model.
        use_ctfidf: Use the c-TF-IDF matrix instead of the semantic topic
            embeddings. Falls back to semantic embeddings if c-TF-IDF is absent.

    Returns:
        (representation matrix, topic ID per row) with the outlier topic removed.
    """
    # BERTopic orders both matrices by ascending topic ID, outlier first when present.
    topic_ids = sorted(topic_model.topic_sizes_.keys())

    representation = None
    if use_ctfidf:
        representation = getattr(topic_model, "c_tf_idf_", None)
        if representation is None:
            representation = topic_model.topic_embeddings_
    else:
        representation = topic_model.topic_embeddings_
        if representation is None:
            representation = getattr(topic_model, "c_tf_idf_", None)

    if representation is None:
        raise ValueError(
            "The model exposes neither topic_embeddings_ nor c_tf_idf_; it may not be fitted."
        )

    if hasattr(representation, "toarray"):
        representation = representation.toarray()
    representation = np.asarray(representation)

    if representation.shape[0] != len(topic_ids):
        raise ValueError(
            f"Topic representation has {representation.shape[0]} rows but the model reports "
            f"{len(topic_ids)} topics; cannot align rows to topic IDs."
        )

    keep = [i for i, topic in enumerate(topic_ids) if topic != OUTLIER_TOPIC]
    return representation[keep], [topic_ids[i] for i in keep]


def mapping_to_merge_groups(mapping: Dict[int, int]) -> List[List[int]]:
    """Turn a topic->representative mapping into `merge_topics` groups.

    Groups are sorted ascending: `merge_topics` keeps `topic_group[0]`, which must
    be the lowest topic ID to match the representative the clustering chose.
    Singleton groups are dropped.
    """
    groups: Dict[int, List[int]] = {}
    for topic, representative in mapping.items():
        groups.setdefault(int(representative), []).append(int(topic))
    return [sorted(group) for group in groups.values() if len(group) > 1]


def cluster_topics(
    topic_model: Any,
    docs: Sequence[str],
    nr_topics: int,
    use_ctfidf: bool = False,
    q: float = 2.0,
    images: Optional[Sequence[str]] = None,
    progress: bool = False,
) -> Any:
    """Merge a fitted model's topics down to `nr_topics` by Vendi Clustering.

    The outlier topic (-1) is held out of the clustering and left intact, matching
    how BERTopic treats it elsewhere.

    Arguments:
        topic_model: A fitted BERTopic model. Modified in place.
        docs: The documents the model was fitted on.
        nr_topics: Target number of topics, excluding the outlier topic.
        use_ctfidf: Cluster on c-TF-IDF vectors instead of semantic embeddings.
        q: Order of the Vendi Score. q=2 uses the accelerated implementation.
        images: Image paths, if the model was fitted on images.
        progress: Show a progress bar if tqdm is installed.

    Returns:
        The same `topic_model`, with topics merged.

    Note:
        BERTopic applies representation models during `fit`. Merging afterwards
        re-runs them on the merged topics, so a model configured with a
        representation model (KeyBERTInspired, MMR, ...) fine-tunes twice: once at
        fit time and once here.
    """
    embeddings, topic_ids = select_topic_representation(topic_model, use_ctfidf=use_ctfidf)

    if nr_topics >= len(topic_ids):
        return topic_model

    topic_sizes = {
        topic: size for topic, size in topic_model.topic_sizes_.items() if topic != OUTLIER_TOPIC
    }

    if q == 2.0:
        clusterer = VendiClustering()
    else:
        clusterer = GeneralVendiClustering(q=q)

    mapping = clusterer.cluster(
        embeddings=embeddings,
        topic_ids=topic_ids,
        topic_sizes=topic_sizes,
        target_k=nr_topics,
        progress=progress,
    )

    groups = mapping_to_merge_groups(mapping)
    if groups:
        topic_model.merge_topics(list(docs), groups, images=list(images) if images else None)

    return topic_model
