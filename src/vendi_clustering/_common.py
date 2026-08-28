"""Shared input handling for the Vendi Clustering classes."""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def prepare_inputs(
    embeddings: np.ndarray,
    topic_ids: Sequence[int],
    topic_sizes: Mapping[int, float],
) -> Tuple[np.ndarray, list, Dict[int, float]]:
    """Validate and canonicalise clustering inputs.

    Rows of `embeddings` correspond positionally to `topic_ids`. Topics are
    returned sorted by ascending topic ID.
i 
    Returns:
        (embeddings sorted by topic ID, sorted topic IDs, a private size copy)
    """
    embeddings = np.asarray(embeddings)
    topic_ids = list(topic_ids)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    if len(topic_ids) != embeddings.shape[0]:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but {len(topic_ids)} topic_ids were given"
        )
    if len(set(topic_ids)) != len(topic_ids):
        raise ValueError("topic_ids contains duplicates")

    missing = [t for t in topic_ids if t not in topic_sizes]
    if missing:
        raise ValueError(f"topic_sizes is missing entries for topics {missing}")

    # Sort by topic ID so row order is a deterministic function of the IDs.
    order = sorted(range(len(topic_ids)), key=lambda i: topic_ids[i])
    active_topics = [topic_ids[i] for i in order]
    sorted_embeddings = embeddings[order]

    # The merge loop writes accumulated sizes into this dict.
    sizes = {t: float(topic_sizes[t]) for t in active_topics}

    return sorted_embeddings, active_topics, sizes


def resolve_target_k(target_k: Optional[int], n_topics: int) -> Optional[int]:
    """Validate `target_k` against the number of input topics."""
    if target_k is None:
        return None
    if target_k < 1:
        raise ValueError(f"target_k must be >= 1, got {target_k}")
    if target_k > n_topics:
        raise ValueError(f"target_k ({target_k}) exceeds the number of topics ({n_topics})")
    return target_k


def progress_bar(total: int, desc: str, enabled: bool):
    """A tqdm bar when tqdm is installed and progress is on, else a no-op."""
    if enabled and total > 0:
        try:
            from tqdm import tqdm

            return tqdm(
                total=total,
                desc=desc,
                ncols=80,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]",
            )
        except ImportError:
            pass
    return _NullBar()


class _NullBar:
    def update(self, n: int = 1) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False
