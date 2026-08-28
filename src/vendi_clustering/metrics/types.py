"""Model-agnostic representation of a fitted topic model's output."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

OUTLIER_TOPIC = -1


@dataclass
class TopicModelOutput:
    """Everything the metrics need, with no dependency on the model that produced it.

    Attributes:
        method: Identifier for the run, e.g. "vendi", "agglomerative", "lda".
        topic_ids: Topic IDs, outlier excluded, aligned with `topic_words`.
        topic_words: Top words per topic, most important first.
        doc_topics: Topic assignment per document; -1 marks an outlier.
        topic_embeddings: Native topic vectors, e.g. BERTopic document centroids.
        metadata: Free-form run details (dataset, seed, target_k, timings).
    """

    method: str
    topic_ids: List[int]
    topic_words: List[List[str]]
    doc_topics: List[int]
    topic_embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.topic_ids) != len(self.topic_words):
            raise ValueError(
                f"{len(self.topic_ids)} topic_ids but {len(self.topic_words)} word lists"
            )
        if OUTLIER_TOPIC in self.topic_ids:
            raise ValueError("topic_ids must exclude the outlier topic")
        for name in ("topic_embeddings",):
            matrix = getattr(self, name)
            if matrix is not None:
                matrix = np.asarray(matrix)
                if matrix.shape[0] != len(self.topic_ids):
                    raise ValueError(
                        f"{name} has {matrix.shape[0]} rows but there are "
                        f"{len(self.topic_ids)} topics"
                    )
                setattr(self, name, matrix)

    @property
    def n_topics(self) -> int:
        return len(self.topic_ids)

    @property
    def n_outliers(self) -> int:
        return sum(1 for t in self.doc_topics if t == OUTLIER_TOPIC)

    def top_words(self, n: int) -> List[List[str]]:
        return [words[:n] for words in self.topic_words]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "topic_ids": [int(t) for t in self.topic_ids],
            "topic_words": [list(words) for words in self.topic_words],
            "doc_topics": [int(t) for t in self.doc_topics],
            "topic_embeddings": _matrix_to_list(self.topic_embeddings),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TopicModelOutput":
        return cls(
            method=payload["method"],
            topic_ids=payload["topic_ids"],
            topic_words=payload["topic_words"],
            doc_topics=payload["doc_topics"],
            topic_embeddings=_list_to_matrix(payload.get("topic_embeddings")),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))
        return path

    @classmethod
    def load(cls, path) -> "TopicModelOutput":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _matrix_to_list(matrix):
    return None if matrix is None else np.asarray(matrix).tolist()


def _list_to_matrix(values):
    return None if values is None else np.asarray(values, dtype=float)
