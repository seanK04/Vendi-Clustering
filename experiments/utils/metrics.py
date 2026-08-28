"""Protocol-facing evaluation, backed by vendi_clustering.metrics."""

from typing import Any, Dict, List, Optional, Sequence

from vendi_clustering.metrics.adapters import bertopic_analyzer, from_bertopic
from vendi_clustering.metrics.scores import evaluate
from vendi_clustering.metrics.types import TopicModelOutput


def to_output(
    topic_model: Any, method: str = "bertopic", metadata: Optional[Dict[str, Any]] = None
) -> TopicModelOutput:
    return from_bertopic(topic_model, method=method, metadata=metadata)


def evaluate_model(topic_model: Any, docs: Sequence[str]) -> Dict[str, float]:
    """Compute all metrics for a fitted BERTopic model."""
    return evaluate(
        from_bertopic(topic_model),
        docs,
        analyzer=bertopic_analyzer(topic_model),
    )


def print_metrics(metrics: Dict[str, float], title: str = "Model Evaluation Results") -> None:
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")

    print("\n[Structural Statistics]")
    print(f"  Number of Topics (k): {metrics.get('n_topics', 0)}")
    print(f"  Outlier Ratio:        {metrics.get('outlier_ratio', 0.0):.2%}")

    print("\n[Semantic Coherence]")
    print(f"  NPMI Coherence:       {metrics.get('coherence_npmi', 0.0):.4f}")
    print(f"  C_v Coherence:        {metrics.get('coherence_cv', 0.0):.4f}")

    print("\n[Topic Diversity]")
    print(f"  Word Uniqueness @10:  {metrics.get('word_uniqueness_10', 0.0):.4f}")

    if "coh" in metrics:
        print(f"\n[Topic-Word Coherence]\n  COH:                  {metrics['coh']:.4f}")


METRIC_COLUMNS: List[str] = [
    "n_topics",
    "n_outliers",
    "outlier_ratio",
    "coherence_cv",
    "coherence_npmi",
    "word_uniqueness_10",
]
