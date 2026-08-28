"""Tests for the model-agnostic evaluation layer."""

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from vendi_clustering.eval.metrics import (
    evaluate,
    tokenize,
    topic_word_coherence,
    vendi_diversity,
    word_uniqueness,
)
from vendi_clustering.eval.types import TopicModelOutput

ANALYZER = CountVectorizer().build_analyzer()


def make_output(**overrides):
    defaults = dict(
        method="test",
        topic_ids=[0, 1, 2],
        topic_words=[
            ["car", "engine", "wheel"],
            ["god", "faith", "church"],
            ["disk", "drive", "memory"],
        ],
        doc_topics=[0, 0, 1, 1, 2, -1],
    )
    defaults.update(overrides)
    return TopicModelOutput(**defaults)


def test_counts_exclude_and_count_outliers():
    output = make_output()
    assert output.n_topics == 3
    assert output.n_outliers == 1


def test_top_words_truncates():
    assert make_output().top_words(2) == [
        ["car", "engine"],
        ["god", "faith"],
        ["disk", "drive"],
    ]


def test_rejects_misaligned_topic_words():
    with pytest.raises(ValueError, match="word lists"):
        make_output(topic_ids=[0, 1])


def test_rejects_outlier_in_topic_ids():
    with pytest.raises(ValueError, match="exclude the outlier"):
        make_output(topic_ids=[-1, 1, 2])


def test_rejects_embeddings_with_wrong_row_count():
    with pytest.raises(ValueError, match="topic_embeddings has 2 rows"):
        make_output(topic_embeddings=np.zeros((2, 4)))


def test_json_round_trip_preserves_everything(tmp_path):
    output = make_output(
        topic_embeddings=np.arange(12, dtype=float).reshape(3, 4),
        metadata={"seed": 42},
    )
    reloaded = TopicModelOutput.load(output.save(tmp_path / "out.json"))

    assert reloaded.method == output.method
    assert reloaded.topic_ids == output.topic_ids
    assert reloaded.topic_words == output.topic_words
    assert reloaded.doc_topics == output.doc_topics
    assert reloaded.metadata == {"seed": 42}
    np.testing.assert_allclose(reloaded.topic_embeddings, output.topic_embeddings)


def test_json_round_trip_keeps_none_embeddings(tmp_path):
    reloaded = TopicModelOutput.load(make_output().save(tmp_path / "out.json"))
    assert reloaded.topic_embeddings is None


def test_word_uniqueness_all_distinct():
    assert word_uniqueness(make_output(), top_n=3) == 1.0


def test_word_uniqueness_all_shared():
    output = make_output(topic_words=[["a", "b"], ["a", "b"], ["a", "b"]])
    assert word_uniqueness(output, top_n=2) == pytest.approx(2 / 6)


@pytest.mark.parametrize("q", [0.5, 1.0, 2.0, "inf"])
def test_vendi_diversity_counts_orthogonal_topics(q):
    output = make_output(topic_embeddings=np.eye(3))
    assert vendi_diversity(output, q=q) == pytest.approx(3.0, rel=1e-6)


@pytest.mark.parametrize("q", [0.5, 1.0, 2.0])
def test_vendi_diversity_collapses_for_identical_topics(q):
    output = make_output(topic_embeddings=np.ones((3, 4)))
    assert vendi_diversity(output, q=q) == pytest.approx(1.0, rel=1e-5)


def test_vendi_diversity_needs_embeddings():
    assert vendi_diversity(make_output(), q=2.0) == 0.0


def test_topic_word_coherence_is_one_for_identical_word_vectors():
    output = make_output()
    coh = topic_word_coherence(output, embed_words=lambda words: np.ones((len(words), 4)))
    assert coh == pytest.approx(1.0)


def test_topic_word_coherence_is_zero_for_orthogonal_word_vectors():
    output = make_output()
    coh = topic_word_coherence(output, embed_words=lambda words: np.eye(len(words)))
    assert coh == pytest.approx(0.0)


def test_tokenize_requires_an_analyzer():
    with pytest.raises(ValueError, match="analyzer is required"):
        tokenize(["a doc"], analyzer=None)


def test_evaluate_returns_the_expected_keys():
    docs = ["car engine wheel", "god faith church", "disk drive memory"] * 4
    output = make_output(doc_topics=[0, 1, 2] * 4, topic_embeddings=np.eye(3))

    metrics = evaluate(output, docs, analyzer=ANALYZER)

    assert set(metrics) == {
        "n_topics",
        "n_outliers",
        "outlier_ratio",
        "coherence_cv",
        "coherence_npmi",
        "word_uniqueness_10",
    }


def test_evaluate_omits_the_opt_in_metrics():
    docs = ["car engine wheel", "god faith church", "disk drive memory"] * 4
    output = make_output(doc_topics=[0, 1, 2] * 4, topic_embeddings=np.eye(3))

    metrics = evaluate(output, docs, analyzer=ANALYZER)

    assert not any(k.startswith("vendi_diversity") for k in metrics)
    assert "word_uniqueness_25" not in metrics


def test_evaluate_adds_coh_only_when_an_embedder_is_given():
    docs = ["car engine wheel", "god faith church", "disk drive memory"] * 4
    output = make_output(doc_topics=[0, 1, 2] * 4, topic_embeddings=np.eye(3))

    metrics = evaluate(
        output, docs, analyzer=ANALYZER, embed_words=lambda w: np.ones((len(w), 4))
    )
    assert metrics["coh"] == pytest.approx(1.0)
