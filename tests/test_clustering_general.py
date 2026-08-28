"""Tests for GeneralVendiClustering (arbitrary order q, via eigendecomposition)."""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from vendi_clustering import GeneralVendiClustering

Q_VALUES = [0.5, 1.0, 2.0, 5.0, float("inf")]


def make_case(seed=0, m=10, dim=6):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(m, dim))
    topic_ids = list(range(m))
    topic_sizes = {t: int(rng.integers(5, 100)) for t in topic_ids}
    return embeddings, topic_ids, topic_sizes


def test_rejects_non_positive_q():
    for q in (0.0, -1.0):
        with pytest.raises(ValueError, match="q must be > 0"):
            GeneralVendiClustering(q=q)


@pytest.mark.parametrize("q", Q_VALUES)
def test_orthogonal_topics_score_equals_topic_count(q):
    """With mutually dissimilar topics K is the identity and VS_q = m for every q."""
    m = 6
    score = GeneralVendiClustering(q=q).compute_vendi_q_score(np.eye(m))
    assert score == pytest.approx(m, rel=1e-9)


@pytest.mark.parametrize("q", Q_VALUES)
def test_identical_topics_score_equals_one(q):
    """When every topic is the same, the effective number of topics is 1."""
    score = GeneralVendiClustering(q=q).compute_vendi_q_score(np.ones((5, 5)))
    assert score == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("q", Q_VALUES)
def test_score_is_bounded_by_topic_count(q):
    embeddings, _, _ = make_case(seed=1)
    K = cosine_similarity(embeddings)
    score = GeneralVendiClustering(q=q).compute_vendi_q_score(K)
    assert 1.0 - 1e-9 <= score <= K.shape[0] + 1e-9


def test_score_is_non_increasing_in_q():
    """VS_q is a Hill number: monotonically non-increasing in q."""
    embeddings, _, _ = make_case(seed=2)
    K = cosine_similarity(embeddings)
    scores = [GeneralVendiClustering(q=q).compute_vendi_q_score(K) for q in sorted(Q_VALUES)]
    for lower, higher in zip(scores, scores[1:]):
        assert higher <= lower + 1e-9


@pytest.mark.parametrize("q", Q_VALUES)
def test_reduces_to_exactly_target_k(q):
    embeddings, topic_ids, topic_sizes = make_case(seed=3)
    mapping = GeneralVendiClustering(q=q).cluster(embeddings, topic_ids, topic_sizes, target_k=3)
    assert len(set(mapping.values())) == 3
    assert set(mapping.keys()) == set(topic_ids)


@pytest.mark.parametrize("q", Q_VALUES)
def test_group_representative_is_the_minimum_id(q):
    embeddings, topic_ids, topic_sizes = make_case(seed=4)
    mapping = GeneralVendiClustering(q=q).cluster(embeddings, topic_ids, topic_sizes, target_k=3)
    for rep in set(mapping.values()):
        group = [t for t, r in mapping.items() if r == rep]
        assert rep == min(group)


def test_caller_topic_sizes_is_not_mutated():
    embeddings, topic_ids, topic_sizes = make_case(seed=5)
    before = dict(topic_sizes)
    GeneralVendiClustering(q=1.0).cluster(embeddings, topic_ids, topic_sizes, target_k=3)
    assert topic_sizes == before


def test_merged_matrix_stays_symmetric_with_unit_diagonal():
    """Each merge must leave a well-formed similarity matrix behind."""
    embeddings, _, topic_sizes = make_case(seed=6, m=8)
    K = cosine_similarity(embeddings)
    general = GeneralVendiClustering(q=1.0)

    merged = general._build_merged_matrix(K, 2, 5, topic_sizes[2], topic_sizes[5])

    assert merged.shape == (7, 7)
    assert np.allclose(merged, merged.T)
    assert np.allclose(np.diag(merged), 1.0)


def test_large_q_approaches_the_infinite_limit():
    embeddings, _, _ = make_case(seed=7)
    K = cosine_similarity(embeddings)
    large = GeneralVendiClustering(q=200.0).compute_vendi_q_score(K)
    limit = GeneralVendiClustering(q=float("inf")).compute_vendi_q_score(K)
    assert large == pytest.approx(limit, rel=1e-2)
