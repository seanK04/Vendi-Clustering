"""Tests for VendiClustering (q=2, algebraically accelerated)."""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

from vendi_clustering import GeneralVendiClustering, VendiClustering


def make_case(seed=0, m=12, dim=8, size_lo=5, size_hi=100):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(m, dim))
    topic_ids = list(range(m))
    topic_sizes = {t: int(rng.integers(size_lo, size_hi)) for t in topic_ids}
    return embeddings, topic_ids, topic_sizes


def partition(mapping):
    """Mapping -> set of groups, so comparisons ignore label choice."""
    return {
        frozenset(t for t, r in mapping.items() if r == rep) for rep in set(mapping.values())
    }


def test_closed_form_matches_actual_merged_matrix():
    """The vectorised T_new must equal ||K_merged||_F^2 for every candidate pair.

    This is the oracle for the algebraic shortcut: the fast path scores candidate
    merges without materialising them, so the prediction has to match the matrix
    that would actually result.
    """
    embeddings, _, sizes_by_id = make_case(seed=1, m=7, dim=5)
    K = cosine_similarity(embeddings)
    n_vec = np.array([float(sizes_by_id[t]) for t in range(7)])

    fast = VendiClustering()
    general = GeneralVendiClustering(q=2.0)
    T, R, G = fast.initialize_cache(K)

    n_sq = n_vec**2
    n_outer = np.outer(n_vec, n_vec)
    c2 = np.maximum(n_sq[:, None] + n_sq[None, :] + 2 * n_outer * K, 1e-10)
    sum_u_sq = (
        n_sq[:, None] * (R[:, None] - K**2)
        + n_sq[None, :] * (R[None, :] - K**2)
        + 2 * n_outer * (G - 2 * K)
    ) / c2
    T_pred = T - 2 * R[:, None] - 2 * R[None, :] + 2 * (K**2) + 2 * sum_u_sq - 1

    for a in range(7):
        for b in range(a + 1, 7):
            K_merged = general._build_merged_matrix(K, a, b, n_vec[a], n_vec[b])
            assert T_pred[a, b] == pytest.approx(np.sum(K_merged**2), abs=1e-10)


def test_vendi2_matches_eigenvalue_definition():
    """VS_2 = m^2 / ||K||_F^2 must agree with the eigenvalue formulation."""
    embeddings, _, _ = make_case(seed=2, m=9, dim=6)
    K = cosine_similarity(embeddings)
    T, _, _ = VendiClustering().initialize_cache(K)

    shortcut = VendiClustering().compute_vendi2_score(T, K.shape[0])
    eigen = GeneralVendiClustering(q=2.0).compute_vendi_q_score(K)
    assert shortcut == pytest.approx(eigen, rel=1e-10)


@pytest.mark.parametrize("target_k", [1, 2, 5, 11])
def test_reduces_to_exactly_target_k(target_k):
    embeddings, topic_ids, topic_sizes = make_case(seed=3)
    mapping = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=target_k)
    assert len(set(mapping.values())) == target_k


def test_mapping_covers_every_input_topic():
    embeddings, topic_ids, topic_sizes = make_case(seed=4)
    mapping = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=4)
    assert set(mapping.keys()) == set(topic_ids)
    assert set(mapping.values()) <= set(topic_ids)


def test_group_representative_is_the_minimum_id():
    """The BERTopic adapter relies on this to line up with merge_topics."""
    embeddings, topic_ids, topic_sizes = make_case(seed=5)
    mapping = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=4)
    for group in partition(mapping):
        assert {mapping[t] for t in group} == {min(group)}


def test_caller_topic_sizes_is_not_mutated():
    embeddings, topic_ids, topic_sizes = make_case(seed=6)
    before = dict(topic_sizes)
    VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=3)
    assert topic_sizes == before


def test_handles_unsorted_non_contiguous_topic_ids():
    """BERTopic topic IDs arrive without the outlier, so gaps are normal."""
    rng = np.random.default_rng(7)
    topic_ids = [7, 2, 9, 4, 1]
    embeddings = rng.normal(size=(5, 6))
    topic_sizes = {t: 10 + t for t in topic_ids}

    mapping = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=2)

    assert set(mapping.keys()) == set(topic_ids)
    for group in partition(mapping):
        assert {mapping[t] for t in group} == {min(group)}


def test_row_order_does_not_change_the_result():
    """Rows are matched to IDs, so shuffling the input rows must be a no-op."""
    embeddings, topic_ids, topic_sizes = make_case(seed=8, m=10)
    baseline = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=3)

    order = np.random.default_rng(0).permutation(len(topic_ids))
    shuffled = VendiClustering().cluster(
        embeddings[order], [topic_ids[i] for i in order], topic_sizes, target_k=3
    )
    assert shuffled == baseline


def test_rejects_mismatched_row_count():
    embeddings, topic_ids, topic_sizes = make_case(seed=9, m=6)
    with pytest.raises(ValueError, match="topic_ids were given"):
        VendiClustering().cluster(embeddings, topic_ids[:-1], topic_sizes, target_k=2)


def test_rejects_duplicate_topic_ids():
    embeddings, _, _ = make_case(seed=10, m=4)
    with pytest.raises(ValueError, match="duplicates"):
        VendiClustering().cluster(embeddings, [0, 1, 1, 2], {0: 5, 1: 5, 2: 5}, target_k=2)


def test_rejects_missing_topic_sizes():
    embeddings, topic_ids, topic_sizes = make_case(seed=11, m=5)
    del topic_sizes[topic_ids[0]]
    with pytest.raises(ValueError, match="missing entries"):
        VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=2)


@pytest.mark.parametrize("target_k", [0, -1, 13])
def test_rejects_out_of_range_target_k(target_k):
    embeddings, topic_ids, topic_sizes = make_case(seed=12, m=12)
    with pytest.raises(ValueError):
        VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=target_k)


# Pinned on a fixed seed. Guards against silent changes to merge selection:
# any edit that moves these groupings will move published numbers too.
GOLDEN_Q2 = {
    0: 0, 1: 1, 2: 2, 3: 1, 4: 1, 5: 1, 6: 0, 7: 7,
    8: 1, 9: 9, 10: 9, 11: 1, 12: 9, 13: 1, 14: 9,
}


def golden_case():
    rng = np.random.default_rng(20260827)
    embeddings = rng.normal(size=(15, 10))
    topic_ids = list(range(15))
    topic_sizes = {t: 10 * (t + 1) for t in topic_ids}
    return embeddings, topic_ids, topic_sizes


def test_output_is_stable():
    embeddings, topic_ids, topic_sizes = golden_case()
    mapping = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=5)
    assert mapping == GOLDEN_Q2


def test_repeated_calls_are_deterministic():
    embeddings, topic_ids, topic_sizes = golden_case()
    first = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=5)
    second = VendiClustering().cluster(embeddings, topic_ids, topic_sizes, target_k=5)
    assert first == second
