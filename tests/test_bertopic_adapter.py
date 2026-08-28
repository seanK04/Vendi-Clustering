"""Tests for the BERTopic adapter."""

import numpy as np

from vendi_clustering.bertopic_adapter import mapping_to_merge_groups


def emulate_bertopic_representative(group):
    """How BERTopic's merge_topics picks the surviving topic: `topic_group[0]`."""
    return group[0]


def test_groups_are_sorted_ascending():
    """merge_topics keeps group[0], so ordering decides the surviving topic."""
    mapping = {5: 1, 1: 1, 9: 1, 3: 3, 7: 3}
    for group in mapping_to_merge_groups(mapping):
        assert group == sorted(group)


def test_representative_survives_the_handoff_to_merge_topics():
    """The invariant the equivalence gate rests on.

    Clustering collapses each group onto its lowest topic ID; BERTopic keeps
    `topic_group[0]`. Passing groups ascending is what makes those agree.
    """
    mapping = {5: 1, 1: 1, 9: 1, 3: 3, 7: 3, 4: 4}

    for group in mapping_to_merge_groups(mapping):
        assert emulate_bertopic_representative(group) == mapping[group[0]]
        assert emulate_bertopic_representative(group) == min(group)


def test_singleton_groups_are_dropped():
    """Topics that merged with nothing need no entry; merge_topics leaves them be."""
    assert mapping_to_merge_groups({0: 0, 1: 1, 2: 1}) == [[1, 2]]


def test_every_merged_topic_appears_exactly_once():
    mapping = {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 0}
    flattened = [t for group in mapping_to_merge_groups(mapping) for t in group]
    assert sorted(flattened) == [0, 1, 2, 3, 5]


def test_group_ids_are_plain_ints():
    """numpy integers would send merge_topics down its non-Iterable branch."""
    mapping = {np.int64(4): np.int64(2), np.int64(2): np.int64(2)}
    groups = mapping_to_merge_groups(mapping)
    assert all(type(topic) is int for group in groups for topic in group)
