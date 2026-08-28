"""Vendi Clustering for arbitrary order q, via eigendecomposition.

Supports any q > 0, including q=1 (Shannon entropy) and q=inf.
For q=2 prefer `VendiClustering` in `clustering.py`.
"""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics.pairwise import cosine_similarity

from ._common import prepare_inputs, progress_bar, resolve_target_k


class GeneralVendiClustering:
    """Vendi Clustering for arbitrary order q.

    Evaluates each candidate merge by building the merged similarity matrix and
    taking its eigenvalues. Total complexity is O(m^5).
    """

    def __init__(self, q: float = 1.0, verbose: bool = False):
        if q <= 0:
            raise ValueError(f"q must be > 0, got {q}")
        self.q = q
        self.verbose = verbose

    def compute_vendi_q_score(self, K: np.ndarray) -> float:
        """Compute VS_q from similarity matrix K by eigendecomposition.

        Arguments:
            K: Similarity matrix (m x m) with unit diagonal.

        Returns:
            Vendi Score of order q.
        """
        n = K.shape[0]
        eigenvalues = np.linalg.eigvalsh(K / n)
        eigenvalues = np.maximum(eigenvalues, 0.0)

        q = self.q

        if q == float("inf"):
            lam_max = np.max(eigenvalues)
            return 1.0 / lam_max if lam_max > 0 else 0.0

        if q == 1.0:
            # Shannon entropy: VS_1 = exp(-sum(lambda * log(lambda)))
            pos = eigenvalues[eigenvalues > 1e-30]
            return float(np.exp(-np.sum(pos * np.log(pos))))

        # General q: VS_q = (sum(lambda^q))^(1/(1-q)), in log space for stability.
        pos = eigenvalues[eigenvalues > 1e-30]
        if len(pos) == 0:
            return 0.0
        log_terms = q * np.log(pos)
        log_sum = logsumexp(log_terms)
        return float(np.exp(log_sum / (1.0 - q)))

    def _build_merged_matrix(
        self, K: np.ndarray, a: int, b: int, n_a: float, n_b: float
    ) -> np.ndarray:
        """Construct the (m-1)x(m-1) similarity matrix after merging topics a and b."""
        m = K.shape[0]
        K_ab = K[a, b]
        c = np.sqrt(n_a**2 + n_b**2 + 2 * n_a * n_b * K_ab)

        # New similarity row via weighted combination
        k_u = (n_a * K[a] + n_b * K[b]) / c
        k_u[a] = 1.0  # self-similarity

        # Remove row/col b, replace row/col a with merged
        mask = np.ones(m, dtype=bool)
        mask[b] = False

        K_new = K[np.ix_(mask, mask)]
        # After masking, index a stays at a if a < b, else shifts to a-1
        new_a = a if a < b else a - 1
        K_new[new_a, :] = k_u[mask]
        K_new[:, new_a] = k_u[mask]

        return K_new

    def find_best_merge(self, K: np.ndarray, n_vec: np.ndarray, current_score: float):
        """Find the merge pair that maximizes the Vendi Score.

        Scans only pairs with b > a, so the returned `idx_a` is always the lower
        index and, given ascending-ID ordering, the lower topic ID.

        Returns:
            (best_delta, idx_a, idx_b)
        """
        m = K.shape[0]
        best_delta = -np.inf
        best_a, best_b = 0, 1

        for a in range(m):
            for b in range(a + 1, m):
                K_merged = self._build_merged_matrix(K, a, b, n_vec[a], n_vec[b])
                new_score = self.compute_vendi_q_score(K_merged)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_a, best_b = a, b

        return best_delta, best_a, best_b

    def apply_merge(
        self,
        K: np.ndarray,
        idx_a: int,
        idx_b: int,
        n_a: float,
        n_b: float,
        topic_embeddings: np.ndarray,
        active_topics: list,
        topic_sizes: dict,
    ):
        """Apply a merge and return updated state."""
        K_new = self._build_merged_matrix(K, idx_a, idx_b, n_a, n_b)

        # Update embeddings (weighted centroid, unnormalized for storage)
        topic_embeddings[idx_a] = (
            n_a * topic_embeddings[idx_a] + n_b * topic_embeddings[idx_b]
        ) / (n_a + n_b)
        topic_sizes[active_topics[idx_a]] = n_a + n_b

        topic_embeddings_new = np.delete(topic_embeddings, idx_b, axis=0)
        active_topics_new = [t for i, t in enumerate(active_topics) if i != idx_b]

        return K_new, active_topics_new, topic_embeddings_new

    def cluster(
        self,
        embeddings: np.ndarray,
        topic_ids: Sequence[int],
        topic_sizes: Mapping[int, float],
        target_k: Optional[int] = None,
        progress: bool = False,
    ) -> Dict[int, int]:
        """Greedily merge topics until `target_k` remain, maximizing VS_q.

        Arguments:
            embeddings: Topic embedding matrix; row i belongs to `topic_ids[i]`.
            topic_ids: Topic ID per row of `embeddings`. Exclude the outlier topic.
            topic_sizes: Document count per topic ID. Not modified.
            target_k: Number of topics to stop at. `None` merges down to one.
            progress: Show a progress bar if tqdm is installed.

        Returns:
            Mapping from every input topic ID to the ID that absorbed it. Each
            group's surviving ID is the lowest ID in that group.
        """
        topic_embeddings, active_topics, topic_sizes = prepare_inputs(
            embeddings, topic_ids, topic_sizes
        )
        m = len(active_topics)
        target_k = resolve_target_k(target_k, m)

        K = cosine_similarity(topic_embeddings)
        current_score = self.compute_vendi_q_score(K)

        q_label = "inf" if self.q == float("inf") else self.q

        if self.verbose:
            print(f"Initial: {m} topics, VS_q={current_score:.4f} (q={q_label})")

        cumulative_mapping = {t: t for t in active_topics}

        iteration = 0
        total_merges = m - target_k if target_k is not None else m - 1

        with progress_bar(total_merges, f"    Vendi clustering (q={q_label})", progress) as pbar:
            while True:
                m = K.shape[0]

                if target_k is not None and m <= target_k:
                    break
                if m <= 1:
                    break

                n_vec = np.array([topic_sizes[t] for t in active_topics])
                best_delta, idx_a, idx_b = self.find_best_merge(K, n_vec, current_score)

                a = active_topics[idx_a]
                b = active_topics[idx_b]

                K, active_topics, topic_embeddings = self.apply_merge(
                    K,
                    idx_a,
                    idx_b,
                    topic_sizes[a],
                    topic_sizes[b],
                    topic_embeddings,
                    active_topics,
                    topic_sizes,
                )

                current_score = self.compute_vendi_q_score(K)

                for t in cumulative_mapping:
                    if cumulative_mapping[t] == b:
                        cumulative_mapping[t] = a

                iteration += 1
                pbar.update(1)

                if self.verbose and iteration % 10 == 0:
                    print(
                        f"Iter {iteration}: {K.shape[0]} topics, "
                        f"dVendi={best_delta:.6f}, "
                        f"VS_q={current_score:.4f}"
                    )

        if self.verbose:
            print(f"Final: {len(active_topics)} topics after {iteration} merges")

        return cumulative_mapping
