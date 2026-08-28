"""Vendi Clustering at order q=2, with algebraic acceleration.

Merges topics so as to preserve the Vendi_2 diversity of the topic set.
For q != 2 see `clustering_general`.
"""

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ._common import prepare_inputs, progress_bar, resolve_target_k


class VendiClustering:
    """Vendi Clustering at q=2 with lookahead algebraic acceleration.

    Uses the Frobenius-norm identity VS_2(K) = m^2 / ||K||_F^2 to score every
    candidate merge in closed form.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def compute_vendi2_score(self, T: float, m: int) -> float:
        """VS_2 from the squared Frobenius norm `T` of an m x m similarity matrix."""
        return (m**2) / T if T > 0 else 0.0

    def initialize_cache(self, K: np.ndarray):
        """Cache the quantities the delta formula needs.

        T = ||K||_F^2, R = row-wise squared sums excluding the diagonal, G = K K^T.
        """
        T = np.sum(K**2)
        R = np.sum(K**2, axis=1) - np.diag(K) ** 2
        G = K @ K.T
        return T, R, G

    def compute_all_deltas_vectorized(
        self, K: np.ndarray, T: float, R: np.ndarray, G: np.ndarray, n_vec: np.ndarray
    ):
        """Score every candidate merge at once.

        Only the strict upper triangle is considered, so the returned `idx_a` is
        always less than `idx_b`. Since topics are held in ascending-ID order, the
        surviving topic of a merge is always the lower ID.

        Returns:
            (best_delta, best_T_new, idx_a, idx_b)
        """
        m = K.shape[0]

        n_sq = n_vec**2
        n_outer = np.outer(n_vec, n_vec)
        c2 = n_sq[:, None] + n_sq[None, :] + 2 * n_outer * K

        # Guard against division by zero/negative from floating point drift.
        c2 = np.maximum(c2, 1e-10)

        R_col = R[:, None]
        R_row = R[None, :]

        Ra_prime = R_col - K**2
        Rb_prime = R_row - K**2
        Gab_prime = G - 2 * K

        numerator = (
            (n_sq[:, None] * Ra_prime) + (n_sq[None, :] * Rb_prime) + (2 * n_outer * Gab_prime)
        )

        sum_u_sq = numerator / c2

        T_new_matrix = T - 2 * R_col - 2 * R_row + 2 * (K**2) + 2 * sum_u_sq - 1

        vendi_old = (m**2) / T
        vendi_new_matrix = ((m - 1) ** 2) / T_new_matrix
        deltas = vendi_new_matrix - vendi_old

        # Mask the diagonal and lower triangle: no self-merges, no duplicate pairs.
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        deltas_upper = np.where(mask, deltas, -np.inf)

        idx_flat = np.argmax(deltas_upper)
        idx_a, idx_b = np.unravel_index(idx_flat, deltas.shape)

        return deltas[idx_a, idx_b], T_new_matrix[idx_a, idx_b], idx_a, idx_b

    def apply_merge(
        self,
        K: np.ndarray,
        T: float,
        R: np.ndarray,
        G: np.ndarray,
        idx_a: int,
        idx_b: int,
        n_a: float,
        n_b: float,
        topic_embeddings: np.ndarray,
        active_topics: list,
        topic_sizes: dict,
    ):
        """Merge topic `idx_b` into `idx_a` and refresh the cached quantities."""
        m = K.shape[0]
        K_ab = K[idx_a, idx_b]
        c = np.sqrt(n_a**2 + n_b**2 + 2 * n_a * n_b * K_ab)

        k_u = (n_a * K[idx_a] + n_b * K[idx_b]) / c
        k_u[idx_a] = 1.0

        mask = np.ones(m, dtype=bool)
        mask[idx_b] = False

        # idx_a < idx_b always, so dropping idx_b leaves idx_a's position unchanged.
        K_new = K[np.ix_(mask, mask)]
        K_new[idx_a, :] = k_u[mask]
        K_new[:, idx_a] = k_u[mask]

        topic_embeddings[idx_a] = (
            n_a * topic_embeddings[idx_a] + n_b * topic_embeddings[idx_b]
        ) / (n_a + n_b)
        topic_sizes[active_topics[idx_a]] = n_a + n_b

        T_new = np.sum(K_new**2)
        R_new = np.sum(K_new**2, axis=1) - np.diag(K_new) ** 2

        k_a = K[:, idx_a]
        k_b = K[:, idx_b]
        G_new = G - np.outer(k_a, k_a) - np.outer(k_b, k_b) + np.outer(k_u, k_u)
        G_new = G_new[np.ix_(mask, mask)]

        topic_embeddings_new = np.delete(topic_embeddings, idx_b, axis=0)
        active_topics_new = [t for i, t in enumerate(active_topics) if i != idx_b]

        return K_new, T_new, R_new, G_new, active_topics_new, topic_embeddings_new

    def cluster(
        self,
        embeddings: np.ndarray,
        topic_ids: Sequence[int],
        topic_sizes: Mapping[int, float],
        target_k: Optional[int] = None,
        progress: bool = False,
    ) -> Dict[int, int]:
        """Greedily merge topics until `target_k` remain.

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
        T, R, G = self.initialize_cache(K)
        vendi_score = self.compute_vendi2_score(T, m)

        if self.verbose:
            print(f"Initial: {m} topics, Vendi2={vendi_score:.4f}")

        cumulative_mapping = {t: t for t in active_topics}

        iteration = 0
        total_merges = m - target_k if target_k is not None else m - 1

        with progress_bar(total_merges, "    Vendi clustering", progress) as pbar:
            while True:
                m = K.shape[0]

                if target_k is not None and m <= target_k:
                    break
                if m <= 1:
                    break

                n_vec = np.array([topic_sizes[t] for t in active_topics])
                best_delta, best_T_new, idx_a, idx_b = self.compute_all_deltas_vectorized(
                    K, T, R, G, n_vec
                )

                a = active_topics[idx_a]
                b = active_topics[idx_b]

                K, T, R, G, active_topics, topic_embeddings = self.apply_merge(
                    K,
                    T,
                    R,
                    G,
                    idx_a,
                    idx_b,
                    topic_sizes[a],
                    topic_sizes[b],
                    topic_embeddings,
                    active_topics,
                    topic_sizes,
                )

                for t in cumulative_mapping:
                    if cumulative_mapping[t] == b:
                        cumulative_mapping[t] = a

                iteration += 1
                pbar.update(1)
                vendi_score = self.compute_vendi2_score(T, K.shape[0])

                if self.verbose and iteration % 10 == 0:
                    print(
                        f"Iter {iteration}: {K.shape[0]} topics, "
                        f"dVendi={best_delta:.6f}, "
                        f"Vendi2={vendi_score:.4f}"
                    )

        if self.verbose:
            print(f"Final: {len(active_topics)} topics after {iteration} merges")

        return cumulative_mapping
